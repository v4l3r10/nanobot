"""One-time legacy memory/profile migration into a per-user wiki vault (Task 7.1).

Task 6.1 made the wiki-on prompt read the per-user vault MOC + vault
``USER.md`` with **no global fallback**. A freshly wiki-enabled user would
therefore lose both their accumulated memory and their profile (only
SOUL/AGENTS/TOOLS would remain) unless their LEGACY global memory is
bootstrapped into the vault. :func:`migrate_legacy` is that one-shot
bootstrap, run from the Dream 4.5 wiki block via
:meth:`Vault.ensure_initialized` so the FIRST wiki-enabled Dream cycle
migrates, then Lint builds the MOC — closing the enable-ordering window the
6.1 review flagged.

Properties (all load-bearing):

* **Pure filesystem, NO LLM, deterministic, idempotent.** One-shot per
  vault via a ``vault.root/.migrated`` marker.
* **Legacy sources are read-only.** ``<workspace>/memory/MEMORY.md`` and
  ``<workspace>/USER.md`` (the exact paths
  :class:`~nanobot.agent.memory.MemoryStore` resolves) are NEVER modified or
  deleted — git history floor / design no-hard-delete.
* **Template guard.** A blank / absent / stock-template ``MEMORY.md`` is NOT
  imported as if it were real memory (reuses the SAME bundled-template
  detection :class:`~nanobot.agent.context.ContextBuilder` uses, via the
  shared leaf helper
  :func:`nanobot.utils.prompt_templates.is_bundled_template_content` — no
  divergent reimplementation).
* **Lint-contract page.** Real memory becomes ONE ``concepts`` page that
  satisfies the 4.3 Lint contract (parseable, ``status="hot"``, ISO today
  dates, schema-known type, never under ``.cold/``) so the next Lint indexes
  it into ``concepts/_index.md`` + the root MOC.

This module is a **leaf**: it imports only ``wiki/page``, ``utils/atomic``
and ``utils/prompt_templates`` (plus stdlib + loguru). It NEVER imports
``memory.py`` / ``context.py`` / ``loop.py`` — :class:`Vault` is referenced
only as a typing annotation under ``TYPE_CHECKING`` — so the
``vault.py -> migrate.py`` wiring edge introduces no import cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.prompt_templates import is_bundled_template_content

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import (no cycle)
    import datetime

    from nanobot.agent.wiki.vault import Vault

__all__ = ["migrate_legacy"]

# Vault-internal one-shot marker. Deliberately at the vault ROOT (NOT under
# wiki/) so it is not a content page and Lint ignores it; it IS under
# memory/users/<slug>/ so Task 5.1's GitStore versions it — a desirable audit
# trail of when the bootstrap ran.
_MARKER_NAME = ".migrated"

# Bundled stock MEMORY.md, relative to nanobot/templates/. Passed verbatim to
# the shared is_bundled_template_content() helper (the SAME path
# ContextBuilder._is_template_content uses for its template guard).
_LEGACY_MEMORY_TEMPLATE = "memory/MEMORY.md"

# The wiki type the imported legacy memory is filed under. Schema-known in the
# bundled master; a deterministic fallback is used if a custom vault SCHEMA
# omits it (see _import_memory_type).
_PREFERRED_TYPE = "concepts"
_IMPORTED_SLUG = "imported-memory"


def _read_text_or_empty(path: Path) -> str:
    """Read *path* as UTF-8, returning "" for a missing/odd file (never raises)."""
    try:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("legacy migration: failed reading {}", path)
        return ""


def _today_iso(today: datetime.date | None) -> str:
    import datetime as _dt

    return (today or _dt.date.today()).isoformat()


def _import_memory_type(vault: Vault) -> str:
    """The schema-known type the imported memory page is filed under.

    Prefers ``concepts`` (schema-known in the bundled master). If a custom
    per-vault SCHEMA omits it, fall back DETERMINISTICALLY to the first
    schema-known type (sorted ascending) so the page still satisfies the 4.3
    Lint contract (schema-known type).
    """
    schema = vault.schema
    if schema.is_known_type(_PREFERRED_TYPE):
        return _PREFERRED_TYPE
    # Deterministic fallback: first type name in sorted order.
    fallback = sorted(schema.types)[0]
    logger.warning(
        "legacy migration: schema lacks {!r}; filing imported memory under "
        "{!r} (first schema-known type)",
        _PREFERRED_TYPE,
        fallback,
    )
    return fallback


def _migrate_memory(workspace: Path, vault: Vault, today_iso: str) -> None:
    """Import a real (non-template, non-blank) legacy MEMORY.md as ONE page.

    No page is written for an absent / blank / stock-template MEMORY.md (the
    template guard reuses the shared bundled-template detection). If the
    target page already exists it is left untouched (never overwritten).
    """
    legacy_memory = workspace / "memory" / "MEMORY.md"
    text = _read_text_or_empty(legacy_memory)
    if not text.strip():
        return  # absent or blank — nothing real to import
    if is_bundled_template_content(text, _LEGACY_MEMORY_TEMPLATE):
        return  # stock template — not real user memory

    type_ = _import_memory_type(vault)
    page_path = vault.page_path(type_, _IMPORTED_SLUG)
    if page_path.exists():
        # The .migrated marker guards re-runs; this is pure defense in depth.
        logger.debug(
            "legacy migration: {} already exists; not overwriting", page_path
        )
        return

    page = Page(
        type=type_,
        title="Imported legacy memory",
        status="hot",
        created=today_iso,
        updated=today_iso,
        last_touched=today_iso,
        tags=["migrated"],
        links_out=[],
        pinned=None,
        body=text,
    )
    atomic_write_text(page_path, serialize_page(page))
    logger.info("legacy migration: imported MEMORY.md -> {}", page_path)


def _migrate_user(workspace: Path, vault: Vault) -> None:
    """Copy a non-blank legacy root USER.md to ``vault.root/USER.md``.

    This is exactly the file Task 6.1 reads as the per-user profile. An
    EXISTING non-blank vault USER.md is never clobbered (the user already has
    profile data); a blank/absent legacy USER.md writes nothing.
    """
    legacy_user = workspace / "USER.md"
    text = _read_text_or_empty(legacy_user)
    if not text.strip():
        return  # absent or blank — no profile to migrate

    vault_user = vault.root / "USER.md"
    existing = _read_text_or_empty(vault_user)
    if existing.strip():
        logger.debug(
            "legacy migration: vault USER.md already populated; not clobbering"
        )
        return

    atomic_write_text(vault_user, text)
    logger.info("legacy migration: copied USER.md -> {}", vault_user)


def migrate_legacy(
    workspace: Path,
    vault: Vault,
    *,
    today: datetime.date | None = None,
) -> bool:
    """One-shot migrate LEGACY global memory/profile into ``vault``.

    Sources (read-only, NEVER modified/deleted): ``workspace/memory/MEMORY.md``
    and ``workspace/USER.md`` — the exact paths
    :class:`~nanobot.agent.memory.MemoryStore` resolves.

    Effects on first call for a given vault:

    * Real (non-blank, non-stock-template) ``MEMORY.md`` -> ONE wiki page at
      ``concepts/imported-memory.md`` satisfying the 4.3 Lint contract.
    * Non-blank legacy ``USER.md`` -> ``vault.root/USER.md`` (Task 6.1's
      profile read path), unless a non-blank vault ``USER.md`` already exists
      (never clobbered).
    * A ``vault.root/.migrated`` marker is written LAST so the migration is
      strictly one-shot per vault — even when there was nothing real to
      import (blank/template/missing legacy files).

    Return contract: ``True`` iff this call PERFORMED the one-shot migration
    attempt (i.e. created the ``.migrated`` marker this call); ``False`` iff
    it short-circuited because the marker already existed (already migrated —
    never re-run). Never raises on missing/odd legacy files; on an unexpected
    ``OSError`` the failure is logged and the marker is still written so the
    bootstrap cannot loop forever (fail-safe, never corrupting the vault or
    touching the legacy sources).

    ``today`` (default: ``datetime.date.today()``) is injectable for
    deterministic testing of the ISO date stamps.
    """
    workspace = Path(workspace)
    marker = vault.root / _MARKER_NAME

    # Already migrated: never re-run, do nothing, report False.
    try:
        if marker.exists():
            return False
    except OSError:  # pragma: no cover - defensive (stat failure)
        logger.exception("legacy migration: marker stat failed; treating as fresh")

    today_iso = _today_iso(today)
    try:
        _migrate_memory(workspace, vault, today_iso)
        _migrate_user(workspace, vault)
    except OSError:
        # Fail-safe: log and STILL set the marker below so a transient FS
        # error can't loop the bootstrap forever. The legacy sources are
        # read-only here, so they remain byte-intact regardless.
        logger.exception(
            "legacy migration: unexpected error; marking migrated to avoid a loop"
        )

    # Marker LAST so the migration is one-shot even if nothing was imported.
    atomic_write_text(marker, f"{today_iso}\n")
    return True
