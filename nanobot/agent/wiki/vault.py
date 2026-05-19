"""Read-only accessor over a per-user wiki vault directory.

A *vault* is the ``<workspace>/memory/users/<slug>`` directory produced by
:func:`nanobot.agent.wiki.paths.vault_dir`. Its ``wiki/`` subtree holds the
Markdown pages, an optional per-vault ``SCHEMA.md``, and Map-of-Content
``_index.md`` files.

:class:`Vault` is the read facade used by the ``wiki_note`` tool, Ingest,
Lint, and MOC injection. It performs **no writes** -- creation, cold-move,
and admission-gating live in later milestones.

The schema resolves per-vault first, falling back to the bundled master at
``nanobot/templates/memory/wiki/SCHEMA.md`` so a freshly created vault (no
``SCHEMA.md`` yet) can still resolve types and decay policy. The bundled
path is derived from the same templates-root the prompt renderer uses
(:data:`nanobot.utils.prompt_templates._TEMPLATES_ROOT`) rather than
hardcoded ``__file__`` arithmetic, keeping the two in lockstep.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.wiki.page import Page, parse_page
from nanobot.agent.wiki.schema import Schema, load_schema
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.prompt_templates import _TEMPLATES_ROOT

__all__ = ["Vault"]

# Bundled master SCHEMA.md, resolved off the shared templates root so it
# tracks any relocation of nanobot/templates/ automatically.
_BUNDLED_SCHEMA = _TEMPLATES_ROOT / "memory" / "wiki" / "SCHEMA.md"

# Files in the wiki tree that are not content pages.
_NON_PAGE_NAMES = {"SCHEMA.md", "_index.md"}

# Path component marking the cold archive subtree.
_COLD_COMPONENT = ".cold"


class Vault:
    """Read-only view over a single user's wiki vault.

    ``vault_root`` is ``<workspace>/memory/users/<slug>``. The vault need
    not exist on disk; accessors degrade gracefully (the schema falls back
    to the bundled master, iteration over a missing ``wiki/`` yields
    nothing).
    """

    def __init__(self, vault_root: Path) -> None:
        self.root: Path = Path(vault_root)
        self.wiki_dir: Path = self.root / "wiki"

    def ensure_initialized(
        self, legacy_workspace: Path | None = None
    ) -> None:
        """Idempotently materialize the vault's ``wiki/`` tree.

        Creates ``wiki_dir`` (``parents=True, exist_ok=True``) and, only if
        ``wiki/SCHEMA.md`` does NOT already exist, copies the bundled master
        SCHEMA into it. The bundled source is :data:`_BUNDLED_SCHEMA` -- the
        SAME templates-root-derived path :attr:`schema` resolves to (no
        hardcoded ``__file__`` arithmetic), so the two stay in lockstep.

        Idempotent: a second call writes nothing (an existing per-vault
        ``SCHEMA.md`` -- bundled-copied or hand-authored -- is never
        overwritten, so the cached :attr:`schema` and existing read-only
        behaviour are unaffected).

        Task 7.1 -- legacy migration: when ``legacy_workspace`` is provided
        (the Dream 4.5 wiki block passes ``self.store.workspace``), a one-time
        bootstrap of that workspace's LEGACY global ``memory/MEMORY.md`` +
        root ``USER.md`` into this vault is attempted AFTER the mkdir+SCHEMA
        copy, gated one-shot by :func:`migrate_legacy`'s own
        ``vault.root/.migrated`` marker. When ``legacy_workspace`` is ``None``
        (the default -- every existing caller / the 4.5 tests) this method is
        BYTE-IDENTICAL to before: mkdir ``wiki/`` + copy SCHEMA only, no
        migration, no ``.migrated`` marker. The migrate import is function
        -local so the leaf ``migrate`` module's ``Vault`` use stays
        typing-only and the ``vault -> migrate`` edge introduces no cycle.
        """
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        schema_path = self.wiki_dir / "SCHEMA.md"
        if not schema_path.exists():
            atomic_write_text(
                schema_path,
                _BUNDLED_SCHEMA.read_text(encoding="utf-8"),
            )
        if legacy_workspace is not None:
            from nanobot.agent.wiki.migrate import migrate_legacy

            migrate_legacy(Path(legacy_workspace), self)

    @cached_property
    def schema(self) -> Schema:
        """The vault's :class:`Schema`.

        Loads ``wiki/SCHEMA.md`` if present, otherwise the bundled master.
        Cached for the lifetime of the instance (the schema is immutable
        and a vault is short-lived per request).
        """
        per_vault = self.wiki_dir / "SCHEMA.md"
        source = per_vault if per_vault.exists() else _BUNDLED_SCHEMA
        return load_schema(source.read_text(encoding="utf-8"))

    def page_path(self, type: str, slug: str) -> Path:
        """Absolute path a page of ``type`` with ``slug`` would live at.

        Does not check existence. Raises :class:`KeyError` (via the schema)
        for an unknown type.
        """
        return self.wiki_dir / self.schema.folder(type) / f"{slug}.md"

    def read_page(self, rel: str) -> Page:
        """Parse the page at ``rel`` (relative to ``wiki_dir``).

        ``rel`` is resolved against ``wiki_dir`` and is required to stay
        inside it (path-traversal guard). Raises :class:`ValueError` if it
        escapes the vault, :class:`FileNotFoundError` if the file is
        absent, and propagates :class:`ValueError` from :func:`parse_page`
        for a malformed page.
        """
        target = (self.wiki_dir / rel).resolve()
        if not is_under(target, self.wiki_dir):
            raise ValueError(
                f"refusing to read {rel!r}: resolves outside the vault"
            )
        if not target.is_file():
            raise FileNotFoundError(str(target))
        return parse_page(target.read_text(encoding="utf-8"))

    def iter_pages(
        self, include_cold: bool = False
    ) -> Iterator[tuple[str, Page]]:
        """Yield ``(relpath, Page)`` for every content page under ``wiki_dir``.

        ``relpath`` is POSIX-style (forward slashes) relative to
        ``wiki_dir`` so callers can use it as a stable, cross-platform dict
        key. ``SCHEMA.md`` and ``_index.md`` are skipped; pages under a
        ``.cold`` path component are skipped unless ``include_cold=True``;
        a page that fails :func:`parse_page` is skipped silently (the Lint
        engine -- not this read accessor -- owns malformed pages).
        """
        if not self.wiki_dir.is_dir():
            return
        for path in sorted(self.wiki_dir.rglob("*.md")):
            if path.name in _NON_PAGE_NAMES:
                continue
            rel = path.relative_to(self.wiki_dir)
            parts = rel.parts
            if not include_cold and _COLD_COMPONENT in parts:
                continue
            try:
                page = parse_page(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            yield rel.as_posix(), page

    def is_empty(self) -> bool:
        """Whether the vault holds no content pages.

        ``SCHEMA.md``, ``_index.md``, and a missing ``wiki/`` directory do
        not count as content.
        """
        for _ in self.iter_pages(include_cold=True):
            return False
        return True
