"""Deterministic attachment -> wiki page writer (no LLM).

Called from two sites that share the same per-vault lock:

* the eager hook in :class:`~nanobot.agent.loop.AgentLoop` — fires the moment
  a channel publishes an :class:`InboundMessage` with non-empty ``media``;
* the Dream-side reconciler — walks conventional workspace directories on
  each Dream cycle and writes anything not yet in the per-vault manifest.

Both ultimately call :func:`write_attachment_page` which is a pure
function of (source file bytes, vault state, manifest). Idempotence is
keyed on ``sha256(content)`` recorded in the per-vault manifest; rerunning
a write for the same bytes is a no-op.

The writer reuses the slug/containment/body-clamp primitives from
:mod:`nanobot.agent.wiki.ingest` so it cannot diverge from the model-side
ingest pipeline on these invariants (C1/C2 in ingest.py).
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.wiki.ingest import (
    _MAX_BODY_CHARS,
    _body_already_present,
    _safe_slug,
    _slug_ok,
)
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import Vault
from nanobot.utils.atomic import atomic_write_text

__all__ = ["WriteResult", "classify_extension", "write_attachment_page"]

_TEXTUAL_EXTS = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv"})


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a single ``write_attachment_page`` call.

    ``status`` is one of:

    * ``"created"`` — new page written
    * ``"appended"`` — page existed, body appended (C2 guard passed)
    * ``"duplicate"`` — sha256 already in manifest OR body already present
    * ``"skipped_binary"`` — non-textual extension
    * ``"error"`` — containment failure, read error, etc. ``reason`` set.

    ``page_rel`` is the vault-relative POSIX path written, or ``None`` for
    skipped/error/duplicate-without-page outcomes.
    """

    status: str
    page_rel: str | None = None
    reason: str | None = None


def classify_extension(ext: str) -> str:
    """Return ``"textual"`` or ``"binary"`` for a lowercased extension."""
    return "textual" if ext.lower() in _TEXTUAL_EXTS else "binary"


# --------------------------------------------------------------------------- #
# Manifest — per-vault dedup ledger keyed by sha256(content).
#
# Lives at ``vault.wiki_dir / ".ingested_attachments.json"`` so it is
# isolated per user and migrates naturally with the wiki tree. The dot
# prefix keeps Lint's ``rglob("*.md")`` from ever seeing it.
# --------------------------------------------------------------------------- #
_MANIFEST_NAME = ".ingested_attachments.json"


def _sha256_of(path: Path) -> str:
    """Streaming sha256 of ``path``'s bytes (chunked, no full read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(vault: Vault) -> Path:
    return vault.wiki_dir / _MANIFEST_NAME


def _load_manifest(vault: Vault) -> dict[str, Any]:
    """Load the manifest, returning a fresh empty one on missing/corrupt."""
    path = _manifest_path(vault)
    if not path.exists():
        return {"version": 1, "entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("malformed manifest")
        if not isinstance(data.get("entries"), list):
            raise ValueError("malformed manifest entries")
        # Normalise version if missing/old.
        data.setdefault("version", 1)
        return data
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning("attachments manifest corrupt ({}); starting fresh", e)
        return {"version": 1, "entries": []}


def _save_manifest(vault: Vault, data: dict[str, Any]) -> None:
    atomic_write_text(
        _manifest_path(vault),
        json.dumps(data, indent=2, ensure_ascii=False),
    )


def _sha256_in_manifest(data: dict[str, Any], sha: str) -> dict[str, Any] | None:
    for entry in data.get("entries", []):
        if entry.get("sha256") == sha:
            return entry
    return None


def _append_entry(
    data: dict[str, Any],
    *,
    sha: str,
    channel: str,
    msg_id: str,
    status: str,
    page: str | None,
    path: str,
    size: int,
    ingested_at: str,
) -> None:
    entry: dict[str, Any] = {
        "sha256": sha,
        "channel": channel,
        "msg_id": msg_id,
        "status": status,
        "path": path,
        "size": size,
        "ingested_at": ingested_at,
    }
    if page is not None:
        entry["page"] = page
    data.setdefault("entries", []).append(entry)


def write_attachment_page(
    vault: Vault,
    slug: str,
    src_path: Path,
    channel: str,
    msg_id: str,
    allowed_roots: list[Path],
) -> WriteResult:
    """Write a single attachment file as a wiki ``inbox`` page.

    Callers MUST already hold ``get_vault_lock(slug)``. The function is sync
    (no I/O that benefits from awaiting); making it sync keeps it usable
    from both async (loop.py hook) and sync (Dream-side reconciler when
    called inside an existing async context) call sites consistently.

    ``allowed_roots`` is the list of containment roots — typically
    ``[get_workspace_path(), get_media_dir()]`` so files under either the
    workspace (peer) or the data dir (telegram, future channels) are
    accepted. A path under NONE of the roots is treated as an escape attempt.
    """
    try:
        resolved_src = src_path.resolve()
    except OSError as e:
        return WriteResult(status="error", reason=f"resolve: {e}")
    allowed_resolved = [r.resolve() for r in allowed_roots]
    if not any(is_under(resolved_src, r) for r in allowed_resolved):
        return WriteResult(status="error", reason="path escapes allowed_roots")
    if not resolved_src.is_file():
        return WriteResult(status="error", reason="not a regular file")

    # Manifest is the cheap first dedup gate, keyed on sha256(content).
    # We load once at the top and save once at the bottom (with a single
    # mutation in between).
    manifest = _load_manifest(vault)
    try:
        sha = _sha256_of(resolved_src)
    except OSError as e:
        return WriteResult(status="error", reason=f"sha256: {e}")
    existing = _sha256_in_manifest(manifest, sha)
    if existing is not None:
        # Same bytes already ingested (or skipped) — no read of file body,
        # no manifest update, byte-stable rerun.
        return WriteResult(
            status="duplicate",
            page_rel=existing.get("page"),
        )

    # Compute provenance fields once — reused across all manifest writes and
    # the page body's ``Source:`` header so they stay consistent for a single
    # ingest call.
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        src_size = resolved_src.stat().st_size
    except OSError as e:
        return WriteResult(status="error", reason=f"stat: {e}")
    src_abs = str(resolved_src)

    if classify_extension(resolved_src.suffix) == "binary":
        _append_entry(
            manifest,
            sha=sha,
            channel=channel,
            msg_id=msg_id,
            status="skipped_binary",
            page=None,
            path=src_abs,
            size=src_size,
            ingested_at=ts,
        )
        _save_manifest(vault, manifest)
        return WriteResult(status="skipped_binary")

    raw_slug = resolved_src.stem
    slug_safe = _safe_slug(raw_slug)
    if not _slug_ok(raw_slug, slug_safe):
        return WriteResult(status="error", reason=f"invalid slug: {raw_slug!r}")

    if not vault.schema.is_known_type("inbox"):
        return WriteResult(status="error", reason="vault schema lacks 'inbox' type")

    try:
        text = resolved_src.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return WriteResult(status="error", reason=f"read: {e}")

    clamped = text if len(text) <= _MAX_BODY_CHARS else text[:_MAX_BODY_CHARS]
    body = f"Source: {channel}/{msg_id} · {ts}\n\n{clamped}"

    target = vault.page_path("inbox", slug_safe).resolve()
    if not is_under(target, vault.wiki_dir):
        return WriteResult(status="error", reason="target escapes vault")

    today = datetime.date.today().isoformat()
    rel = f"inbox/{slug_safe}.md"

    if target.exists():
        try:
            page = parse_page(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            return WriteResult(status="error", reason=f"parse existing: {e}")
        if _body_already_present(page.body, body):
            # Defense-in-depth: sha256 missed (different framing) but the
            # body is already there. Still record the manifest entry so a
            # rerun goes through the cheap sha gate next time.
            _append_entry(
                manifest,
                sha=sha,
                channel=channel,
                msg_id=msg_id,
                status="ingested",
                page=rel,
                path=src_abs,
                size=src_size,
                ingested_at=ts,
            )
            _save_manifest(vault, manifest)
            return WriteResult(status="duplicate", page_rel=rel)
        prefix = page.body.rstrip("\n")
        page.body = f"{prefix}\n\n{body}" if prefix else body
        page.updated = today
        page.last_touched = today
        for t in (channel, msg_id):
            if t not in page.tags:
                page.tags.append(t)
        atomic_write_text(target, serialize_page(page))
        _append_entry(
            manifest,
            sha=sha,
            channel=channel,
            msg_id=msg_id,
            status="ingested",
            page=rel,
            path=src_abs,
            size=src_size,
            ingested_at=ts,
        )
        _save_manifest(vault, manifest)
        return WriteResult(status="appended", page_rel=rel)

    page = Page(
        type="inbox",
        title=(resolved_src.stem[:120] or slug_safe),
        status="hot",
        created=today,
        updated=today,
        last_touched=today,
        tags=[channel, msg_id],
        links_out=[],
        pinned=None,
        body=body,
    )
    atomic_write_text(target, serialize_page(page))
    _append_entry(
        manifest,
        sha=sha,
        channel=channel,
        msg_id=msg_id,
        status="ingested",
        page=rel,
        path=src_abs,
        size=src_size,
        ingested_at=ts,
    )
    _save_manifest(vault, manifest)
    return WriteResult(status="created", page_rel=rel)
