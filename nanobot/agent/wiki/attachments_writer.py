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

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.wiki.vault import Vault

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
    if classify_extension(resolved_src.suffix) == "binary":
        # Manifest recording for binaries happens in Task 2d alongside textual.
        return WriteResult(status="skipped_binary")
    raise NotImplementedError("textual path: Task 2c")
