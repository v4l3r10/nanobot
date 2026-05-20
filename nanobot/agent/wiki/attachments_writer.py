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
