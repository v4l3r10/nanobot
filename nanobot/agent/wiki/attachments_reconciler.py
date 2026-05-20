"""Dream-side reconciler: walk conventional workspace dirs and write
attachments not yet in the per-vault manifest.

Symmetric companion to the eager hook in :class:`AgentLoop`: both call
:func:`~nanobot.agent.wiki.attachments_writer.write_attachment_page`, both
acquire ``get_vault_lock(slug)`` (the caller does, not this module), both
share the ``.ingested_attachments.json`` manifest. The reconciler catches
files that arrived before the eager hook was deployed, files for which the
eager hook failed, and files dropped into the workspace out-of-band (e.g.
by Umanio / peer delivery that bypasses the nanobot bus).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from loguru import logger

__all__ = ["ReconcileReport", "run_attachments_reconcile"]


@dataclass(frozen=True)
class _DiscoveredAttachment:
    """One file discovered by a source iterator."""
    path: Path
    channel: str
    msg_id: str


@dataclass
class ReconcileReport:
    """Outcome of one reconcile sweep. Sums the per-file results from
    every source root walked this cycle."""
    created: list[str] = field(default_factory=list)
    appended: list[str] = field(default_factory=list)
    duplicates: int = 0
    skipped_binary: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _iter_peer_files(peer_root: Path) -> Iterator[_DiscoveredAttachment]:
    """Walk ``<peer_root>/<msg_subdir>/<files>`` — peer.py uses
    ``safe_filename(msg_id)`` as the subdir name (NOT a literal ``msg_``
    prefix). The bronzo-v0.2.0 base prunes empty subdirs, so every subdir
    we see should contain at least one file. Dot-prefixed subdirs / files
    are skipped (hidden / future markers).
    """
    if not peer_root.is_dir():
        return
    for msg_dir in sorted(peer_root.iterdir()):
        if not msg_dir.is_dir() or msg_dir.name.startswith("."):
            continue
        for f in sorted(msg_dir.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            yield _DiscoveredAttachment(path=f, channel="peer", msg_id=msg_dir.name)


def _iter_flat_files(directory: Path, channel: str) -> Iterator[_DiscoveredAttachment]:
    """Walk a flat directory of files (Telegram-style layout). Subdirs
    and dot-prefixed files are skipped. ``msg_id`` defaults to the file
    stem since flat layouts have no embedded msg_id structure."""
    if not directory.is_dir():
        return
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        yield _DiscoveredAttachment(path=f, channel=channel, msg_id=f.stem)
