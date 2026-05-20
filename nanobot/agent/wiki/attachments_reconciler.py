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

from nanobot.agent.wiki.attachments_writer import write_attachment_page
from nanobot.agent.wiki.vault import Vault
from nanobot.config.paths import get_media_dir, get_workspace_path

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


# Type alias for the source factory: takes a root Path, returns an iterator
# of discovered attachments.
_SourceIterator = Callable[[Path], Iterator[_DiscoveredAttachment]]
_Source = tuple[str, Path, _SourceIterator]


def _default_sources() -> list[_Source]:
    """Return the production source registry as ``(channel, root, iterator)``
    triples.

    Sources live in DIFFERENT roots: peer under the workspace, telegram
    under the data dir. The reconciler iterates each independently. The
    PeerConfig.media_subdir default is ``"peer"`` — if a deployment changes
    it the channel-side eager hook still works (it acts on ``msg.media``
    paths directly), but the reconciler would miss the renamed dir. v1
    hardcodes ``"peer"`` to match the bundled default; a follow-up branch
    can plumb it through from config if needed.
    """
    return [
        ("peer",
         get_workspace_path() / "peer",
         _iter_peer_files),
        ("telegram",
         get_media_dir("telegram"),
         lambda root: _iter_flat_files(root, channel="telegram")),
    ]


async def run_attachments_reconcile(
    vault: Vault,
    slug: str,
    sources: list[_Source] | None = None,
) -> ReconcileReport:
    """Walk all known sources, write anything not yet ingested.

    The caller holds ``get_vault_lock(slug)`` — same as ``run_ingest`` /
    ``run_lint``. This function is intentionally NOT acquiring the lock
    itself so it composes with the existing Dream loop block in
    ``memory.py`` (one lock per slug for all wiki work in the cycle).

    ``sources`` is injectable for tests; production callers omit it and get
    the default registry (peer + telegram). The containment allowlist for
    the writer is derived from the source roots: a path under NO source
    root is rejected by the writer (defense in depth — a symlink inside a
    source dir pointing elsewhere is caught).
    """
    report = ReconcileReport()
    if sources is None:
        sources = _default_sources()
    allowed_roots = [root for _, root, _ in sources]
    for channel_name, root, iterator_factory in sources:
        try:
            iterator = iterator_factory(root)
        except OSError as e:
            report.errors.append((channel_name, f"iterator: {e}"))
            continue
        for item in iterator:
            try:
                result = write_attachment_page(
                    vault, slug, item.path, item.channel, item.msg_id,
                    allowed_roots=allowed_roots,
                )
            except Exception as e:
                logger.exception("attachment write failed for {}", item.path)
                report.errors.append((str(item.path), str(e)))
                continue
            if result.status == "created":
                report.created.append(result.page_rel)
            elif result.status == "appended":
                report.appended.append(result.page_rel)
            elif result.status == "duplicate":
                report.duplicates += 1
            elif result.status == "skipped_binary":
                report.skipped_binary += 1
            elif result.status == "error":
                report.errors.append((str(item.path), result.reason or ""))
    return report
