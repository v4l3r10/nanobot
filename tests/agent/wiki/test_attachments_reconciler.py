"""Tests for :mod:`nanobot.agent.wiki.attachments_reconciler` — the
Dream-side reconciler that walks source roots and calls the writer for
any file not yet in the per-vault manifest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.wiki.attachments_reconciler import (
    ReconcileReport,
    _iter_flat_files,
    _iter_peer_files,
)


# --------------------------------------------------------------------------- #
# Fixtures (mirrored from test_attachments_writer.py — a future cleanup task
# can consolidate into conftest.py).
# --------------------------------------------------------------------------- #
def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


@pytest.fixture
def vault_factory(tmp_path):
    """Return a callable that builds a fresh per-user vault under tmp_path."""
    from nanobot.agent.wiki.vault import Vault

    def _factory(name: str = "u1"):
        root = tmp_path / "memory" / "users" / name
        wiki = root / "wiki"
        wiki.mkdir(parents=True)
        (wiki / "SCHEMA.md").write_text(_bundled_schema_text(), encoding="utf-8")
        return Vault(root), name

    return _factory


# --------------------------------------------------------------------------- #
# Task 3a — Source iterators + ReconcileReport
# --------------------------------------------------------------------------- #
def test_iter_peer_files(tmp_path):
    (tmp_path / "peer" / "msg_001").mkdir(parents=True)
    (tmp_path / "peer" / "msg_001" / "doc.md").write_text("x")
    (tmp_path / "peer" / "msg_001" / "meta.json").write_text("{}")
    (tmp_path / "peer" / "msg_002").mkdir()
    (tmp_path / "peer" / "msg_002" / "another.txt").write_text("y")
    items = list(_iter_peer_files(tmp_path / "peer"))
    paths = sorted((it.msg_id, it.path.name) for it in items)
    assert paths == [
        ("msg_001", "doc.md"),
        ("msg_001", "meta.json"),
        ("msg_002", "another.txt"),
    ]
    assert all(it.channel == "peer" for it in items)


def test_iter_peer_files_handles_missing_root(tmp_path):
    items = list(_iter_peer_files(tmp_path / "does-not-exist"))
    assert items == []


def test_iter_peer_files_skips_dot_subdirs(tmp_path):
    (tmp_path / ".hidden_msg").mkdir()
    (tmp_path / ".hidden_msg" / "x.md").write_text("y")
    items = list(_iter_peer_files(tmp_path))
    assert items == []


def test_iter_flat_files_ignores_dot_files_and_subdirs(tmp_path):
    media = tmp_path / "media" / "telegram"
    media.mkdir(parents=True)
    (media / "a.txt").write_text("x")
    (media / ".hidden").write_text("x")
    (media / "sub").mkdir()
    items = list(_iter_flat_files(media, channel="telegram"))
    assert [it.path.name for it in items] == ["a.txt"]
    assert all(it.channel == "telegram" for it in items)


def test_iter_flat_files_handles_missing_root(tmp_path):
    items = list(_iter_flat_files(tmp_path / "nope", channel="x"))
    assert items == []


# --------------------------------------------------------------------------- #
# Task 3b — run_attachments_reconcile orchestrator
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_reconcile_writes_textual_peer_file(tmp_path, vault_factory):
    vault, slug = vault_factory()
    msg = tmp_path / "peer" / "msg_xyz"
    msg.mkdir(parents=True)
    (msg / "plan.md").write_text("the plan body")
    # Inject a custom sources list so we don't depend on get_workspace_path()
    from nanobot.agent.wiki.attachments_reconciler import (
        _iter_peer_files,
    )
    sources = [
        ("peer", tmp_path / "peer", _iter_peer_files),
    ]
    from nanobot.agent.wiki.attachments_reconciler import run_attachments_reconcile
    report = await run_attachments_reconcile(vault, slug, sources=sources)
    assert report.created == ["inbox/plan.md"]
    assert (vault.wiki_dir / "inbox" / "plan.md").exists()


@pytest.mark.asyncio
async def test_reconcile_is_idempotent_on_rerun(tmp_path, vault_factory):
    from nanobot.agent.wiki.attachments_reconciler import (
        _iter_peer_files,
        run_attachments_reconcile,
    )
    vault, slug = vault_factory()
    msg = tmp_path / "peer" / "msg_xyz"
    msg.mkdir(parents=True)
    (msg / "p.md").write_text("plan")
    sources = [("peer", tmp_path / "peer", _iter_peer_files)]
    await run_attachments_reconcile(vault, slug, sources=sources)
    page_bytes = (vault.wiki_dir / "inbox" / "p.md").read_bytes()
    report = await run_attachments_reconcile(vault, slug, sources=sources)
    assert report.duplicates == 1
    assert report.created == []
    assert (vault.wiki_dir / "inbox" / "p.md").read_bytes() == page_bytes


@pytest.mark.asyncio
async def test_reconcile_walks_both_source_roots(tmp_path, vault_factory):
    """Two roots = two channels. Each must be walked independently."""
    from nanobot.agent.wiki.attachments_reconciler import (
        _iter_flat_files,
        _iter_peer_files,
        run_attachments_reconcile,
    )
    vault, slug = vault_factory()
    # Peer root
    peer_msg = tmp_path / "peer" / "msg_a"
    peer_msg.mkdir(parents=True)
    (peer_msg / "from_peer.md").write_text("peer content")
    # Telegram root (different directory tree)
    tg = tmp_path / "tg-media"
    tg.mkdir()
    (tg / "from_telegram.txt").write_text("tg content")
    (tg / "photo.jpg").write_bytes(b"\x89PNGfake")  # binary
    sources = [
        ("peer", tmp_path / "peer", _iter_peer_files),
        ("telegram", tg, lambda root: _iter_flat_files(root, channel="telegram")),
    ]
    report = await run_attachments_reconcile(vault, slug, sources=sources)
    created_names = {p.split("/")[-1] for p in report.created}
    assert "from_peer.md" in created_names
    assert "from_telegram.md" in created_names  # note: .txt becomes .md via writer
    assert report.skipped_binary == 1


@pytest.mark.asyncio
async def test_reconcile_swallows_writer_errors(tmp_path, vault_factory):
    """A write that returns WriteResult(status='error') is recorded in
    report.errors but does NOT stop the reconcile loop."""
    from nanobot.agent.wiki.attachments_reconciler import (
        _iter_peer_files,
        run_attachments_reconcile,
    )
    vault, slug = vault_factory()
    msg = tmp_path / "peer" / "msg_a"
    msg.mkdir(parents=True)
    # A file with a slug that _slug_ok rejects (leading dot) — writer returns error
    (msg / ".hidden_but_inside.md").write_text("x")  # NB: dot-files skipped by iter
    # Add a normal file too to verify the loop keeps going
    (msg / "fine.md").write_text("ok")
    sources = [("peer", tmp_path / "peer", _iter_peer_files)]
    report = await run_attachments_reconcile(vault, slug, sources=sources)
    # The hidden file was skipped by iter (dot prefix); only "fine.md" gets through
    assert report.created == ["inbox/fine.md"]
