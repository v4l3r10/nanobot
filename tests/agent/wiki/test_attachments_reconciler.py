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
