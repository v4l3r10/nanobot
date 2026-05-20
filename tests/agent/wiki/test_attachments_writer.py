"""Tests for :mod:`nanobot.agent.wiki.attachments_writer` — the deterministic
attachment -> wiki page writer used by the eager AgentLoop hook AND the
Dream-side reconciler.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.wiki.attachments_writer import (
    WriteResult,
    classify_extension,
    _TEXTUAL_EXTS,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


@pytest.fixture
def vault_factory(tmp_path):
    """Return a callable that builds a fresh per-user vault under tmp_path.

    Mirrors the ``_vault`` helper in ``test_ingest.py`` — drops the bundled
    SCHEMA.md into ``wiki/`` so ``Vault.schema`` resolves the new 'inbox'
    type without a migration step.
    """
    from nanobot.agent.wiki.vault import Vault

    def _factory(name: str = "u1"):
        root = tmp_path / "memory" / "users" / name
        wiki = root / "wiki"
        wiki.mkdir(parents=True)
        (wiki / "SCHEMA.md").write_text(_bundled_schema_text(), encoding="utf-8")
        return Vault(root), name

    return _factory


# --------------------------------------------------------------------------- #
# Task 2a — Result dataclass + classify-by-extension
# --------------------------------------------------------------------------- #
def test_textual_extensions():
    for ext in (".md", ".txt", ".json", ".yaml", ".yml", ".csv"):
        assert classify_extension(ext) == "textual"


def test_binary_extensions():
    for ext in (".png", ".jpg", ".pdf", ".bin", ".mp3", ""):
        assert classify_extension(ext) == "binary"


def test_writeresult_shape():
    r = WriteResult(status="created", page_rel="inbox/x.md")
    assert r.status == "created"
    assert r.page_rel == "inbox/x.md"
    assert r.reason is None
