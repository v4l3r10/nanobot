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
    write_attachment_page,
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


# --------------------------------------------------------------------------- #
# Task 2b — Containment + binary skip + missing file
# --------------------------------------------------------------------------- #
def test_binary_skipped(tmp_path, vault_factory):
    vault, slug = vault_factory()
    f = tmp_path / "x.png"
    f.write_bytes(b"\x89PNG")
    r = write_attachment_page(vault, slug, f, "test", "m1", allowed_roots=[tmp_path])
    assert r.status == "skipped_binary"
    assert r.page_rel is None


def test_escape_workspace_rejected(tmp_path, vault_factory):
    vault, slug = vault_factory()
    outside = tmp_path.parent / "evil.md"
    outside.write_text("evil")
    r = write_attachment_page(
        vault, slug, outside, "test", "m1", allowed_roots=[tmp_path]
    )
    assert r.status == "error"
    assert "escape" in (r.reason or "")


def test_missing_file(tmp_path, vault_factory):
    vault, slug = vault_factory()
    r = write_attachment_page(
        vault, slug, tmp_path / "nope.md", "test", "m1", allowed_roots=[tmp_path]
    )
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# Task 2c — Textual create + APPEND fallback + body clamp
# --------------------------------------------------------------------------- #
def test_create_new_inbox_page(tmp_path, vault_factory):
    vault, slug = vault_factory()
    src = tmp_path / "plan.md"
    src.write_text("# Plan\n\nFirst line of the plan body.\n")
    r = write_attachment_page(
        vault, slug, src, "peer", "m-123", allowed_roots=[tmp_path]
    )
    assert r.status == "created"
    assert r.page_rel == "inbox/plan.md"
    page_file = vault.wiki_dir / "inbox" / "plan.md"
    assert page_file.exists()
    txt = page_file.read_text(encoding="utf-8")
    assert "type: inbox" in txt
    assert "status: hot" in txt
    assert "- peer" in txt    # tags
    assert "- m-123" in txt
    assert "Source: peer/m-123" in txt
    assert "First line of the plan body." in txt


def test_collision_appends(tmp_path, vault_factory):
    vault, slug = vault_factory()
    src = tmp_path / "plan.md"
    src.write_text("first body")
    write_attachment_page(vault, slug, src, "peer", "m-1", allowed_roots=[tmp_path])
    # Second file, same slug, different content:
    src.write_text("second body completely different")
    r = write_attachment_page(
        vault, slug, src, "peer", "m-2", allowed_roots=[tmp_path]
    )
    assert r.status == "appended"
    txt = (vault.wiki_dir / "inbox" / "plan.md").read_text(encoding="utf-8")
    assert "first body" in txt
    assert "second body" in txt


def test_body_clamped_at_8000(tmp_path, vault_factory):
    vault, slug = vault_factory()
    src = tmp_path / "big.md"
    src.write_text("X" * 20_000)
    r = write_attachment_page(
        vault, slug, src, "test", "m-1", allowed_roots=[tmp_path]
    )
    assert r.status == "created"
    txt = (vault.wiki_dir / "inbox" / "big.md").read_text(encoding="utf-8")
    assert len(txt) < 9000  # body clamped + frontmatter + source header < 9000


# --------------------------------------------------------------------------- #
# Task 2d — Manifest read/write + sha256 dedup
# --------------------------------------------------------------------------- #
def test_manifest_records_created(tmp_path, vault_factory):
    vault, slug = vault_factory()
    src = tmp_path / "doc.md"
    src.write_text("hello")
    write_attachment_page(vault, slug, src, "peer", "m-1", allowed_roots=[tmp_path])
    manifest = vault.wiki_dir / ".ingested_attachments.json"
    assert manifest.exists()
    import json
    data = json.loads(manifest.read_text())
    assert data["version"] == 1
    assert len(data["entries"]) == 1
    e = data["entries"][0]
    assert e["channel"] == "peer"
    assert e["msg_id"] == "m-1"
    assert e["status"] == "ingested"
    assert e["page"] == "inbox/doc.md"
    assert len(e["sha256"]) == 64


def test_same_bytes_different_path_is_duplicate(tmp_path, vault_factory):
    vault, slug = vault_factory()
    a = tmp_path / "a.md"
    a.write_text("identical bytes")
    write_attachment_page(vault, slug, a, "peer", "m-1", allowed_roots=[tmp_path])
    b = tmp_path / "b.md"
    b.write_text("identical bytes")
    r = write_attachment_page(vault, slug, b, "peer", "m-2", allowed_roots=[tmp_path])
    assert r.status == "duplicate"
    assert not (vault.wiki_dir / "inbox" / "b.md").exists()


def test_rerun_same_file_is_noop(tmp_path, vault_factory):
    vault, slug = vault_factory()
    src = tmp_path / "doc.md"
    src.write_text("hello")
    write_attachment_page(vault, slug, src, "peer", "m-1", allowed_roots=[tmp_path])
    page_bytes_1 = (vault.wiki_dir / "inbox" / "doc.md").read_bytes()
    r = write_attachment_page(vault, slug, src, "peer", "m-1", allowed_roots=[tmp_path])
    assert r.status == "duplicate"
    page_bytes_2 = (vault.wiki_dir / "inbox" / "doc.md").read_bytes()
    assert page_bytes_1 == page_bytes_2


def test_binary_recorded_in_manifest(tmp_path, vault_factory):
    vault, slug = vault_factory()
    src = tmp_path / "img.png"
    src.write_bytes(b"\x89PNGfake")
    write_attachment_page(vault, slug, src, "peer", "m-1", allowed_roots=[tmp_path])
    import json
    data = json.loads((vault.wiki_dir / ".ingested_attachments.json").read_text())
    assert any(e["status"] == "skipped_binary" for e in data["entries"])
