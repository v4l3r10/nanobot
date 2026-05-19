from pathlib import Path

import pytest

from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.agent.wiki.schema import load_schema
from nanobot.agent.wiki.vault import Vault


def _bundled_schema_text() -> str:
    # Resolve via the same mechanism Vault uses; for the test just read the repo file.
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


def _page(title, status="hot"):
    return Page(type="people", title=title, status=status,
                created="2026-05-18", updated="2026-05-18", last_touched="2026-05-18",
                tags=[], links_out=[], pinned=None, body=f"{title} body.\n")


def _make_vault(tmp_path, with_schema=True, with_pages=True, with_cold=True):
    root = tmp_path / "memory" / "users" / "u1"
    wiki = root / "wiki"
    (wiki / "people").mkdir(parents=True)
    if with_schema:
        (wiki / "SCHEMA.md").write_text(_bundled_schema_text(), encoding="utf-8")
    if with_pages:
        (wiki / "people" / "alice.md").write_text(serialize_page(_page("Alice")), encoding="utf-8")
        (wiki / "people" / "_index.md").write_text("# people\n- [[people/alice]]\n", encoding="utf-8")
    if with_cold:
        (wiki / ".cold" / "people").mkdir(parents=True)
        (wiki / ".cold" / "people" / "bob.md").write_text(
            serialize_page(_page("Bob", status="cold")), encoding="utf-8")
    return Vault(root)


def test_schema_loads_from_vault(tmp_path):
    v = _make_vault(tmp_path)
    assert v.schema.cold_after_days("people") == 180
    assert v.schema.folder("projects") == "projects"


def test_schema_falls_back_to_bundled_when_absent(tmp_path):
    v = _make_vault(tmp_path, with_schema=False, with_pages=False, with_cold=False)
    # no per-vault SCHEMA.md -> bundled master is used
    assert v.schema.folder("projects") == "projects"
    assert v.schema.cold_after_days("decisions") is None


def test_read_page(tmp_path):
    v = _make_vault(tmp_path)
    p = v.read_page("people/alice.md")
    assert p.title == "Alice" and p.type == "people"


def test_read_page_missing_raises(tmp_path):
    v = _make_vault(tmp_path)
    with pytest.raises(FileNotFoundError):
        v.read_page("people/nobody.md")


def test_read_page_traversal_blocked(tmp_path):
    v = _make_vault(tmp_path)
    with pytest.raises(ValueError):
        v.read_page("../../../../etc/passwd")


def test_iter_pages_skips_cold_and_index(tmp_path):
    v = _make_vault(tmp_path)
    hot = {rel for rel, _ in v.iter_pages()}
    assert hot == {"people/alice.md"}              # no _index.md, no .cold
    allp = {rel for rel, _ in v.iter_pages(include_cold=True)}
    assert allp == {"people/alice.md", ".cold/people/bob.md"}


def test_iter_pages_skips_malformed(tmp_path):
    v = _make_vault(tmp_path)
    (v.wiki_dir / "people" / "broken.md").write_text("not frontmatter", encoding="utf-8")
    rels = {rel for rel, _ in v.iter_pages()}
    assert rels == {"people/alice.md"}            # broken.md silently skipped


def test_is_empty(tmp_path):
    empty = Vault(tmp_path / "memory" / "users" / "fresh")
    assert empty.is_empty() is True
    v = _make_vault(tmp_path)
    assert v.is_empty() is False


# --- ensure_initialized (Task 4.5 B) ---------------------------------------


def test_ensure_initialized_creates_wiki_and_copies_schema(tmp_path):
    """A fresh vault: ensure_initialized creates wiki/ and copies the bundled
    SCHEMA.md verbatim (byte-equal, parseable via load_schema)."""
    v = Vault(tmp_path / "memory" / "users" / "fresh")
    assert not v.wiki_dir.exists()

    v.ensure_initialized()

    assert v.wiki_dir.is_dir()
    schema_file = v.wiki_dir / "SCHEMA.md"
    assert schema_file.is_file()
    written = schema_file.read_text(encoding="utf-8")
    # Byte-equal to the bundled master.
    assert written == _bundled_schema_text()
    # Parseable: a real Schema can be loaded from the copied file.
    schema = load_schema(written)
    assert schema.folder("people") == "people"
    assert schema.cold_after_days("people") == 180


def test_ensure_initialized_is_idempotent(tmp_path):
    """A second ensure_initialized makes ZERO changes (byte-stable SCHEMA.md)."""
    v = Vault(tmp_path / "memory" / "users" / "fresh")
    v.ensure_initialized()
    schema_file = v.wiki_dir / "SCHEMA.md"
    first = schema_file.read_bytes()
    mtime = schema_file.stat().st_mtime_ns

    v.ensure_initialized()

    assert schema_file.read_bytes() == first
    # The file was not rewritten (idempotent: a second call writes nothing).
    assert schema_file.stat().st_mtime_ns == mtime


def test_ensure_initialized_no_legacy_workspace_is_byte_identical(tmp_path):
    """Task 7.1 wiring guard: ensure_initialized() WITHOUT a legacy workspace
    behaves EXACTLY as before — mkdir wiki/ + copy SCHEMA only, NO migration,
    NO .migrated marker, NO MEMORY.md MOC / imported-memory page."""
    v = Vault(tmp_path / "memory" / "users" / "fresh")

    v.ensure_initialized()

    assert v.wiki_dir.is_dir()
    assert (v.wiki_dir / "SCHEMA.md").is_file()
    # The 4.5 contract: nothing beyond wiki/ + SCHEMA.md.
    assert not (v.root / ".migrated").exists()
    assert not (v.root / "USER.md").exists()
    assert not (v.root / "MEMORY.md").exists()
    assert v.is_empty()
    # The vault root holds exactly the wiki/ dir (no migration side effects).
    assert sorted(p.name for p in v.root.iterdir()) == ["wiki"]


def test_ensure_initialized_with_none_legacy_is_byte_identical(tmp_path):
    """Passing legacy_workspace=None explicitly is the same no-migration path."""
    v = Vault(tmp_path / "memory" / "users" / "fresh")
    v.ensure_initialized(legacy_workspace=None)
    assert not (v.root / ".migrated").exists()
    assert sorted(p.name for p in v.root.iterdir()) == ["wiki"]


def test_ensure_initialized_does_not_overwrite_existing_schema(tmp_path):
    """An existing per-vault SCHEMA.md is preserved verbatim (never clobbered
    by the bundled master)."""
    v = _make_vault(tmp_path, with_schema=False, with_pages=False, with_cold=False)
    v.wiki_dir.mkdir(parents=True, exist_ok=True)
    custom = (
        "# Custom Wiki Schema\n```yaml\n"
        "types:\n  people: { folder: people, cold_after_days: 7 }\n"
        "required_frontmatter: [type, title, status, created, updated, last_touched]\n"
        "moc_max_lines: 42\n```\n"
    )
    (v.wiki_dir / "SCHEMA.md").write_text(custom, encoding="utf-8")

    v.ensure_initialized()

    assert (v.wiki_dir / "SCHEMA.md").read_text(encoding="utf-8") == custom
    # The custom schema is what resolves (cached property reads per-vault first).
    assert Vault(v.root).schema.cold_after_days("people") == 7
