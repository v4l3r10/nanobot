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
    """An existing per-vault SCHEMA.md is preserved (never clobbered by the
    bundled master). The Task-6 one-shot upgrade may APPEND an ``inbox`` type
    line if missing, but user customizations on other types and other YAML
    keys (``moc_max_lines``, ``required_frontmatter``) are preserved verbatim.
    """
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

    # The custom schema is preserved: the user's cold_after_days for people
    # and the custom moc_max_lines are untouched (cached property reads
    # per-vault first).
    schema = Vault(v.root).schema
    assert schema.cold_after_days("people") == 7
    assert schema.moc_max_lines == 42
    assert schema.required_frontmatter == [
        "type", "title", "status", "created", "updated", "last_touched",
    ]

    # Byte-level "nothing else changed" check: the post-upgrade text minus
    # the single inserted inbox line must equal the original schema text.
    from nanobot.agent.wiki.vault import _INBOX_SCHEMA_LINE
    final = (v.wiki_dir / "SCHEMA.md").read_text(encoding="utf-8")
    # The helper inserts the inbox line with a trailing newline.
    inserted = _INBOX_SCHEMA_LINE
    assert inserted in final
    assert final.replace(inserted, "", 1) == custom


def test_ensure_initialized_upgrades_old_schema_to_include_inbox(tmp_path):
    """Vaults created before the inbox type was added must be upgraded
    in-place when ensure_initialized runs. The upgrade must preserve other
    user-customized parts of SCHEMA.md."""
    vault_root = tmp_path / "v"
    vault_root.mkdir()
    wiki = vault_root / "wiki"
    wiki.mkdir()
    # Pre-Task-0 schema: only `people` type, custom cold value, custom moc_max_lines.
    (wiki / "SCHEMA.md").write_text(
        "# Wiki Schema\n"
        "```yaml\n"
        "types:\n"
        "  people: { folder: people, cold_after_days: 200 }\n"
        "required_frontmatter: [type, title, status]\n"
        "moc_max_lines: 99\n"
        "```\n"
    )
    from nanobot.agent.wiki.vault import Vault
    vault = Vault(vault_root)
    vault.ensure_initialized(None)
    # After upgrade:
    assert vault.schema.is_known_type("inbox")
    assert vault.schema.folder("inbox") == "inbox"
    assert vault.schema.cold_after_days("inbox") == 30
    # Pre-existing custom values preserved:
    assert vault.schema.is_known_type("people")
    assert vault.schema.cold_after_days("people") == 200
    assert vault.schema.moc_max_lines == 99
    assert vault.schema.required_frontmatter == ["type", "title", "status"]


def test_ensure_initialized_upgrade_is_idempotent(tmp_path):
    """Running ensure_initialized twice must not double-add the inbox type
    or otherwise mutate the schema file content beyond the first run."""
    vault_root = tmp_path / "v"
    vault_root.mkdir()
    wiki = vault_root / "wiki"
    wiki.mkdir()
    (wiki / "SCHEMA.md").write_text(
        "# Wiki Schema\n"
        "```yaml\n"
        "types:\n"
        "  people: { folder: people, cold_after_days: 200 }\n"
        "```\n"
    )
    from nanobot.agent.wiki.vault import Vault
    vault = Vault(vault_root)
    vault.ensure_initialized(None)
    bytes_after_first = (wiki / "SCHEMA.md").read_bytes()
    # Second call must be a true no-op
    vault2 = Vault(vault_root)
    vault2.ensure_initialized(None)
    bytes_after_second = (wiki / "SCHEMA.md").read_bytes()
    assert bytes_after_first == bytes_after_second, "second ensure_initialized mutated the schema"
    # inbox should still be exactly once
    text = (wiki / "SCHEMA.md").read_text(encoding="utf-8")
    assert text.count("inbox:") == 1


def test_ensure_initialized_modern_schema_unchanged(tmp_path):
    """A schema that ALREADY has the inbox type (modern bundled) is not
    rewritten — the upgrade is gated on absence of the inbox type."""
    vault_root = tmp_path / "v"
    vault_root.mkdir()
    wiki = vault_root / "wiki"
    wiki.mkdir()
    schema_text = (
        "# Wiki Schema\n"
        "```yaml\n"
        "types:\n"
        "  people: { folder: people, cold_after_days: 180 }\n"
        "  inbox:  { folder: inbox,  cold_after_days: 30 }\n"
        "```\n"
    )
    (wiki / "SCHEMA.md").write_text(schema_text)
    bytes_before = (wiki / "SCHEMA.md").read_bytes()
    from nanobot.agent.wiki.vault import Vault
    Vault(vault_root).ensure_initialized(None)
    bytes_after = (wiki / "SCHEMA.md").read_bytes()
    assert bytes_before == bytes_after
