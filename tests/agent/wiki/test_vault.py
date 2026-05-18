from pathlib import Path

import pytest

from nanobot.agent.wiki.page import Page, serialize_page
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
