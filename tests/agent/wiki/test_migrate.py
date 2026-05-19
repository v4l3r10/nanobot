"""Task 7.1 — one-time legacy MEMORY.md/USER.md migration into a per-user vault.

``migrate_legacy(workspace, vault)`` is pure filesystem, LLM-free,
deterministic and idempotent (one-shot per vault via a ``.migrated``
marker). It NEVER modifies or deletes the legacy sources. Real legacy
memory becomes ONE ``concepts/imported-memory.md`` page satisfying the 4.3
Lint contract; legacy ``USER.md`` becomes ``vault.root/USER.md`` (what Task
6.1 reads). The end-to-end test proves the next Lint indexes the imported
page into ``concepts/_index.md`` + the root MOC.
"""

from __future__ import annotations

import datetime
from importlib.resources import files as pkg_files
from pathlib import Path

from nanobot.agent.wiki.lint import run_lint
from nanobot.agent.wiki.migrate import migrate_legacy
from nanobot.agent.wiki.page import parse_page
from nanobot.agent.wiki.vault import Vault


def _fresh_vault(tmp_path: Path) -> tuple[Path, Vault]:
    """A workspace + an initialized fresh vault on memory/users/<slug>."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    vault = Vault(workspace / "memory" / "users" / "unified_default")
    vault.ensure_initialized()  # mkdir wiki/ + copy bundled SCHEMA (no legacy)
    return workspace, vault


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bundled_memory_template() -> str:
    return (
        pkg_files("nanobot") / "templates" / "memory" / "MEMORY.md"
    ).read_text(encoding="utf-8")


def test_migrates_legacy_memory_and_user(tmp_path):
    workspace, vault = _fresh_vault(tmp_path)
    legacy_memory = workspace / "memory" / "MEMORY.md"
    legacy_user = workspace / "USER.md"
    _write(legacy_memory, "LEGACY-MEM-FACT")
    _write(legacy_user, "LEGACY-USER-PROFILE")
    mem_before = legacy_memory.read_bytes()
    user_before = legacy_user.read_bytes()

    result = migrate_legacy(workspace, vault)

    assert result is True

    page_file = vault.page_path("concepts", "imported-memory")
    assert page_file.is_file()
    page = parse_page(page_file.read_text(encoding="utf-8"))
    today = datetime.date.today().isoformat()
    assert page.type == "concepts"
    assert page.status == "hot"
    assert page.created == today
    assert page.updated == today
    assert page.last_touched == today
    assert "LEGACY-MEM-FACT" in page.body
    # Not under the cold archive subtree.
    assert ".cold" not in page_file.relative_to(vault.wiki_dir).parts

    assert (vault.root / "USER.md").read_text(encoding="utf-8") == "LEGACY-USER-PROFILE"
    assert (vault.root / ".migrated").is_file()

    # Legacy sources are NEVER modified or deleted.
    assert legacy_memory.read_bytes() == mem_before
    assert legacy_user.read_bytes() == user_before


def test_migration_is_idempotent(tmp_path):
    workspace, vault = _fresh_vault(tmp_path)
    _write(workspace / "memory" / "MEMORY.md", "LEGACY-MEM-FACT")
    _write(workspace / "USER.md", "LEGACY-USER-PROFILE")

    first = migrate_legacy(workspace, vault)
    marker = vault.root / ".migrated"
    marker_bytes = marker.read_bytes()
    page_file = vault.page_path("concepts", "imported-memory")
    page_bytes = page_file.read_bytes()
    user_bytes = (vault.root / "USER.md").read_bytes()

    second = migrate_legacy(workspace, vault)

    assert first is True
    assert second is False
    # No second/duplicate page; marker + vault USER.md byte-stable.
    assert page_file.read_bytes() == page_bytes
    assert marker.read_bytes() == marker_bytes
    assert (vault.root / "USER.md").read_bytes() == user_bytes
    concepts_dir = page_file.parent
    imported = sorted(p.name for p in concepts_dir.glob("imported-memory*.md"))
    assert imported == ["imported-memory.md"]


def test_template_memory_not_migrated(tmp_path):
    workspace, vault = _fresh_vault(tmp_path)
    # MEMORY.md is byte-identical to the bundled stock template.
    _write(workspace / "memory" / "MEMORY.md", _bundled_memory_template())
    _write(workspace / "USER.md", "REAL-USER")

    result = migrate_legacy(workspace, vault)

    assert result is True
    # No bogus imported-memory page for stock-template content.
    assert not vault.page_path("concepts", "imported-memory").exists()
    # One-shot marker still set.
    assert (vault.root / ".migrated").is_file()
    # A real root USER.md is still migrated.
    assert (vault.root / "USER.md").read_text(encoding="utf-8") == "REAL-USER"


def test_missing_legacy_files_safe(tmp_path):
    workspace, vault = _fresh_vault(tmp_path)
    # No memory/MEMORY.md, no root USER.md.

    result = migrate_legacy(workspace, vault)

    assert result is True
    assert not vault.page_path("concepts", "imported-memory").exists()
    assert not (vault.root / "USER.md").exists()
    assert (vault.root / ".migrated").is_file()


def test_does_not_clobber_existing_vault_user(tmp_path):
    workspace, vault = _fresh_vault(tmp_path)
    (vault.root / "USER.md").write_text("EXISTING", encoding="utf-8")
    _write(workspace / "USER.md", "LEGACY-DIFFERENT")
    _write(workspace / "memory" / "MEMORY.md", "LEGACY-MEM-FACT")

    result = migrate_legacy(workspace, vault)

    assert result is True
    # An existing non-blank vault USER.md is never overwritten.
    assert (vault.root / "USER.md").read_text(encoding="utf-8") == "EXISTING"


def test_end_to_end_migrate_then_lint_indexes_imported_page(tmp_path):
    """The 6.1 contract: after migrate, Lint indexes the imported page into
    concepts/_index.md and the root MEMORY.md MOC so the wiki read path
    surfaces migrated memory."""
    workspace, vault = _fresh_vault(tmp_path)
    _write(workspace / "memory" / "MEMORY.md", "LEGACY-MEM-FACT")

    assert migrate_legacy(workspace, vault) is True

    run_lint(vault, datetime.date.today())

    index = vault.wiki_dir / "concepts" / "_index.md"
    assert index.is_file()
    assert "[[concepts/imported-memory]]" in index.read_text(encoding="utf-8")

    moc = vault.root / "MEMORY.md"
    assert moc.is_file()
    moc_text = moc.read_text(encoding="utf-8")
    assert "[[concepts/_index]]" in moc_text
    assert "[[concepts/imported-memory]]" in moc_text

    # Still hot / not archived to .cold.
    page_file = vault.page_path("concepts", "imported-memory")
    assert page_file.is_file()
    assert ".cold" not in page_file.relative_to(vault.wiki_dir).parts


def test_blank_legacy_memory_not_migrated(tmp_path):
    workspace, vault = _fresh_vault(tmp_path)
    _write(workspace / "memory" / "MEMORY.md", "   \n\n  ")

    result = migrate_legacy(workspace, vault)

    assert result is True
    assert not vault.page_path("concepts", "imported-memory").exists()
    assert (vault.root / ".migrated").is_file()
