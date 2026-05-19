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

import pytest

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


# --- Review follow-up I2: a non-utf8 legacy file must not loop migration ----


def test_non_utf8_legacy_memory_does_not_raise_or_loop(tmp_path):
    """I2 regression: a non-utf8 ``memory/MEMORY.md`` previously raised
    ``UnicodeDecodeError`` (a ``ValueError``/``UnicodeError``, NOT ``OSError``)
    out of ``migrate_legacy`` — escaping its ``except OSError`` fail-safe so
    the ``.migrated`` marker was NEVER written and Dream re-ran the migration
    every cycle forever. After the fix: no raise, marker IS written (one-shot),
    ``migrate_legacy`` returns True, the legacy bytes are untouched, and any
    imported page is parseable."""
    workspace, vault = _fresh_vault(tmp_path)
    legacy_memory = workspace / "memory" / "MEMORY.md"
    # Latin-1 bytes that are NOT valid UTF-8 (0xe9 = 'é' in latin-1 is an
    # invalid lone continuation byte in UTF-8). Real "memory" text so the
    # template/blank guards do not short-circuit before the decode.
    raw = "cafe resume facade naive".encode("ascii") + b" \xe9\xff\xfe biographie"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")  # genuinely non-utf8
    legacy_memory.parent.mkdir(parents=True, exist_ok=True)
    legacy_memory.write_bytes(raw)
    mem_before = legacy_memory.read_bytes()

    # Must NOT raise (pre-fix this raised UnicodeDecodeError).
    result = migrate_legacy(workspace, vault)

    assert result is True
    marker = vault.root / ".migrated"
    assert marker.is_file()
    # One-shot: a second call short-circuits on the marker (no infinite loop).
    assert migrate_legacy(workspace, vault) is False
    # Legacy source byte-unchanged (never modified/deleted).
    assert legacy_memory.read_bytes() == mem_before
    # If a page was imported it must be parseable (errors="replace" content).
    page_file = vault.page_path("concepts", "imported-memory")
    if page_file.is_file():
        page = parse_page(page_file.read_text(encoding="utf-8"))
        assert page.type == "concepts"
        assert page.status == "hot"


def test_non_utf8_legacy_user_does_not_raise_or_loop(tmp_path):
    """I2 regression for the USER.md path: same failure mode via the second
    ``_read_text_or_empty`` call inside ``migrate_legacy``."""
    workspace, vault = _fresh_vault(tmp_path)
    legacy_user = workspace / "USER.md"
    legacy_user.write_bytes(b"Prenom Jose Muller \xff\xfe\xe9 profile")
    user_before = legacy_user.read_bytes()

    result = migrate_legacy(workspace, vault)

    assert result is True
    assert (vault.root / ".migrated").is_file()
    assert migrate_legacy(workspace, vault) is False
    assert legacy_user.read_bytes() == user_before


# --- Review follow-up I1: imported page body is bounded --------------------


def test_oversized_legacy_memory_body_is_capped(tmp_path):
    """I1 regression: a multi-MB legacy ``memory/MEMORY.md`` must not become
    one unbounded page body (per-cycle Lint re-parse / git blob cost). The
    imported body is truncated to ``_MAX_IMPORT_BODY_CHARS`` plus a clear
    truncation marker; it stays parseable; the legacy source is unchanged."""
    from nanobot.agent.wiki.migrate import _MAX_IMPORT_BODY_CHARS

    workspace, vault = _fresh_vault(tmp_path)
    legacy_memory = workspace / "memory" / "MEMORY.md"
    huge = "A" * (_MAX_IMPORT_BODY_CHARS * 3)
    legacy_memory.parent.mkdir(parents=True, exist_ok=True)
    legacy_memory.write_text(huge, encoding="utf-8")
    mem_before = legacy_memory.read_bytes()

    result = migrate_legacy(workspace, vault)

    assert result is True
    page_file = vault.page_path("concepts", "imported-memory")
    assert page_file.is_file()
    page = parse_page(page_file.read_text(encoding="utf-8"))
    # Body capped at the bound + a (short) truncation marker line.
    assert len(page.body) <= _MAX_IMPORT_BODY_CHARS + 200
    assert len(page.body) < len(huge)
    assert "truncated at import" in page.body
    assert page.type == "concepts"
    assert page.status == "hot"
    # Legacy source byte-unchanged (full original preserved on disk).
    assert legacy_memory.read_bytes() == mem_before


def test_under_cap_legacy_memory_body_not_truncated(tmp_path):
    """A normal-sized legacy memory is imported verbatim (no marker)."""
    from nanobot.agent.wiki.migrate import _MAX_IMPORT_BODY_CHARS

    workspace, vault = _fresh_vault(tmp_path)
    body = "real memory line\n" * 10
    assert len(body) < _MAX_IMPORT_BODY_CHARS
    _write(workspace / "memory" / "MEMORY.md", body)

    assert migrate_legacy(workspace, vault) is True

    page = parse_page(
        vault.page_path("concepts", "imported-memory").read_text(encoding="utf-8")
    )
    assert page.body == body
    assert "truncated at import" not in page.body
