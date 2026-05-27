"""Tests for ``rebuild_indexes_and_moc`` -- the cheap deterministic pass.

This is exactly Lint phases 1 (read-only scan) + 6 (regenerate indexes) +
7 (regenerate MOC), WITHOUT the mutating curation phases and WITHOUT
touching ``.lint.log``. Vaults are built on ``tmp_path`` and the pages are
written with the full required frontmatter so they parse as valid HOT pages
through the production :func:`parse_page` path.
"""

import datetime as dt
from pathlib import Path

from nanobot.agent.wiki.lint import rebuild_indexes_and_moc, run_lint
from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.agent.wiki.vault import Vault


def _vault(tmp_path: Path) -> Vault:
    v = Vault(tmp_path / "v")
    v.ensure_initialized()  # no legacy_workspace
    return v


def _page(v: Vault, type_: str, slug: str, title: str, body: str = "x") -> None:
    # The bundled SCHEMA requires [type, title, status, created, updated,
    # last_touched] and parse_page rejects a page missing any of them, so
    # created/updated are included here for the page to parse as a valid
    # HOT page (minimal adjustment over the spec stub).
    folder = v.schema.folder(type_)
    d = v.wiki_dir / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.md").write_text(
        f"---\ntype: {type_}\ntitle: {title}\nstatus: hot\n"
        f"created: 2026-05-19\nupdated: 2026-05-19\n"
        f"last_touched: 2026-05-19\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_rebuild_generates_index_and_moc_from_frontmatter(tmp_path):
    v = _vault(tmp_path)
    _page(v, "concepts", "alpha", "Alpha")
    changed = rebuild_indexes_and_moc(v)
    assert changed is True
    folder = v.schema.folder("concepts")
    idx = (v.wiki_dir / folder / "_index.md").read_text(encoding="utf-8")
    assert f"[[{folder}/alpha]]" in idx
    moc = (v.root / "MEMORY.md").read_text(encoding="utf-8")
    assert "# Memory" in moc and f"[[{folder}/alpha]]" in moc


def test_rebuild_is_idempotent(tmp_path):
    v = _vault(tmp_path)
    _page(v, "concepts", "alpha", "Alpha")
    assert rebuild_indexes_and_moc(v) is True
    moc1 = (v.root / "MEMORY.md").read_bytes()
    assert rebuild_indexes_and_moc(v) is False  # no-op second run
    assert (v.root / "MEMORY.md").read_bytes() == moc1


def test_rebuild_does_not_move_cool_or_dedup(tmp_path):
    """Only _index.md / MEMORY.md may change; pages and .cold/ untouched."""
    v = _vault(tmp_path)
    _page(v, "concepts", "alpha", "Alpha")
    folder = v.schema.folder("concepts")
    page = v.wiki_dir / folder / "alpha.md"
    before = page.read_bytes()
    cold = v.wiki_dir / ".cold"
    rebuild_indexes_and_moc(v)
    assert page.read_bytes() == before          # page never rewritten/moved
    assert not cold.exists()                    # no decay/cooling happened


def test_rebuild_matches_run_lint_regeneration_on_settled_vault(tmp_path):
    """On a settled (no curation needed) vault, the cheap pass and full
    run_lint produce the SAME _index.md / MEMORY.md (no divergence)."""
    v = _vault(tmp_path)
    _page(v, "concepts", "alpha", "Alpha")
    _page(v, "people", "bob", "Bob")
    run_lint(v, dt.date(2026, 5, 19))

    # Snapshot the settled vault: MOC + every _index.md (relative keys, sorted).
    moc_lint = (v.root / "MEMORY.md").read_bytes()
    index_snapshot = {
        p.relative_to(v.wiki_dir): p.read_bytes()
        for p in sorted(v.wiki_dir.rglob("_index.md"))
    }
    assert index_snapshot, "run_lint should have produced at least one _index.md"

    # Wipe everything the cheap pass is responsible for regenerating.
    (v.root / "MEMORY.md").unlink()
    for rel in index_snapshot:
        (v.wiki_dir / rel).unlink()

    assert rebuild_indexes_and_moc(v) is True

    # MOC byte-parity.
    assert (v.root / "MEMORY.md").read_bytes() == moc_lint

    # _index.md byte-parity: exact same set of relpaths, exact same bytes.
    rebuilt_indexes = {
        p.relative_to(v.wiki_dir): p.read_bytes()
        for p in sorted(v.wiki_dir.rglob("_index.md"))
    }
    assert sorted(rebuilt_indexes) == sorted(index_snapshot)
    assert rebuilt_indexes == index_snapshot


def test_rebuild_includes_reheated_page_still_under_cold(tmp_path):
    """A read-reheated page (status=hot, physically under .cold/) re-enters the
    MOC/_index on the cheap rebuild, rendered with its CANONICAL link, without
    waiting for the heavy Dream relocate."""
    v = _vault(tmp_path)
    folder = v.schema.folder("people")
    cold_dir = v.wiki_dir / ".cold" / folder
    cold_dir.mkdir(parents=True, exist_ok=True)
    (cold_dir / "sam.md").write_text(
        serialize_page(Page(
            type="people", title="Sam", status="hot",  # status hot...
            created="2026-05-27", updated="2026-05-27", last_touched="2026-05-27",
            body="x\n",
        )),
        encoding="utf-8",
    )  # ...but physically under .cold/

    assert rebuild_indexes_and_moc(v) is True
    moc = (v.root / "MEMORY.md").read_text(encoding="utf-8")
    assert f"[[{folder}/sam]]" in moc, "canonical link must appear in the MOC"
    assert ".cold/" not in moc, "must NOT render a .cold/ link"
    idx = (v.wiki_dir / folder / "_index.md").read_text(encoding="utf-8")
    assert f"[[{folder}/sam]]" in idx


def test_rebuild_excludes_genuinely_cold_page(tmp_path):
    """A genuinely cold page (status=cold under .cold/) stays OUT of the MOC."""
    v = _vault(tmp_path)
    folder = v.schema.folder("people")
    cold_dir = v.wiki_dir / ".cold" / folder
    cold_dir.mkdir(parents=True, exist_ok=True)
    (cold_dir / "gone.md").write_text(
        serialize_page(Page(
            type="people", title="Gone", status="cold",
            created="2020-01-01", updated="2020-01-01", last_touched="2020-01-01",
            body="x\n",
        )),
        encoding="utf-8",
    )
    rebuild_indexes_and_moc(v)
    moc_path = v.root / "MEMORY.md"
    moc = moc_path.read_text(encoding="utf-8") if moc_path.exists() else ""
    assert "gone" not in moc, "a genuinely cold page must not appear in the MOC"
