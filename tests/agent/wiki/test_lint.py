"""Tests for the deterministic, idempotent Lint engine (Task 4.3).

Vaults are built on ``tmp_path`` with the real :func:`serialize_page` and the
bundled SCHEMA so the assertions exercise the production parse/serialize path.
The two non-negotiable properties (idempotence + determinism) get dedicated
full-tree byte-snapshot tests; the per-phase tests assert filesystem end-state
plus the :class:`LintReport`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from nanobot.agent.wiki.lint import LintReport, run_lint
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import Vault

TODAY = dt.date(2026, 5, 18)


def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


def _page(
    *,
    type="people",
    title="T",
    status="hot",
    created="2020-01-01",
    updated="2026-05-18",
    last_touched="2026-05-18",
    tags=None,
    links_out=None,
    pinned=None,
    body="body\n",
):
    return Page(
        type=type,
        title=title,
        status=status,
        created=created,
        updated=updated,
        last_touched=last_touched,
        tags=list(tags or []),
        links_out=list(links_out or []),
        pinned=pinned,
        body=body,
    )


def _vault(tmp_path, *, schema=True):
    root = tmp_path / "memory" / "users" / "u1"
    wiki = root / "wiki"
    wiki.mkdir(parents=True)
    if schema:
        (wiki / "SCHEMA.md").write_text(_bundled_schema_text(), encoding="utf-8")
    return Vault(root)


def _write(vault: Vault, rel: str, page: Page) -> Path:
    p = vault.wiki_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(serialize_page(page), encoding="utf-8")
    return p


def _snapshot(vault: Vault) -> dict[str, bytes]:
    """Full byte snapshot of every file under the vault root (POSIX keys)."""
    snap: dict[str, bytes] = {}
    for p in sorted(vault.root.rglob("*")):
        if p.is_file():
            snap[p.relative_to(vault.root).as_posix()] = p.read_bytes()
    return snap


class TestStaleToCold:
    def test_stale_hot_page_moved_to_cold(self, tmp_path):
        v = _vault(tmp_path)
        _write(v, "people/alice.md", _page(title="Alice", last_touched="2025-01-01"))
        rep = run_lint(v, TODAY)

        old = v.wiki_dir / "people" / "alice.md"
        new = v.wiki_dir / ".cold" / "people" / "alice.md"
        assert not old.exists()
        assert new.exists()
        page = parse_page(new.read_text(encoding="utf-8"))
        assert page.status == "cold"
        assert "people/alice.md" in rep.cooled

    def test_pinned_stale_not_cooled(self, tmp_path):
        v = _vault(tmp_path)
        _write(
            v,
            "people/bob.md",
            _page(title="Bob", last_touched="2000-01-01", pinned=True),
        )
        rep = run_lint(v, TODAY)
        assert (v.wiki_dir / "people" / "bob.md").exists()
        assert not (v.wiki_dir / ".cold" / "people" / "bob.md").exists()
        assert rep.cooled == []


class TestReheat:
    def test_hot_page_under_cold_is_relocated(self, tmp_path):
        v = _vault(tmp_path)
        _write(
            v,
            ".cold/people/bob.md",
            _page(title="Bob", status="hot", last_touched="2026-05-18"),
        )
        rep = run_lint(v, TODAY)

        assert not (v.wiki_dir / ".cold" / "people" / "bob.md").exists()
        new = v.wiki_dir / "people" / "bob.md"
        assert new.exists()
        assert parse_page(new.read_text(encoding="utf-8")).status == "hot"
        assert ".cold/people/bob.md -> people/bob.md" in rep.reheated

    def test_reheated_page_not_immediately_recooled(self, tmp_path):
        v = _vault(tmp_path)
        _write(
            v,
            ".cold/people/bob.md",
            _page(title="Bob", status="hot", last_touched=TODAY.isoformat()),
        )
        run_lint(v, TODAY)
        # Lands hot and is NOT re-cooled (last_touched == today).
        assert (v.wiki_dir / "people" / "bob.md").exists()
        assert not (v.wiki_dir / ".cold" / "people" / "bob.md").exists()


class TestDedup:
    # NOTE: a case-only-distinct filename pair (``a.md`` / ``A.md``) is
    # unrepresentable on a case-insensitive filesystem (Windows/macOS) -- the
    # second write just overwrites the first physical file. The portable way
    # two hot pages end up in the same ``(type, slug.lower())`` group is the
    # documented reheat-relocate collision: a ``status: hot`` page still under
    # ``.cold/`` whose hot target already exists. Both survive into dedup.

    def test_reheat_collision_merges_keeper_is_newest_updated(self, tmp_path):
        v = _vault(tmp_path)
        # Existing hot page (older `updated`).
        _write(
            v,
            "projects/svc.md",
            _page(type="projects", title="Svc", updated="2026-05-01", body="HOT\n"),
        )
        # A reheated page still under .cold with the SAME slug, newer updated.
        _write(
            v,
            ".cold/projects/svc.md",
            _page(
                type="projects",
                title="SvcCold",
                status="hot",
                updated="2026-05-10",
                last_touched="2026-05-18",
                body="COLDBODY\n",
            ),
        )
        rep = run_lint(v, TODAY)

        # Reheat-relocate defers the collision (cold copy left in place,
        # still status hot). Dedup merges: keeper = max `updated` = the
        # reheated one (2026-05-10) at relpath .cold/projects/svc.md; loser
        # = projects/svc.md (updated 2026-05-01). Because the keeper is
        # logically hot it is RELOCATED out of .cold/ to its hot home.
        assert (v.wiki_dir / "projects" / "svc.md").exists()
        assert not (v.wiki_dir / ".cold" / "projects" / "svc.md").exists()
        kp = parse_page(
            (v.wiki_dir / "projects" / "svc.md").read_text(encoding="utf-8")
        )
        assert kp.status == "hot"
        assert "## merged from projects/svc.md" in kp.body
        assert "COLDBODY" in kp.body  # keeper's own body retained
        assert "HOT" in kp.body  # loser body absorbed
        assert (".cold/projects/svc.md", "projects/svc.md") in rep.merged
        assert ".cold/projects/svc.md -> projects/svc.md" in rep.reheated

    def test_relpath_tiebreak_when_updated_equal(self, tmp_path):
        v = _vault(tmp_path)
        _write(
            v,
            "projects/svc.md",
            _page(type="projects", title="Svc", updated="2026-05-10", body="HOT\n"),
        )
        _write(
            v,
            ".cold/projects/svc.md",
            _page(
                type="projects",
                title="SvcCold",
                status="hot",
                updated="2026-05-10",
                last_touched="2026-05-18",
                body="COLDBODY\n",
            ),
        )
        rep = run_lint(v, TODAY)
        # Tie on `updated` -> keeper is the relpath that sorts first:
        # ".cold/projects/svc.md" < "projects/svc.md" ('.' < 'p'). The
        # keeper is logically hot so it is relocated to its hot home; the
        # loser projects/svc.md is merged in and deleted.
        assert (v.wiki_dir / "projects" / "svc.md").exists()
        assert not (v.wiki_dir / ".cold" / "projects" / "svc.md").exists()
        kp = parse_page(
            (v.wiki_dir / "projects" / "svc.md").read_text(encoding="utf-8")
        )
        assert "## merged from projects/svc.md" in kp.body
        assert (".cold/projects/svc.md", "projects/svc.md") in rep.merged


class TestBrokenLinks:
    def test_broken_link_recorded_not_modified(self, tmp_path):
        v = _vault(tmp_path)
        page = _page(title="Alice", links_out=["projects/ghost"])
        before = serialize_page(page)
        _write(v, "people/alice.md", page)
        rep = run_lint(v, TODAY)

        assert ("people/alice.md", "projects/ghost") in rep.broken_links
        # The page body/frontmatter is untouched by the broken-link phase.
        got = parse_page((v.wiki_dir / "people" / "alice.md").read_text(encoding="utf-8"))
        assert got.links_out == ["projects/ghost"]
        assert serialize_page(got) == before

    def test_link_to_cold_page_not_broken(self, tmp_path):
        v = _vault(tmp_path)
        _write(
            v,
            "people/alice.md",
            _page(title="Alice", links_out=["projects/payment-svc"]),
        )
        _write(
            v,
            ".cold/projects/payment-svc.md",
            _page(type="projects", title="Pay", status="cold"),
        )
        rep = run_lint(v, TODAY)
        assert ("people/alice.md", "projects/payment-svc") not in rep.broken_links


class TestMalformed:
    def test_malformed_file_left_untouched(self, tmp_path):
        v = _vault(tmp_path)
        bad = v.wiki_dir / "people" / "bad.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("garbage not frontmatter", encoding="utf-8")
        rep = run_lint(v, TODAY)

        assert "people/bad.md" in rep.malformed
        assert bad.read_text(encoding="utf-8") == "garbage not frontmatter"
        # Never indexed.
        idx = v.wiki_dir / "people" / "_index.md"
        if idx.exists():
            assert "bad" not in idx.read_text(encoding="utf-8")


class TestIndexRegeneration:
    def test_index_regenerated_deterministically(self, tmp_path):
        v = _vault(tmp_path)
        _write(v, "people/alice.md", _page(title="Alice"))
        _write(v, "people/carol.md", _page(title="Carol"))
        # Stale, wrong index: missing carol, has a ghost stub.
        (v.wiki_dir / "people" / "_index.md").write_text(
            "# people index\n\n- [[people/alice]]\n- [[people/ghost]]\n",
            encoding="utf-8",
        )
        rep = run_lint(v, TODAY)

        idx = (v.wiki_dir / "people" / "_index.md").read_text(encoding="utf-8")
        assert idx == "# people index\n\n- [[people/alice]]\n- [[people/carol]]\n"
        assert "people/_index.md" in rep.indexes_regenerated
        # carol was an orphan (absent from the old index).
        assert any("carol" in o for o in rep.orphans_fixed)

    def test_empty_type_index_emptied_not_deleted(self, tmp_path):
        v = _vault(tmp_path)
        # An _index.md exists but the type has zero hot pages.
        (v.wiki_dir / "projects").mkdir(parents=True, exist_ok=True)
        (v.wiki_dir / "projects" / "_index.md").write_text(
            "# projects index\n\n- [[projects/stale]]\n", encoding="utf-8"
        )
        run_lint(v, TODAY)
        idx = v.wiki_dir / "projects" / "_index.md"
        assert idx.exists()
        assert idx.read_text(encoding="utf-8") == "# projects index\n\n"


class TestMocRegeneration:
    def test_moc_has_map_and_recent(self, tmp_path):
        v = _vault(tmp_path)
        _write(
            v,
            "people/alice.md",
            _page(title="Alice", last_touched="2026-05-10"),
        )
        _write(
            v,
            "projects/svc.md",
            _page(type="projects", title="Svc", last_touched="2026-05-18"),
        )
        rep = run_lint(v, TODAY)
        moc = (v.root / "MEMORY.md").read_text(encoding="utf-8")
        assert moc.startswith("# Memory\n\n")
        assert "## Map\n" in moc
        assert "- [[people/_index]]\n" in moc
        assert "- [[projects/_index]]\n" in moc
        assert "## Recent\n" in moc
        # Recent ordered by last_touched DESC: svc (05-18) before alice (05-10).
        recent = moc.split("## Recent\n", 1)[1]
        assert recent.index("projects/svc") < recent.index("people/alice")
        assert rep.moc_regenerated is True

    def test_moc_respects_max_lines_truncation(self, tmp_path):
        # Tiny moc_max_lines via a custom per-vault SCHEMA so truncation fires.
        root = tmp_path / "memory" / "users" / "u1"
        wiki = root / "wiki"
        wiki.mkdir(parents=True)
        (wiki / "SCHEMA.md").write_text(
            "# Wiki Schema\n```yaml\n"
            "types:\n"
            "  people: { folder: people, cold_after_days: 180 }\n"
            "required_frontmatter: [type, title, status]\n"
            "moc_max_lines: 8\n"
            "```\n",
            encoding="utf-8",
        )
        v = Vault(root)
        # 10 hot pages, distinct last_touched so ordering is total.
        for i in range(10):
            day = 10 + i  # 2026-05-10 .. 2026-05-19
            _write(
                v,
                f"people/p{i:02d}.md",
                _page(title=f"P{i}", last_touched=f"2026-05-{day:02d}"),
            )
        run_lint(v, TODAY)
        moc = (v.root / "MEMORY.md").read_text(encoding="utf-8")
        lines = moc.splitlines()
        assert len(lines) <= 8
        # Newest kept, oldest dropped: p09 (05-19) present, p00 (05-10) absent.
        assert "people/p09" in moc
        assert "people/p00" not in moc


class TestLintLog:
    def test_log_written_then_idempotent(self, tmp_path):
        v = _vault(tmp_path)
        _write(v, "people/alice.md", _page(title="Alice", last_touched="2025-01-01"))
        run_lint(v, TODAY)
        log_path = v.root / ".lint.log"
        assert log_path.exists()
        log1 = log_path.read_bytes()
        assert f"## lint {TODAY.isoformat()}".encode() in log1
        assert b"- cooled people/alice.md" in log1

        # Second run: zero actions -> .lint.log byte-identical.
        rep2 = run_lint(v, TODAY)
        assert log_path.read_bytes() == log1
        assert rep2.changed is False


class TestIdempotenceAndDeterminism:
    def _messy_vault(self, tmp_path, name="u1"):
        root = tmp_path / "memory" / "users" / name
        wiki = root / "wiki"
        wiki.mkdir(parents=True)
        (wiki / "SCHEMA.md").write_text(_bundled_schema_text(), encoding="utf-8")
        v = Vault(root)
        # stale -> cold
        _write(v, "people/alice.md", _page(title="Alice", last_touched="2024-01-01"))
        # hot, fresh
        _write(v, "people/carol.md", _page(title="Carol", last_touched="2026-05-18"))
        # reheated cold (status hot, under .cold)
        _write(
            v,
            ".cold/people/bob.md",
            _page(title="Bob", status="hot", last_touched="2026-05-18"),
        )
        # genuinely cold page (link target)
        _write(
            v,
            ".cold/projects/old.md",
            _page(type="projects", title="Old", status="cold"),
        )
        # dedup pair via the portable reheat-collision pattern (a
        # case-only-distinct filename pair is unrepresentable on a
        # case-insensitive FS). An existing hot projects/svc.md plus a
        # status:hot .cold/projects/svc.md -> reheat-relocate defers,
        # dedup merges, keeper relocated to hot.
        _write(
            v,
            "projects/svc.md",
            _page(type="projects", title="Svc", updated="2026-05-01", body="s1\n"),
        )
        _write(
            v,
            ".cold/projects/svc.md",
            _page(
                type="projects",
                title="Svc2",
                status="hot",
                updated="2026-05-09",
                last_touched="2026-05-18",
                body="s2\n",
            ),
        )
        # broken link + valid cold link
        _write(
            v,
            "concepts/auth.md",
            _page(
                type="concepts",
                title="Auth",
                links_out=["projects/ghost", "projects/old"],
            ),
        )
        # malformed
        (wiki / "concepts" / "bad.md").write_text("not a page", encoding="utf-8")
        # stale agent-written index
        (wiki / "people" / "_index.md").write_text(
            "# people index\n\n- [[people/zzz]]\n", encoding="utf-8"
        )
        return v

    def test_second_run_is_a_total_noop(self, tmp_path):
        v = self._messy_vault(tmp_path)
        run_lint(v, TODAY)
        snap1 = _snapshot(v)
        rep2 = run_lint(v, TODAY)
        snap2 = _snapshot(v)
        assert snap1 == snap2
        assert rep2.changed is False
        assert isinstance(rep2, LintReport)

    def test_two_identical_vaults_yield_identical_bytes(self, tmp_path):
        va = self._messy_vault(tmp_path, name="va")
        vb = self._messy_vault(tmp_path, name="vb")
        run_lint(va, TODAY)
        run_lint(vb, TODAY)
        sa = _snapshot(va)
        sb = _snapshot(vb)
        assert sa == sb
