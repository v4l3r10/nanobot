"""Tests for GitStore — git-backed version control for memory files."""

import pytest
from pathlib import Path

from nanobot.utils.gitstore import GitStore, CommitInfo


TRACKED = ["SOUL.md", "USER.md", "memory/MEMORY.md"]


@pytest.fixture
def git(tmp_path):
    """Uninitialized GitStore."""
    return GitStore(tmp_path, tracked_files=TRACKED)


@pytest.fixture
def git_ready(git):
    """Initialized GitStore with one initial commit."""
    git.init()
    return git


class TestInit:
    def test_not_initialized_by_default(self, git, tmp_path):
        assert not git.is_initialized()
        assert not (tmp_path / ".git").is_dir()

    def test_init_creates_git_dir(self, git, tmp_path):
        assert git.init()
        assert (tmp_path / ".git").is_dir()

    def test_init_idempotent(self, git_ready):
        assert not git_ready.init()

    def test_init_creates_gitignore(self, git_ready):
        gi = git_ready._workspace / ".gitignore"
        assert gi.exists()
        content = gi.read_text(encoding="utf-8")
        for f in TRACKED:
            assert f"!{f}" in content

    def test_init_touches_tracked_files(self, git_ready):
        for f in TRACKED:
            assert (git_ready._workspace / f).exists()

    def test_init_makes_initial_commit(self, git_ready):
        commits = git_ready.log()
        assert len(commits) == 1
        assert "init" in commits[0].message


class TestBuildGitignore:
    def test_subdirectory_dirs(self, git):
        content = git._build_gitignore()
        assert "!memory/\n" in content
        for f in TRACKED:
            assert f"!{f}\n" in content
        assert content.startswith("/*\n")

    def test_root_level_files_no_dir_entries(self, tmp_path):
        gs = GitStore(tmp_path, tracked_files=["a.md", "b.md"])
        content = gs._build_gitignore()
        assert "!a.md\n" in content
        assert "!b.md\n" in content
        dir_lines = [l for l in content.split("\n") if l.startswith("!") and l.endswith("/")]
        assert dir_lines == []


class TestAutoCommit:
    def test_returns_none_when_not_initialized(self, git):
        assert git.auto_commit("test") is None

    def test_commits_file_change(self, git_ready):
        (git_ready._workspace / "SOUL.md").write_text("updated", encoding="utf-8")
        sha = git_ready.auto_commit("update soul")
        assert sha is not None
        assert len(sha) == 8

    def test_returns_none_when_no_changes(self, git_ready):
        assert git_ready.auto_commit("no change") is None

    def test_commit_appears_in_log(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("v2", encoding="utf-8")
        sha = git_ready.auto_commit("update soul")
        commits = git_ready.log()
        assert len(commits) == 2
        assert commits[0].sha == sha

    def test_does_not_create_empty_commits(self, git_ready):
        git_ready.auto_commit("nothing 1")
        git_ready.auto_commit("nothing 2")
        assert len(git_ready.log()) == 1  # only init commit


class TestLog:
    def test_empty_when_not_initialized(self, git):
        assert git.log() == []

    def test_newest_first(self, git_ready):
        ws = git_ready._workspace
        for i in range(3):
            (ws / "SOUL.md").write_text(f"v{i}", encoding="utf-8")
            git_ready.auto_commit(f"commit {i}")

        commits = git_ready.log()
        assert len(commits) == 4  # init + 3
        assert "commit 2" in commits[0].message
        assert "init" in commits[-1].message

    def test_max_entries(self, git_ready):
        ws = git_ready._workspace
        for i in range(10):
            (ws / "SOUL.md").write_text(f"v{i}", encoding="utf-8")
            git_ready.auto_commit(f"c{i}")
        assert len(git_ready.log(max_entries=3)) == 3

    def test_commit_info_fields(self, git_ready):
        c = git_ready.log()[0]
        assert isinstance(c, CommitInfo)
        assert len(c.sha) == 8
        assert c.timestamp
        assert c.message


class TestDiffCommits:
    def test_empty_when_not_initialized(self, git):
        assert git.diff_commits("a", "b") == ""

    def test_diff_between_two_commits(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("original", encoding="utf-8")
        git_ready.auto_commit("v1")
        (ws / "SOUL.md").write_text("modified", encoding="utf-8")
        git_ready.auto_commit("v2")

        commits = git_ready.log()
        diff = git_ready.diff_commits(commits[1].sha, commits[0].sha)
        assert "modified" in diff

    def test_invalid_sha_returns_empty(self, git_ready):
        assert git_ready.diff_commits("deadbeef", "cafebabe") == ""


class TestFindCommit:
    def test_finds_by_prefix(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("v2", encoding="utf-8")
        sha = git_ready.auto_commit("v2")
        found = git_ready.find_commit(sha[:4])
        assert found is not None
        assert found.sha == sha

    def test_returns_none_for_unknown(self, git_ready):
        assert git_ready.find_commit("deadbeef") is None


class TestShowCommitDiff:
    def test_returns_commit_with_diff(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("content", encoding="utf-8")
        sha = git_ready.auto_commit("add content")
        result = git_ready.show_commit_diff(sha)
        assert result is not None
        commit, diff = result
        assert commit.sha == sha
        assert "content" in diff

    def test_first_commit_has_empty_diff(self, git_ready):
        init_sha = git_ready.log()[-1].sha
        result = git_ready.show_commit_diff(init_sha)
        assert result is not None
        _, diff = result
        assert diff == ""

    def test_returns_none_for_unknown(self, git_ready):
        assert git_ready.show_commit_diff("deadbeef") is None


class TestCommitInfoFormat:
    def test_format_with_diff(self):
        from nanobot.utils.gitstore import CommitInfo
        c = CommitInfo(sha="abcd1234", message="test commit\nsecond line", timestamp="2026-04-02 12:00")
        result = c.format(diff="some diff")
        assert "test commit" in result
        assert "`abcd1234`" in result
        assert "some diff" in result

    def test_format_without_diff(self):
        from nanobot.utils.gitstore import CommitInfo
        c = CommitInfo(sha="abcd1234", message="test", timestamp="2026-04-02 12:00")
        result = c.format()
        assert "(no file changes)" in result


class TestRevert:
    def test_returns_none_when_not_initialized(self, git):
        assert git.revert("abc") is None

    def test_undoes_commit_changes(self, git_ready):
        """revert(sha) should undo the given commit by restoring to its parent."""
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("v2 content", encoding="utf-8")
        git_ready.auto_commit("v2")

        commits = git_ready.log()
        # commits[0] = v2 (HEAD), commits[1] = init
        # Revert v2 → restore to init's state (empty SOUL.md)
        new_sha = git_ready.revert(commits[0].sha)
        assert new_sha is not None
        assert (ws / "SOUL.md").read_text(encoding="utf-8") == ""

    def test_root_commit_returns_none(self, git_ready):
        """Cannot revert the root commit (no parent to restore to)."""
        commits = git_ready.log()
        assert len(commits) == 1
        assert git_ready.revert(commits[0].sha) is None

    def test_invalid_sha_returns_none(self, git_ready):
        assert git_ready.revert("deadbeef") is None


class TestMemoryStoreGitProperty:
    def test_git_property_exposes_gitstore(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path)
        assert isinstance(store.git, GitStore)

    def test_git_property_is_same_object(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path)
        assert store.git is store._git


# ---------------------------------------------------------------------------
# Task 5.1 — versioning the per-user wiki vaults under memory/users/**
# ---------------------------------------------------------------------------

# The full static base MemoryStore uses (includes the dream cursor); the
# 5.1 vault expansion is layered on top of this in GitStore.
LEGACY_FOUR = ["SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor"]


def _commit_tree_paths(workspace: Path) -> set[str]:
    """Return the set of repo-relative POSIX paths in the HEAD commit tree."""
    from dulwich.repo import Repo

    paths: set[str] = set()
    with Repo(str(workspace)) as repo:
        head = repo.refs[b"HEAD"]
        commit = repo[head]
        tree = repo[commit.tree]

        def _walk(t, prefix=""):
            for name, _mode, sha in t.items():
                n = name.decode()
                obj = repo[sha]
                if obj.type_name == b"tree":
                    _walk(obj, prefix + n + "/")
                else:
                    paths.add(prefix + n)

        _walk(tree)
    return paths


@pytest.fixture
def vault_git(tmp_path):
    """A GitStore using the same static base as MemoryStore (legacy 4)."""
    return GitStore(tmp_path, tracked_files=list(LEGACY_FOUR))


class TestVaultVersioning:
    def test_no_vault_tracks_exactly_legacy_four(self, vault_git, tmp_path):
        """GOLDEN regression for 5.1: no memory/users/ → byte-identical to before.

        The commit tree must contain exactly the legacy-4 files (plus the
        .gitignore the init writes) and zero memory/users entries. The
        effective-tracked set must equal exactly the static base.
        """
        vault_git.init()
        # init() touches all 4 legacy files, so all are committed at init.
        (tmp_path / "SOUL.md").write_text("soul v2", encoding="utf-8")
        sha = vault_git.auto_commit("update")
        assert sha is not None

        tree = _commit_tree_paths(tmp_path)
        assert tree == {".gitignore", *LEGACY_FOUR}
        assert not any(p.startswith("memory/users") for p in tree)

        # No vault dir → effective set is exactly the static base.
        assert vault_git._effective_tracked_files() == list(LEGACY_FOUR)

        # .gitignore only gains a single additive allow-rule that matches
        # nothing here (no memory/users/ dir exists in a stock workspace).
        gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert "!memory/users/**\n" in gi
        # No trailing-slash vault dir rule — keeps the legacy dir-entry
        # contract intact for root-only tracked sets.
        assert "!memory/users/\n" not in gi

    def test_vault_files_are_committed(self, vault_git, tmp_path):
        """Vault files (incl. dot dirs/files, nested) must be staged."""
        vault_git.init()
        base = tmp_path / "memory" / "users" / "unified_default"
        (base / "wiki" / "people").mkdir(parents=True, exist_ok=True)
        (base / "wiki" / ".cold" / "concepts").mkdir(parents=True, exist_ok=True)
        (base / "MEMORY.md").write_text("vault mem", encoding="utf-8")
        (base / "wiki" / "SCHEMA.md").write_text("schema", encoding="utf-8")
        (base / "wiki" / "people" / "alice.md").write_text("alice", encoding="utf-8")
        (base / "wiki" / "people" / "_index.md").write_text("idx", encoding="utf-8")
        (base / "wiki" / ".cold" / "concepts" / "old.md").write_text("old", encoding="utf-8")
        (base / ".lint.log").write_text("lint", encoding="utf-8")

        sha = vault_git.auto_commit("ingest vault")
        assert sha is not None

        tree = _commit_tree_paths(tmp_path)
        pfx = "memory/users/unified_default/"
        for rel in [
            "MEMORY.md",
            "wiki/SCHEMA.md",
            "wiki/people/alice.md",
            "wiki/people/_index.md",
            "wiki/.cold/concepts/old.md",
            ".lint.log",
        ]:
            assert pfx + rel in tree, f"{pfx + rel} missing from commit tree"

    def test_revert_restores_vault(self, vault_git, tmp_path):
        """revert() must restore the vault tree (5.2 /dream-restore relies on this).

        Documented revert semantics: revert(C) restores every path that
        exists in C's PARENT tree to its parent-state content, AND deletes
        any currently-tracked file that is absent from the parent tree
        (i.e. files the reverted commit *added* are removed). This makes
        "restore the wiki to a prior commit" correct for files present in
        one tree but not the other.
        """
        base = tmp_path / "memory" / "users" / "unified_default" / "wiki" / "people"
        base.mkdir(parents=True, exist_ok=True)
        alice = base / "alice.md"
        bob = base / "bob.md"

        vault_git.init()
        alice.write_text("A", encoding="utf-8")
        vault_git.auto_commit("state A")  # alice=A, no bob

        alice.write_text("B", encoding="utf-8")
        bob.write_text("bob exists", encoding="utf-8")
        sha_b = vault_git.auto_commit("state B")  # alice=B, bob added
        assert sha_b is not None

        # Revert state B → undo it → back to state A (alice=A, bob gone)
        new_sha = vault_git.revert(sha_b)
        assert new_sha is not None
        assert alice.read_text(encoding="utf-8") == "A"
        assert not bob.exists(), "bob.md added in B must be removed by reverting B"

        tree = _commit_tree_paths(tmp_path)
        pfx = "memory/users/unified_default/wiki/people/"
        assert pfx + "alice.md" in tree
        assert pfx + "bob.md" not in tree

    def test_revert_restores_legacy_alongside_vault(self, vault_git, tmp_path):
        """Reverting also rolls back the legacy 4, not just the vault."""
        vault_git.init()
        vault = tmp_path / "memory" / "users" / "unified_default" / "wiki"
        vault.mkdir(parents=True, exist_ok=True)
        (tmp_path / "SOUL.md").write_text("soul A", encoding="utf-8")
        (vault / "SCHEMA.md").write_text("schema A", encoding="utf-8")
        vault_git.auto_commit("state A")

        (tmp_path / "SOUL.md").write_text("soul B", encoding="utf-8")
        (vault / "SCHEMA.md").write_text("schema B", encoding="utf-8")
        sha_b = vault_git.auto_commit("state B")
        assert sha_b is not None

        assert vault_git.revert(sha_b) is not None
        assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "soul A"
        assert (vault / "SCHEMA.md").read_text(encoding="utf-8") == "schema A"

    def test_multiuser_vaults_all_tracked(self, vault_git, tmp_path):
        """Multiple vault slugs are all tracked, no cross-omission."""
        vault_git.init()
        for slug in ("telegram_1", "telegram_2"):
            d = tmp_path / "memory" / "users" / slug / "wiki" / "people"
            d.mkdir(parents=True, exist_ok=True)
            (d / "page.md").write_text(f"page for {slug}", encoding="utf-8")

        sha = vault_git.auto_commit("two vaults")
        assert sha is not None

        tree = _commit_tree_paths(tmp_path)
        assert "memory/users/telegram_1/wiki/people/page.md" in tree
        assert "memory/users/telegram_2/wiki/people/page.md" in tree

    def test_gitignore_allows_vault_excludes_other(self, vault_git, tmp_path):
        """The /* deny still blocks non-allowlisted paths (no over-widening)."""
        vault_git.init()
        base = tmp_path / "memory" / "users" / "unified_default" / "wiki"
        base.mkdir(parents=True, exist_ok=True)
        (base / "SCHEMA.md").write_text("schema", encoding="utf-8")

        # Stray files OUTSIDE the tracked set.
        (tmp_path / "sessions").mkdir(exist_ok=True)
        (tmp_path / "sessions" / "foo.jsonl").write_text("stray", encoding="utf-8")
        (tmp_path / "scratch.txt").write_text("stray", encoding="utf-8")

        sha = vault_git.auto_commit("vault + strays present")
        assert sha is not None

        tree = _commit_tree_paths(tmp_path)
        assert "memory/users/unified_default/wiki/SCHEMA.md" in tree
        assert "sessions/foo.jsonl" not in tree
        assert "scratch.txt" not in tree

    def test_effective_tracked_files_is_sorted_and_deterministic(self, vault_git, tmp_path):
        """The scan must be deterministic (sorted), base first."""
        d = tmp_path / "memory" / "users"
        (d / "z_slug" / "wiki").mkdir(parents=True, exist_ok=True)
        (d / "a_slug" / "wiki").mkdir(parents=True, exist_ok=True)
        (d / "z_slug" / "wiki" / "p.md").write_text("z", encoding="utf-8")
        (d / "a_slug" / "MEMORY.md").write_text("a", encoding="utf-8")

        eff1 = vault_git._effective_tracked_files()
        eff2 = vault_git._effective_tracked_files()
        assert eff1 == eff2
        assert eff1[: len(LEGACY_FOUR)] == list(LEGACY_FOUR)
        vault_part = eff1[len(LEGACY_FOUR):]
        assert vault_part == sorted(vault_part)
        assert "memory/users/a_slug/MEMORY.md" in vault_part
        assert "memory/users/z_slug/wiki/p.md" in vault_part


# ---------------------------------------------------------------------------
# Code-review follow-up to Task 5.1 — revert must be a true per-commit
# inverse (C1 data loss), atomic (C2), symlink-safe (I1), single-scan (I2).
# These are the contract /dream-restore (Task 5.2) relies on.
# ---------------------------------------------------------------------------


class TestRevertPerCommitInverse:
    def test_revert_nontip_preserves_later_files(self, vault_git, tmp_path):
        """C1 regression: reverting a NON-tip commit must undo ONLY that
        commit and must NOT destroy vault files created by *later* commits.

        On the pre-fix code this fails: revert(B) reverts to B's parent
        tree over the whole effective set, so the unrelated page p2 added
        by the later commit C (not in B's parent tree, on disk) is deleted
        and that deletion is committed — silent committed data loss.
        """
        people = tmp_path / "memory" / "users" / "unified_default" / "wiki" / "people"
        people.mkdir(parents=True, exist_ok=True)
        p1 = people / "p1.md"
        p2 = people / "p2.md"

        vault_git.init()
        p1.write_text("A", encoding="utf-8")
        vault_git.auto_commit("commit A: p1=A")  # A

        p1.write_text("B", encoding="utf-8")
        sha_b = vault_git.auto_commit("commit B: p1=B")  # B (modifies p1)
        assert sha_b is not None

        p2.write_text("P2", encoding="utf-8")
        sha_c = vault_git.auto_commit("commit C: add unrelated p2")  # C
        assert sha_c is not None

        # Revert the NON-tip commit B. B only modified p1 (A→B).
        new_sha = vault_git.revert(sha_b)
        assert new_sha is not None

        # B's modify is undone …
        assert p1.read_text(encoding="utf-8") == "A"
        # … but C's later, unrelated add survives untouched.
        assert p2.exists(), "p2 added by later commit C must NOT be destroyed"
        assert p2.read_text(encoding="utf-8") == "P2"

        tree = _commit_tree_paths(tmp_path)
        pfx = "memory/users/unified_default/wiki/people/"
        assert pfx + "p1.md" in tree
        assert pfx + "p2.md" in tree, "the revert commit must not record p2's deletion"

        # And the revert commit itself recorded NO deletion of p2: the diff
        # of the revert vs its parent must not remove p2.
        revert_diff = vault_git.diff_commits(sha_c, new_sha)
        assert "p2.md" not in revert_diff, (
            "reverting B must not touch p2 at all (no spurious deletion)"
        )

    def test_revert_of_adding_commit_preserves_later_unrelated_file(
        self, vault_git, tmp_path
    ):
        """Reverting a commit that ADDED a file removes only that file;
        a later unrelated file is preserved."""
        people = tmp_path / "memory" / "users" / "unified_default" / "wiki" / "people"
        people.mkdir(parents=True, exist_ok=True)
        base = people / "base.md"
        added = people / "added.md"
        later = people / "later.md"

        vault_git.init()
        base.write_text("base", encoding="utf-8")
        vault_git.auto_commit("A: base")  # A

        added.write_text("added by B", encoding="utf-8")
        sha_b = vault_git.auto_commit("B: add 'added'")  # B adds added.md
        assert sha_b is not None

        later.write_text("later", encoding="utf-8")
        vault_git.auto_commit("C: add unrelated 'later'")  # C adds later.md

        # Revert B (a non-tip commit that ADDED added.md).
        assert vault_git.revert(sha_b) is not None

        assert not added.exists(), "added.md (added by reverted B) must be removed"
        assert later.exists(), "later.md (added by later C) must be preserved"
        assert later.read_text(encoding="utf-8") == "later"
        assert base.read_text(encoding="utf-8") == "base"

        tree = _commit_tree_paths(tmp_path)
        pfx = "memory/users/unified_default/wiki/people/"
        assert pfx + "base.md" in tree
        assert pfx + "added.md" not in tree
        assert pfx + "later.md" in tree

    def test_revert_leaves_unchanged_legacy_files_untouched(self, vault_git, tmp_path):
        """A revert must touch ONLY the affected paths; unrelated unchanged
        legacy files must not even be rewritten (per-commit inverse, not a
        whole-tree reset)."""
        vault = tmp_path / "memory" / "users" / "unified_default" / "wiki"
        vault.mkdir(parents=True, exist_ok=True)
        vault_git.init()
        (tmp_path / "SOUL.md").write_text("soul stays", encoding="utf-8")
        (vault / "SCHEMA.md").write_text("schema A", encoding="utf-8")
        vault_git.auto_commit("A")  # A: SOUL set, SCHEMA=A

        (vault / "SCHEMA.md").write_text("schema B", encoding="utf-8")
        sha_b = vault_git.auto_commit("B: only SCHEMA changes")  # B
        assert sha_b is not None

        soul_mtime_before = (tmp_path / "SOUL.md").stat().st_mtime_ns
        assert vault_git.revert(sha_b) is not None

        assert (vault / "SCHEMA.md").read_text(encoding="utf-8") == "schema A"
        # SOUL.md was not part of commit B → must be byte- and mtime-stable.
        assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "soul stays"
        assert (tmp_path / "SOUL.md").stat().st_mtime_ns == soul_mtime_before, (
            "an unchanged legacy file must not be rewritten by revert"
        )


class TestRevertAtomicAndScan:
    def test_revert_is_atomic_per_file(self, vault_git, tmp_path, monkeypatch):
        """C2: every file rewrite in revert goes through atomic_write_text
        (no raw write_text), so a crash mid-revert leaves each file fully
        old or fully new — never truncated."""
        import nanobot.utils.gitstore as gs

        people = tmp_path / "memory" / "users" / "unified_default" / "wiki" / "people"
        people.mkdir(parents=True, exist_ok=True)
        page = people / "alice.md"

        vault_git.init()
        page.write_text("A", encoding="utf-8")
        vault_git.auto_commit("A")
        page.write_text("B", encoding="utf-8")
        sha_b = vault_git.auto_commit("B")
        assert sha_b is not None

        calls: list[str] = []
        real = gs.atomic_write_text

        def _spy(path, content, **kw):
            calls.append(str(path))
            return real(path, content, **kw)

        monkeypatch.setattr(gs, "atomic_write_text", _spy)

        assert vault_git.revert(sha_b) is not None
        assert page.read_text(encoding="utf-8") == "A"
        # The reverted page was rewritten via the atomic helper.
        assert any(str(page) == c or c.endswith("alice.md") for c in calls), (
            "revert must rewrite files via atomic_write_text, not write_text"
        )
        # No stale temp file left behind anywhere in the vault.
        assert not list(
            (tmp_path / "memory").rglob("*.tmp")
        ), "atomic write must not leave a .tmp file"

    def test_revert_double_scan_avoided(self, vault_git, tmp_path):
        """I2: a single revert must not scan the unbounded vault tree more
        than once (revert scans, then triggers auto_commit which would
        scan AGAIN). Bounded invariant: <= 1 vault scan per revert."""
        people = tmp_path / "memory" / "users" / "unified_default" / "wiki" / "people"
        people.mkdir(parents=True, exist_ok=True)
        page = people / "alice.md"

        vault_git.init()
        page.write_text("A", encoding="utf-8")
        vault_git.auto_commit("A")
        page.write_text("B", encoding="utf-8")
        sha_b = vault_git.auto_commit("B")
        assert sha_b is not None

        original = GitStore._scan_vault_files
        count = {"n": 0}

        def _counting(self):
            count["n"] += 1
            return original(self)

        GitStore._scan_vault_files = _counting
        try:
            assert vault_git.revert(sha_b) is not None
        finally:
            GitStore._scan_vault_files = original

        assert count["n"] <= 1, (
            f"revert scanned the vault {count['n']}x; bounded invariant is <=1"
        )


class TestScanSymlinkSafety:
    def test_scan_excludes_symlinks(self, vault_git, tmp_path):
        """I1: a symlink inside the agent-writable vault pointing OUTSIDE
        the workspace must never be scanned/committed (exfiltration)."""
        import os

        secret_dir = tmp_path.parent / "outside_secret_dir"
        secret_dir.mkdir(parents=True, exist_ok=True)
        secret = secret_dir / "secret.txt"
        secret.write_text("TOP SECRET", encoding="utf-8")

        wiki = tmp_path / "memory" / "users" / "unified_default" / "wiki"
        wiki.mkdir(parents=True, exist_ok=True)
        (wiki / "real.md").write_text("legit page", encoding="utf-8")

        link = wiki / "leak.md"
        try:
            os.symlink(secret, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted in this environment")

        scanned = vault_git._scan_vault_files()
        assert any(p.endswith("real.md") for p in scanned)
        assert all("leak.md" not in p for p in scanned), (
            "a symlink inside the vault must be excluded from the scan"
        )

        vault_git.init()
        sha = vault_git.auto_commit("commit vault with a symlink present")
        assert sha is not None
        tree = _commit_tree_paths(tmp_path)
        assert "memory/users/unified_default/wiki/real.md" in tree
        assert "memory/users/unified_default/wiki/leak.md" not in tree
        # The external secret content never entered the commit tree.
        from dulwich.repo import Repo

        with Repo(str(tmp_path)) as repo:
            blob_data = b""
            head = repo.refs[b"HEAD"]
            for entry in repo.get_walker():
                tree_obj = repo[entry.commit.tree]
                for _, _, sha_b in _iter_all_blobs(repo, tree_obj):
                    blob_data += repo[sha_b].data
        assert b"TOP SECRET" not in blob_data


def _iter_all_blobs(repo, tree, prefix=""):
    for name, _mode, sha in tree.items():
        obj = repo[sha]
        if obj.type_name == b"tree":
            yield from _iter_all_blobs(repo, obj, prefix + name.decode() + "/")
        elif obj.type_name == b"blob":
            yield prefix + name.decode(), _mode, sha
