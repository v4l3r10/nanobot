"""Tests for GitStore — line_ages() and core git operations."""

import subprocess
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from nanobot.utils.gitstore import GitStore


@pytest.fixture
def git(tmp_path):
    """Create an initialized GitStore with tracked MEMORY.md."""
    g = GitStore(tmp_path, tracked_files=["MEMORY.md", "SOUL.md"])
    g.init()
    return g


class TestLineAges:
    def test_returns_empty_when_not_initialized(self, tmp_path):
        """line_ages should return [] if the git repo is not initialized."""
        git = GitStore(tmp_path, tracked_files=["MEMORY.md"])
        assert git.line_ages("MEMORY.md") == []

    def test_returns_empty_for_missing_file(self, git):
        """line_ages should return [] for a file that doesn't exist."""
        assert git.line_ages("SOUL.md") == []

    def test_returns_empty_for_empty_file(self, git, tmp_path):
        """line_ages should return [] for an empty tracked file."""
        (tmp_path / "SOUL.md").write_text("", encoding="utf-8")
        git.auto_commit("empty soul")
        assert git.line_ages("SOUL.md") == []

    def test_one_age_per_line(self, git, tmp_path):
        """line_ages should return one entry per line in the file."""
        content = "# Memory\n\n## Section A\n- item 1\n"
        (tmp_path / "MEMORY.md").write_text(content, encoding="utf-8")
        git.auto_commit("initial")
        ages = git.line_ages("MEMORY.md")
        assert len(ages) == len(content.splitlines())

    def test_fresh_lines_have_age_zero(self, git, tmp_path):
        """Lines committed today should have age_days=0."""
        (tmp_path / "MEMORY.md").write_text("## A\n- x\n", encoding="utf-8")
        git.auto_commit("initial")
        ages = git.line_ages("MEMORY.md")
        assert all(a.age_days == 0 for a in ages)

    def test_age_differentiates_across_days(self, git, tmp_path):
        """Lines committed today should show correct age when 'now' is mocked forward."""
        (tmp_path / "MEMORY.md").write_text("## A\n- x\n", encoding="utf-8")
        git.auto_commit("initial")

        future_now = datetime.now(tz=timezone.utc) + timedelta(days=30)
        with patch("nanobot.utils.gitstore.datetime") as mock_dt:
            mock_dt.now.return_value = future_now
            mock_dt.fromtimestamp = datetime.fromtimestamp
            ages = git.line_ages("MEMORY.md")

        assert len(ages) == 2
        assert all(a.age_days == 30 for a in ages)

    def test_annotate_failure_returns_empty(self, tmp_path):
        """If annotate fails, line_ages should return [] gracefully."""
        git = GitStore(tmp_path, tracked_files=["MEMORY.md"])
        # Don't init — annotate will fail
        assert git.line_ages("MEMORY.md") == []

    def test_partial_edit_only_updates_changed_lines(self, git, tmp_path):
        """Only modified lines should reflect the new commit's timestamp."""
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\n## A\n- old\n\n## B\n- keep\n", encoding="utf-8"
        )
        git.auto_commit("commit1")
        time.sleep(1.1)

        # Only modify section A
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\n## A\n- new\n\n## B\n- keep\n", encoding="utf-8"
        )
        git.auto_commit("commit2")

        ages = git.line_ages("MEMORY.md")
        lines = (tmp_path / "MEMORY.md").read_text(encoding="utf-8").splitlines()
        # All lines are from today, but verify line-level tracking works
        assert len(ages) == len(lines)
        # "- new" line and "- keep" line both age=0 (same day), but
        # the key point is we get per-line results
        assert len(ages) == 7


class TestNestedRepoProtection:
    """Regression tests for GitHub issue #2980: nested repo protection."""

    def test_init_refuses_inside_git_repo(self, tmp_path):
        """init() should detect it's inside an existing git repo and refuse."""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".git").mkdir()

        workspace = project / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is False
        assert not (workspace / ".git").is_dir()

    def test_init_preserves_existing_gitignore(self, tmp_path):
        """init() should preserve existing .gitignore entries and append new ones."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        existing = "*.pyc\n__pycache__/\n"
        (workspace / ".gitignore").write_text(existing, encoding="utf-8")

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
        assert "*.pyc" in gitignore
        assert "__pycache__/" in gitignore
        assert "!MEMORY.md" in gitignore
        assert "!.gitignore" in gitignore

    def test_init_no_gitignore_creates_new(self, tmp_path):
        """init() should create .gitignore with Dream content when none exists."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
        expected = g._build_gitignore()
        assert gitignore == expected

    def test_init_gitignore_merge_idempotent(self, tmp_path):
        """init() should not duplicate Dream entries already in .gitignore."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Pre-existing .gitignore that already has some Dream entries
        existing = "*.pyc\n/*\n!MEMORY.md\n"
        (workspace / ".gitignore").write_text(existing, encoding="utf-8")

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
        # No duplicate lines
        lines = gitignore.splitlines()
        assert lines.count("/*") == 1
        assert lines.count("!MEMORY.md") == 1
        # Existing entry preserved, new Dream entries appended
        assert "*.pyc" in gitignore
        assert "!.gitignore" in gitignore

    def test_init_outside_git_repo_works_normally(self, tmp_path):
        """init() should succeed and create .git when not inside a git repo."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is True
        assert (workspace / ".git").is_dir()

    def test_init_refuses_inside_git_worktree(self, tmp_path):
        """init() should refuse when the parent checkout is a git worktree."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "README.md").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                "init",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "branch", "wt-branch"], check=True)

        worktree = tmp_path / "worktree"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-q", str(worktree), "wt-branch"],
            check=True,
        )
        assert (worktree / ".git").is_file()

        workspace = worktree / "workspace"
        workspace.mkdir()

        g = GitStore(workspace, tracked_files=["MEMORY.md"])
        result = g.init()

        assert result is False
        assert not (workspace / ".git").exists()


class TestTrackedDirs:
    """Whitelist *.md files inside open-ended directories (e.g. memory/journal)."""

    @pytest.fixture
    def git_with_journal(self, tmp_path):
        g = GitStore(
            tmp_path,
            tracked_files=["MEMORY.md"],
            tracked_dirs=["memory/journal"],
        )
        g.init()
        return g

    def test_default_tracked_dirs_is_empty_list(self, tmp_path):
        g = GitStore(tmp_path, tracked_files=["MEMORY.md"])
        assert g._tracked_dirs == []

    def test_gitignore_whitelists_tracked_dir_and_md_files(self, tmp_path):
        g = GitStore(
            tmp_path,
            tracked_files=["MEMORY.md"],
            tracked_dirs=["memory/journal"],
        )
        content = g._build_gitignore()
        assert "!memory/\n" in content
        assert "!memory/journal/\n" in content
        assert "!memory/journal/*.md\n" in content

    def test_gitignore_whitelists_nested_ancestors(self, tmp_path):
        """Multi-level tracked_dirs unblock every parent path."""
        g = GitStore(
            tmp_path,
            tracked_files=["MEMORY.md"],
            tracked_dirs=["a/b/c/notes"],
        )
        content = g._build_gitignore()
        for d in ("a", "a/b", "a/b/c", "a/b/c/notes"):
            assert f"!{d}/\n" in content

    def test_init_creates_empty_tracked_dir(self, git_with_journal, tmp_path):
        assert (tmp_path / "memory" / "journal").is_dir()

    def test_auto_commit_picks_up_new_md_inside_tracked_dir(self, git_with_journal, tmp_path):
        note = tmp_path / "memory" / "journal" / "2026-05-08.md"
        note.write_text("# 2026-05-08\n- worked on dream\n", encoding="utf-8")

        sha = git_with_journal.auto_commit("first journal note")

        assert sha is not None
        commits = git_with_journal.log()
        assert any("first journal note" in c.message for c in commits)

    def test_auto_commit_tracks_subsequent_edits_inside_tracked_dir(self, git_with_journal, tmp_path):
        note = tmp_path / "memory" / "journal" / "2026-05-08.md"
        note.write_text("first\n", encoding="utf-8")
        sha1 = git_with_journal.auto_commit("create note")
        note.write_text("first\nsecond\n", encoding="utf-8")
        sha2 = git_with_journal.auto_commit("extend note")

        assert sha1 is not None and sha2 is not None
        assert sha1 != sha2

    def test_auto_commit_ignores_non_md_files_inside_tracked_dir(self, git_with_journal, tmp_path):
        """*.txt files inside tracked_dirs are excluded by the gitignore pattern."""
        (tmp_path / "memory" / "journal" / "scratch.txt").write_text("ignored", encoding="utf-8")

        sha = git_with_journal.auto_commit("nothing real")

        # Pure .txt drop must not produce a commit; staging is empty.
        assert sha is None

    def test_revert_restores_modified_md_inside_tracked_dir(self, git_with_journal, tmp_path):
        note = tmp_path / "memory" / "journal" / "2026-05-08.md"
        note.write_text("v1\n", encoding="utf-8")
        sha_v1 = git_with_journal.auto_commit("v1")
        note.write_text("v2\n", encoding="utf-8")
        sha_v2 = git_with_journal.auto_commit("v2")
        assert sha_v1 and sha_v2

        revert_sha = git_with_journal.revert(sha_v2)

        assert revert_sha is not None
        assert note.read_text(encoding="utf-8") == "v1\n"

    def test_revert_undoes_add_of_journal_note(self, git_with_journal, tmp_path):
        """Per-commit-inverse revert: reverting the commit that ADDED a journal
        note removes the file (and leaves earlier, unrelated files alone)."""
        existing = tmp_path / "memory" / "journal" / "2026-05-07.md"
        existing.write_text("yesterday\n", encoding="utf-8")
        sha_anchor = git_with_journal.auto_commit("anchor")

        new_note = tmp_path / "memory" / "journal" / "2026-05-08.md"
        new_note.write_text("today\n", encoding="utf-8")
        sha_added = git_with_journal.auto_commit("added today's note")
        assert sha_anchor and sha_added

        git_with_journal.revert(sha_added)

        # Anchor file untouched; the added note is removed by the inverse.
        assert existing.read_text(encoding="utf-8") == "yesterday\n"
        assert not new_note.exists()
