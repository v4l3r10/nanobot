"""Git-backed version control for memory files, using dulwich."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from nanobot.utils.atomic import atomic_write_text


@dataclass
class CommitInfo:
    sha: str  # Short SHA (8 chars)
    message: str
    timestamp: str  # Formatted datetime

    def format(self, diff: str = "") -> str:
        """Format this commit for display, optionally with a diff."""
        header = f"## {self.message.splitlines()[0]}\n`{self.sha}` — {self.timestamp}\n"
        if diff:
            return f"{header}\n```diff\n{diff}\n```"
        return f"{header}\n(no file changes)"


@dataclass
class LineAge:
    """Age of a single line based on git blame."""

    age_days: int  # days since last modification


def _compute_line_ages(annotated) -> list[LineAge]:
    """Convert annotate results to per-line ages."""
    now = datetime.now(tz=timezone.utc).date()
    ages: list[LineAge] = []
    for (commit, _tree_entry), _line_bytes in annotated:
        dt = datetime.fromtimestamp(commit.commit_time, tz=timezone.utc).date()
        ages.append(LineAge(age_days=(now - dt).days))
    return ages


class GitStore:
    """Git-backed version control for memory files."""

    def __init__(
        self,
        workspace: Path,
        tracked_files: list[str],
        tracked_dirs: list[str] | None = None,
    ):
        self._workspace = workspace
        self._tracked_files = tracked_files
        # Tracked directories whitelist *.md files inside the given relative
        # paths (e.g. "memory/journal" tracks every memory/journal/*.md).
        # Used by Dream's per-day journal layer where the file set is open-
        # ended and fixed-list tracked_files would not work.
        self._tracked_dirs = list(tracked_dirs or [])

    def is_initialized(self) -> bool:
        """Check if the git repo has been initialized."""
        return (self._workspace / ".git").is_dir()

    # -- effective tracked set -------------------------------------------------

    # Per-user wiki vaults live under this prefix (see
    # ``nanobot.agent.wiki.paths.vault_dir``: ``memory/users/<slug>``). They
    # are NOT in the static ``_tracked_files`` base; instead they are scanned
    # dynamically so a stock / wiki-disabled workspace (no ``memory/users/``)
    # commits byte-identically to before — exactly the static base.
    _VAULTS_REL = ("memory", "users")

    def _scan_vault_files(self) -> list[str]:
        """Deterministically scan every file under ``memory/users/**``.

        Returns repo-relative POSIX paths, sorted. Empty when the directory
        is absent (the stock / wiki-disabled case → no behavior change).

        Security (I1 — symlink exfiltration): ``memory/users/`` is the
        agent-writable vault root. ``rglob`` follows symlinks and CPython's
        traversal of symlinked *directories* is version-dependent, so we
        are explicit rather than relying on ``rglob`` semantics: a candidate
        is included only if it is a regular file, is not itself a symlink,
        has **no** symlinked component anywhere between the vault root and
        the file, and its real (fully resolved) path still lies under the
        workspace. Any path failing these (e.g. ``leak.md -> ../secret``)
        is silently skipped so external content can never enter a commit.

        Scalability (I2 — documented forward-looking characteristic): this
        is an O(total entries under ``memory/users/**``) directory walk and
        it runs on **every** ``auto_commit`` (i.e. every Dream cycle, under
        the Dream lock) and once per :meth:`revert`. The design's ``.cold/``
        archive is an unbounded floor (git holds history), so this cost
        grows monotonically with vault size. Cross-cycle caching is
        deliberately out of scope here (YAGNI); :meth:`revert` already
        avoids the redundant second scan (see I2 there). Later milestones
        inherit this note: if vault size becomes a latency problem, the fix
        is an incremental/index-backed scan, not ad-hoc caching.
        """
        root = self._workspace.joinpath(*self._VAULTS_REL)
        if not root.is_dir():
            return []
        try:
            ws_real = self._workspace.resolve()
            root_real = root.resolve()
        except OSError:
            return []
        found: list[str] = []
        for p in root.rglob("*"):
            try:
                if not p.is_file() or p.is_symlink():
                    # Not a regular file, or the leaf itself is a symlink.
                    continue
                # Reject any symlinked component between the vault root and
                # the file (rglob may have descended through one), and make
                # sure the real target stays inside the workspace.
                real = p.resolve()
                if root_real not in real.parents and real != root_real:
                    continue
                if ws_real not in real.parents:
                    continue
                rel = p.relative_to(self._workspace)
            except OSError:
                continue
            found.append(rel.as_posix())
        return sorted(found)

    def _effective_tracked_files(self) -> list[str]:
        """The full tracked set at the moment of commit/revert.

        ``static base + tracked_dirs/*.md + sorted(files under memory/users/**)``.
        The static base is preserved verbatim and first so anything else
        reading ``self._tracked_files`` (init touch, gitignore base) is
        unaffected.
        """
        return (
            list(self._tracked_files)
            + self._enumerate_tracked_dir_files()
            + self._scan_vault_files()
        )

    # -- init ------------------------------------------------------------------

    def init(self) -> bool:
        """Initialize a git repo if not already initialized.

        Creates .gitignore and makes an initial commit.
        Returns True if a new repo was created, False if already exists.
        """
        if self.is_initialized():
            return False

        if self._is_inside_git_repo():
            logger.warning(
                "Workspace {} is already inside a git repo; "
                "skipping nested repo initialization",
                self._workspace,
            )
            return False

        try:
            from dulwich import porcelain

            porcelain.init(str(self._workspace))

            # Write .gitignore (merge with existing if present)
            gitignore = self._workspace / ".gitignore"
            dream_entries = self._build_gitignore()
            if gitignore.exists():
                existing = gitignore.read_text(encoding="utf-8")
                existing_lines = set(existing.splitlines())
                new_lines = [
                    line
                    for line in dream_entries.splitlines()
                    if line not in existing_lines
                ]
                if new_lines:
                    merged = existing.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
                    gitignore.write_text(merged, encoding="utf-8")
            else:
                gitignore.write_text(dream_entries, encoding="utf-8")

            # Ensure tracked files exist (touch them if missing) so the initial
            # commit has something to track.
            for rel in self._tracked_files:
                p = self._workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text("", encoding="utf-8")

            # Tracked dirs are created empty — git itself does not track empty
            # directories, but they will be picked up the moment a file lands
            # inside (e.g. the first journal note Dream writes).
            for rel in self._tracked_dirs:
                (self._workspace / rel).mkdir(parents=True, exist_ok=True)

            # Initial commit
            porcelain.add(
                str(self._workspace),
                paths=[".gitignore"] + self._tracked_files + self._enumerate_tracked_dir_files(),
            )
            porcelain.commit(
                str(self._workspace),
                message=b"init: nanobot memory store",
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            logger.info("Git store initialized at {}", self._workspace)
            return True
        except Exception:
            logger.exception("Git store init failed for {}", self._workspace)
            return False

    # -- daily operations ------------------------------------------------------

    def auto_commit(
        self,
        message: str,
        extra_paths: list[str] | None = None,
        _add_paths: list[str] | None = None,
    ) -> str | None:
        """Stage tracked memory files and commit if there are changes.

        ``extra_paths`` lets callers (notably :meth:`revert`) pass paths
        that must be force-staged this commit. Two uses:

        * Paths that no longer exist on disk so their *deletion* is staged —
          the dynamic ``memory/users/**`` scan only sees files that still
          exist, so a reverted-away wiki page would otherwise linger.
        * Paths :meth:`revert` recreated/rewrote whose ``porcelain.status``
          classification is dulwich-version dependent. A statically-tracked
          legacy file that ``revert`` recreates (absent from the HEAD index)
          may be reported by some dulwich builds as *clean* — not unstaged,
          not staged, not untracked. Passing it here makes ``not
          extra_paths`` False so the no-op short-circuit cannot fire and the
          recovery is deterministically ``porcelain.add``-ed and committed,
          regardless of the status classification. ``porcelain.add``
          tolerates the mixed set (recreated files exist; deleted ones do
          not — a missing tracked path is staged as a deletion).

        A non-empty ``extra_paths`` therefore *guarantees* a commit attempt
        whenever the caller actually changed something; the status-gated
        no-op short-circuit only applies when ``extra_paths`` is empty.

        ``_add_paths`` is an internal optimization (I2): :meth:`revert`
        already computed the effective tracked set for its own work, so it
        passes it through here to avoid a redundant second
        :meth:`_scan_vault_files` walk of the (unbounded) vault within a
        single revert. External callers must not use it.

        Returns the short commit SHA, or None if nothing to commit.
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich import porcelain

            # .gitignore excludes everything except tracked files, so any
            # staged/unstaged/untracked change must be in our files. New
            # per-user vault files (Task 5.1) and journal notes appear as
            # *untracked* until first committed — count them too, otherwise
            # a freshly created wiki vault or journal note would never be
            # versioned. gitignore guarantees untracked entries are only
            # allowlisted paths (strays are ignored, not untracked).
            st = porcelain.status(str(self._workspace))
            if (
                not st.unstaged
                and not any(st.staged.values())
                and not st.untracked
                and not extra_paths
            ):
                return None

            msg_bytes = message.encode("utf-8") if isinstance(message, str) else message
            add_paths = (
                list(_add_paths)
                if _add_paths is not None
                else self._effective_tracked_files()
            )
            if extra_paths:
                seen = set(add_paths)
                add_paths += [p for p in extra_paths if p not in seen]
            porcelain.add(str(self._workspace), paths=add_paths)
            sha_bytes = porcelain.commit(
                str(self._workspace),
                message=msg_bytes,
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            if sha_bytes is None:
                return None
            sha = sha_bytes.hex()[:8]
            logger.debug("Git auto-commit: {} ({})", sha, message)
            return sha
        except Exception:
            logger.exception("Git auto-commit failed: {}", message)
            return None

    # -- internal helpers ------------------------------------------------------

    def _enumerate_tracked_dir_files(self) -> list[str]:
        """Return workspace-relative paths of every ``*.md`` inside tracked_dirs.

        Used to extend ``add`` calls so the open-ended file set inside a
        tracked directory (e.g. ``memory/journal/*.md``) is staged the same
        way as fixed tracked_files.
        """
        files: list[str] = []
        for rel in self._tracked_dirs:
            base = self._workspace / rel
            if not base.is_dir():
                continue
            for md in sorted(base.glob("*.md")):
                files.append(str(md.relative_to(self._workspace)))
        return files

    def _resolve_sha(self, short_sha: str) -> bytes | None:
        """Resolve a short SHA prefix to the full SHA bytes."""
        try:
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                try:
                    sha = repo.refs[b"HEAD"]
                except KeyError:
                    return None

                while sha:
                    if sha.hex().startswith(short_sha):
                        return sha
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        break
                    sha = commit.parents[0] if commit.parents else None
            return None
        except Exception:
            return None

    def _is_inside_git_repo(self) -> bool:
        """Check if self._workspace is already inside a git repository.

        Walks up from self._workspace to the filesystem root, returning True
        if any parent directory contains a .git entry.

        Git worktrees and submodules can use a ``.git`` file instead of a
        directory, so we must treat either form as "already inside a repo".
        """
        current = self._workspace.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    def _build_gitignore(self) -> str:
        """Generate .gitignore content from tracked files and directories."""
        dirs: set[str] = set()
        for f in self._tracked_files:
            parent = str(Path(f).parent)
            if parent != ".":
                dirs.add(parent)
        # Each tracked_dir contributes the dir itself and every parent so the
        # whitelist chain from root to leaf is uninterrupted by the leading /*.
        for d in self._tracked_dirs:
            p = Path(d)
            for ancestor in [p, *p.parents]:
                a = str(ancestor)
                if a != ".":
                    dirs.add(a)
        lines = ["/*"]
        for d in sorted(dirs):
            lines.append(f"!{d}/")
        for f in self._tracked_files:
            lines.append(f"!{f}")
        for d in self._tracked_dirs:
            lines.append(f"!{d}/*.md")
        # Re-include the per-user wiki vault subtree (Task 5.1). A single
        # recursive ``!memory/users/**`` rule is sufficient and minimal: it
        # un-ignores every descendant of ``memory/users/`` under the leading
        # ``/*`` deny (empirically verified to stage nested dot-paths such as
        # ``wiki/.cold/...`` and ``.lint.log``). It is purely additive — when
        # ``memory/users/`` is absent it matches nothing, so the stock /
        # wiki-disabled workspace's ``.gitignore`` differs only by this one
        # match-nothing line and commits byte-identically. It is deliberately
        # NOT a trailing-slash dir rule, so it does not perturb the existing
        # ``_build_gitignore`` dir-entry contract for root-only tracked sets.
        vault_glob = f"!{'/'.join(self._VAULTS_REL)}/**"
        if vault_glob not in lines:
            lines.append(vault_glob)
        lines.append("!.gitignore")
        return "\n".join(lines) + "\n"

    # -- query -----------------------------------------------------------------

    def log(self, max_entries: int = 20) -> list[CommitInfo]:
        """Return simplified commit log."""
        if not self.is_initialized():
            return []

        try:
            from dulwich.repo import Repo

            entries: list[CommitInfo] = []
            with Repo(str(self._workspace)) as repo:
                try:
                    head = repo.refs[b"HEAD"]
                except KeyError:
                    return []

                sha = head
                while sha and len(entries) < max_entries:
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        break
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(commit.commit_time),
                    )
                    msg = commit.message.decode("utf-8", errors="replace").strip()
                    entries.append(CommitInfo(
                        sha=sha.hex()[:8],
                        message=msg,
                        timestamp=ts,
                    ))
                    sha = commit.parents[0] if commit.parents else None

            return entries
        except Exception:
            logger.exception("Git log failed")
            return []

    def line_ages(self, file_path: str) -> list[LineAge]:
        """Compute the age of each line in a tracked file via git blame.

        Returns one LineAge per line, in order.
        Returns an empty list if the repo is not initialized, the file is
        empty, or annotation fails.
        """

        if not self.is_initialized():
            return []

        target = self._workspace / file_path
        if not target.exists() or target.stat().st_size == 0:
            return []

        try:
            from dulwich import porcelain

            annotated = porcelain.annotate(str(self._workspace), file_path)
        except Exception:
            logger.exception("Git line_ages annotate failed for {}", file_path)
            return []

        if not annotated:
            return []

        return _compute_line_ages(annotated)

    def diff_commits(self, sha1: str, sha2: str) -> str:
        """Show diff between two commits."""
        if not self.is_initialized():
            return ""

        try:
            from dulwich import porcelain

            full1 = self._resolve_sha(sha1)
            full2 = self._resolve_sha(sha2)
            if not full1 or not full2:
                return ""

            out = io.BytesIO()
            porcelain.diff(
                str(self._workspace),
                commit=full1,
                commit2=full2,
                outstream=out,
            )
            return out.getvalue().decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Git diff_commits failed")
            return ""

    def find_commit(self, short_sha: str, max_entries: int = 20) -> CommitInfo | None:
        """Find a commit by short SHA prefix match."""
        for c in self.log(max_entries=max_entries):
            if c.sha.startswith(short_sha):
                return c
        return None

    def show_commit_diff(self, short_sha: str, max_entries: int = 20) -> tuple[CommitInfo, str] | None:
        """Find a commit and return it with its diff vs the parent."""
        commits = self.log(max_entries=max_entries)
        for i, c in enumerate(commits):
            if c.sha.startswith(short_sha):
                if i + 1 < len(commits):
                    diff = self.diff_commits(commits[i + 1].sha, c.sha)
                else:
                    diff = ""
                return c, diff
        return None

    # -- restore ---------------------------------------------------------------

    def revert(self, commit: str) -> str | None:
        """Revert commit ``C``: a **true per-commit inverse**.

        ``revert(C)`` undoes *only* the changes ``C`` itself introduced,
        regardless of where ``C`` sits in history, and leaves every path
        ``C`` did not touch exactly as it currently is on disk. This is the
        exact contract ``/dream-restore`` (Task 5.2) relies on so a user can
        safely revert *any* of the last commits — not just the tip.

        Algorithm:

        * Read both ``C``'s tree and ``C``'s parent tree.
        * The **affected set** = every path whose blob differs between the
          two trees, plus every path present in exactly one of them — i.e.
          precisely the paths ``C`` added, modified, or deleted (a clean
          ``diff(C-parent, C)``). Paths identical in both trees, and paths
          absent from both (e.g. unrelated vault pages created by *later*
          commits, or unchanged legacy files), are **never** in this set.
        * For each affected path: if it exists in ``C``'s **parent** tree,
          rewrite it to that parent-state content (undoing ``C``'s modify,
          or recreating what ``C`` deleted); if it is **absent** from the
          parent tree, delete it on disk and stage the deletion (undoing
          what ``C`` *added* — e.g. a new wiki page). Empty dirs left by a
          deletion are pruned.
        * All other paths — including later, unrelated ``memory/users/**``
          pages and untouched legacy files — are left 100% untouched (not
          even rewritten/restated). This makes ``revert(C)`` the algebraic
          inverse of ``C`` and never causes silent committed data loss of
          files created after ``C``.

        All file rewrites go through :func:`atomic_write_text` so a crash
        mid-revert leaves at most one file in-flight and every other file
        fully old or fully new (recoverable) — the codebase durability bar.

        Commit guarantee (B1): whenever ``revert`` makes ANY filesystem
        change — rewriting/recreating a restored path *or* deleting a
        ``C``-added path — the resulting revert commit is *always* created
        and records exactly those paths. It never leaves a recovered file
        uncommitted/unprotected (which a later Dream/compaction rebuilding
        memory from HEAD would silently destroy). This holds regardless of
        how the underlying dulwich build classifies a recreated
        statically-tracked file in ``porcelain.status``.

        Returns the new revert commit SHA when it changed anything (the
        algebraic inverse, committed and atomic, never touching unrelated
        or later files). Returns None ONLY on failure, or on a *genuine*
        no-op — ``C`` changed nothing under tracking, or its effect is
        already undone — in which case NO empty commit is created. For
        ``/dream-restore`` (Task 5.2): a None return means "nothing to
        undo"; a non-None sha means "restored, HEAD is now <sha>".
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich.repo import Repo

            full_sha = self._resolve_sha(commit)
            if not full_sha:
                logger.warning("Git revert: SHA not found: {}", commit)
                return None

            with Repo(str(self._workspace)) as repo:
                commit_obj = repo[full_sha]
                if commit_obj.type_name != b"commit":
                    return None

                if not commit_obj.parents:
                    logger.warning("Git revert: cannot revert root commit {}", commit)
                    return None

                c_tree = repo[commit_obj.tree]
                parent_obj = repo[commit_obj.parents[0]]
                parent_tree = repo[parent_obj.tree]

                # The affected set is exactly C's own added/modified/deleted
                # paths (diff of C vs its parent) — nothing else. Determined
                # purely from the two trees, so later/unrelated files are
                # provably outside it and are never touched.
                affected = self._diff_tree_paths(repo, parent_tree, c_tree)

                # Every path this revert actually changed on disk
                # (restored/recreated content OR deleted a C-added file).
                # Deletions are included here too — the old separate
                # ``deleted`` list existed only to feed extra_paths; now
                # the whole set is force-staged, which both stages the
                # removals (the dynamic scan can't see gone files) and
                # guarantees recreated legacy files are committed (B1).
                touched: list[str] = []
                for filepath in affected:
                    parent_content = self._read_blob_from_tree(
                        repo, parent_tree, filepath
                    )
                    dest = self._workspace / filepath
                    if parent_content is not None:
                        # Present in C's parent → restore parent content
                        # (undo C's modify, or recreate what C deleted).
                        # Skip the rewrite — and do NOT mark the path
                        # touched — when the on-disk content already equals
                        # the parent state. This keeps revert a true
                        # per-commit inverse that "never restates an
                        # unchanged file" (mtime-stable) AND makes ``touched``
                        # reflect only *real* work, so a genuine no-op (e.g.
                        # reverting the same commit twice) yields an empty
                        # ``touched`` and the B1 force-commit cannot create
                        # an empty/no-op commit.
                        try:
                            already = (
                                dest.is_file()
                                and not dest.is_symlink()
                                and dest.read_text(encoding="utf-8")
                                == parent_content
                            )
                        except (OSError, UnicodeDecodeError):
                            already = False
                        if already:
                            continue
                        atomic_write_text(dest, parent_content)
                        touched.append(filepath)
                    elif dest.exists():
                        # Absent in C's parent but present in C → C added
                        # it. Undo the add.
                        dest.unlink()
                        self._prune_empty_dirs(dest.parent)
                        touched.append(filepath)

            if not touched:
                # Genuine no-op: C changed nothing under tracking, or its
                # changes are already undone (e.g. revert-of-the-same-commit
                # twice). Do NOT force a commit — return None without
                # creating an empty/no-op commit. This is the ONLY path that
                # returns None for an initialized repo with a valid parent.
                return None

            # B1 fix: revert did real work, so the inverse MUST be committed
            # — never silently dropped by auto_commit's status-gated no-op
            # short-circuit. Pass the FULL touched set (restored/recreated
            # ∪ deleted) via extra_paths: this both (a) stages deletions the
            # dynamic scan can't see (files now gone) and (b) force-stages
            # recreated statically-tracked legacy files (e.g.
            # memory/MEMORY.md, SOUL.md) that some dulwich builds report as
            # "clean" after recreation, which would otherwise hit the no-op
            # guard and return None with the recovery uncommitted (B1). With
            # ``touched`` non-empty, ``not extra_paths`` is False so the
            # guard cannot short-circuit and porcelain.add+commit run.
            # I2: pass the already-known effective tracked set through so
            # auto_commit does NOT re-scan the unbounded vault a second time
            # within this single revert (bounded invariant: <=1 scan).
            msg = f"revert: undo {commit}"
            return self.auto_commit(
                msg,
                extra_paths=touched,
                _add_paths=self._effective_tracked_files(),
            )
        except Exception:
            logger.exception("Git revert failed for {}", commit)
            return None

    def _prune_empty_dirs(self, directory: Path) -> None:
        """Remove now-empty dirs up to (not including) the workspace root."""
        ws = self._workspace.resolve()
        current = directory
        while current.resolve() != ws and ws in current.resolve().parents:
            try:
                next(current.iterdir())
                return  # not empty
            except StopIteration:
                parent = current.parent
                try:
                    current.rmdir()
                except OSError:
                    return
                current = parent
            except FileNotFoundError:
                return

    @staticmethod
    def _iter_tree_paths(repo, tree, prefix: str = ""):
        """Yield every blob path (POSIX, repo-relative) under a tree object."""
        for name, _mode, sha in tree.items():
            n = name.decode()
            obj = repo[sha]
            if obj.type_name == b"tree":
                yield from GitStore._iter_tree_paths(repo, obj, prefix + n + "/")
            elif obj.type_name == b"blob":
                yield prefix + n

    @staticmethod
    def _blob_index(repo, tree, prefix: str = "") -> dict[str, bytes]:
        """Map every blob path under ``tree`` to its blob SHA (bytes).

        Deterministic full walk; used to diff two trees by content.
        """
        out: dict[str, bytes] = {}
        for name, _mode, sha in tree.items():
            n = name.decode()
            obj = repo[sha]
            if obj.type_name == b"tree":
                out.update(GitStore._blob_index(repo, obj, prefix + n + "/"))
            elif obj.type_name == b"blob":
                out[prefix + n] = sha
        return out

    @staticmethod
    def _diff_tree_paths(repo, tree_a, tree_b) -> list[str]:
        """Paths that differ between ``tree_a`` (parent) and ``tree_b`` (C).

        Returns the sorted set of paths that are added, removed, or have a
        different blob SHA between the two trees — i.e. exactly the paths a
        single commit changed relative to its parent. A path with an
        identical blob SHA in both trees is **not** returned (so revert
        never restates an unchanged file), and a path absent from both is
        of course absent here (so unrelated/later files are never touched).
        Deterministic (sorted) ordering.
        """
        idx_a = GitStore._blob_index(repo, tree_a)
        idx_b = GitStore._blob_index(repo, tree_b)
        changed = {
            p
            for p in set(idx_a) | set(idx_b)
            if idx_a.get(p) != idx_b.get(p)
        }
        return sorted(changed)

    @staticmethod
    def _read_blob_from_tree(repo, tree, filepath: str) -> str | None:
        """Read a blob's content from a tree object by walking path parts."""
        parts = Path(filepath).parts
        current = tree
        for part in parts:
            try:
                entry = current[part.encode()]
            except KeyError:
                return None
            obj = repo[entry[1]]
            if obj.type_name == b"blob":
                return obj.data.decode("utf-8", errors="replace")
            if obj.type_name == b"tree":
                current = obj
            else:
                return None
        return None
