"""Deterministic, idempotent Lint engine for a single wiki vault.

Lint is the Dream-side curator (design §3/§5; Karpathy "Lint"). It runs each
Dream interval (wired under the per-vault lock in Task 4.5) and performs
deterministic filesystem maintenance on **one** vault: no LLM, no lock taken
here, no network. It implements the four Karpathy checks plus dedup and decay,
then regenerates the per-type ``_index.md`` files and the root ``MEMORY.md``
Map-of-Content from frontmatter.

Two properties are non-negotiable and load-bearing for the rest of the system
(the MOC is injected into the prompt in Task 6.1, the vault is versioned in
git):

* **Idempotence** -- running :func:`run_lint` twice in a row makes ZERO
  filesystem changes on the second run. Every page, ``_index.md``,
  ``MEMORY.md`` and ``.lint.log`` byte is identical after run #2 vs run #1.
  Achieved by (i) only writing a file when its new bytes differ from the
  current bytes, and (ii) only appending to ``.lint.log`` when a real action
  occurred this run (a zero-action run touches nothing anywhere).
* **Determinism** -- every collection is sorted; ties are broken by the POSIX
  relpath ascending. Two independently-built identical vaults lint to
  byte-identical trees.

A third, equally load-bearing invariant governs every page MOVE (C1 review):

* **No-clobber move invariant** -- no ``run_lint`` execution may
  :func:`atomic_write_text` to a destination relpath currently occupied by a
  *different surviving* (parseable, non-malformed) entry. A single
  authoritative "occupied" map (relpath -> owning entry) is threaded through
  every move phase and kept in lock-step with on-disk reality (the source
  relpath is removed and the destination relpath added as each move
  completes). When a computed destination is already held by another live
  entry the move is **deferred** -- the page is left where it is *this run*
  and (because the colliding pages necessarily share ``(type,
  slug.lower())``) :func:`_dedup` then MERGES them: keeper = max ``updated``,
  loser body appended under ``## merged from {relpath}``, loser deleted. No
  page is ever overwritten or silently lost; worst case two colliding pages
  are merged, best case one is relocated without collision. A post-dedup
  stale->cold sweep then settles any merged-but-stale keeper so the whole
  run reaches a fixpoint (run #2 == run #1, zero changes).

Crash-safety ordering contract (M3) -- every move is **write-new (atomic)
THEN unlink-old; never reorder**. A crash mid-move must leave a recoverable
duplicate, never a lost page. Task 4.4/4.5 integrators MUST NOT reorder this.

Phase order (each a small private function for testability):

1. ``_scan`` -- walk ``wiki_dir`` ourselves so we see ``.cold/`` and
   malformed files; malformed pages are recorded and EXCLUDED from every
   later phase (Lint owns corruption detection -- 4.2 review).
2. ``_reheat_relocate`` -- a ``status == "hot"`` page still physically under
   a ``.cold`` component is moved out to its hot location (the documented
   hand-off from ``wiki_note`` reheat-on-read, design §4). Runs BEFORE
   stale->cold so a just-reheated page is not re-cooled.
3. ``_stale_to_cold`` -- a hot-located page for which
   :func:`should_cool` is True is moved into ``.cold/``.
4. ``_dedup`` -- hot pages sharing ``(type, slug.lower())`` are merged into
   the keeper (max ``updated``, relpath tiebreak); losers deleted.
5. ``_broken_links`` -- ``links_out`` refs with no hot OR cold target are
   recorded (never auto-fixed; design §5).
6. ``_regenerate_indexes`` -- each type ``_index.md`` rebuilt from
   frontmatter.
7. ``_regenerate_moc`` -- the root ``MEMORY.md`` MOC rebuilt, capped at
   ``schema.moc_max_lines``.
8. ``_append_log`` -- one audit block appended to ``.lint.log`` iff any
   action occurred this run.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.wiki.decay import should_cool
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import _COLD_COMPONENT, _NON_PAGE_NAMES, Vault
from nanobot.utils.atomic import atomic_write_text

__all__ = ["LintReport", "run_lint"]


@dataclass
class LintReport:
    """Structured, deterministically-ordered outcome of one Lint run.

    ``changed`` is a convenience: True iff any action occurred this run (and
    therefore iff a ``.lint.log`` block was appended). A run with
    ``changed is False`` made ZERO filesystem changes.
    """

    cooled: list[str] = field(default_factory=list)
    reheated: list[str] = field(default_factory=list)
    merged: list[tuple[str, str]] = field(default_factory=list)
    broken_links: list[tuple[str, str]] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    orphans_fixed: list[str] = field(default_factory=list)
    indexes_regenerated: list[str] = field(default_factory=list)
    moc_regenerated: bool = False

    @property
    def changed(self) -> bool:
        """Whether a real filesystem MUTATION occurred this run.

        ``malformed`` and ``broken_links`` are deliberately EXCLUDED: they
        are steady-state *findings* (Lint never fixes them -- malformed
        files are left untouched, broken links are recorded not auto-fixed),
        so a malformed/broken-link file that persists across runs would
        otherwise re-trigger a ``.lint.log`` block forever and break the
        non-negotiable idempotence guarantee. They are still audited, but
        only inside a log block that a genuine mutation already produced
        (see :func:`_append_log`). A run with ``changed is False`` made ZERO
        filesystem changes.
        """
        return bool(
            self.cooled
            or self.reheated
            or self.merged
            or self.orphans_fixed
            or self.indexes_regenerated
            or self.moc_regenerated
        )


@dataclass
class _Entry:
    """A parseable page discovered by the scan phase."""

    relpath: str  # POSIX, relative to wiki_dir
    page: Page
    path: Path  # absolute current location
    in_cold: bool  # _COLD_COMPONENT in path.parts


def _slug_of(relpath: str) -> str:
    """Page slug = basename without ``.md`` (POSIX relpath in)."""
    return relpath.rsplit("/", 1)[-1][: -len(".md")]


# Max rendered length of a sanitized inline string (title in the MOC).
_SAFE_INLINE_MAX = 120


def _safe_inline(text: str) -> str:
    """Sanitize untrusted free text for SINGLE-LINE rendering into the MOC.

    The MOC (root ``MEMORY.md``) is injected verbatim into the system prompt
    (Task 6.1) and Ingest (Task 4.4) feeds Lint untrusted, model-/user-shaped
    titles. Rendering a raw ``page.title`` into a
    ``- [[{folder}/{slug}]] — {title}`` line is a prompt-injection vector: a
    newline forges extra MOC lines, and ``]]`` / ``[[`` forge spurious
    wikilinks. This helper makes any free text safe to interpolate inline:

    * collapse every run of ASCII/Unicode whitespace (incl. newlines, tabs,
      CR) to a single space, so the result is exactly one physical line;
    * neutralize wikilink delimiters by inserting a space inside them
      (``]]`` -> ``] ]``, ``[[`` -> ``[ [``) so no forged ``[[..]]`` can
      survive while the visible text is preserved;
    * trim leading/trailing space and cap length at ``_SAFE_INLINE_MAX``
      (deterministic hard bound; over-long titles are an abuse signal).

    Deterministic and idempotent: ``_safe_inline(_safe_inline(x)) ==
    _safe_inline(x)`` for all inputs (the substitutions never reintroduce a
    delimiter or whitespace run).
    """
    collapsed = " ".join(text.split())
    neutralized = collapsed.replace("]]", "] ]").replace("[[", "[ [")
    return neutralized[:_SAFE_INLINE_MAX].strip()


def _write_if_changed(path: Path, content: str) -> bool:
    """Atomically write ``content`` to ``path`` iff the *text* would differ.

    Returns True iff a write actually happened (the idempotence primitive:
    a no-op rewrite is never performed and never reported as an action).

    The comparison is on **decoded text**, not raw bytes, on purpose:
    :func:`atomic_write_text` opens the file in text mode, so on Windows it
    performs newline translation (``\\n`` -> ``\\r\\n``). A naive
    ``content.encode() == path.read_bytes()`` check would therefore *never*
    match on Windows and Lint would rewrite every file on every run --
    fatal for the idempotence guarantee. ``Path.read_text`` decodes with
    universal newlines (``\\r\\n`` -> ``\\n``), so comparing the decoded
    text to ``content`` is correct and platform-independent: if the text is
    unchanged we skip the write entirely and the raw on-disk bytes (with
    whatever native line endings) stay byte-stable across runs.
    """
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            pass
    atomic_write_text(path, content)
    return True


# --------------------------------------------------------------------------- #
# Phase 1 -- scan
# --------------------------------------------------------------------------- #
def _scan(vault: Vault, report: LintReport) -> list[_Entry]:
    """Walk ``wiki_dir`` and split files into parseable entries vs malformed.

    Malformed files are recorded in ``report.malformed`` (sorted) and
    EXCLUDED from every later phase -- they are never moved, merged, deleted
    or indexed (left untouched for a human / Dream; 4.2 review).
    """
    entries: list[_Entry] = []
    malformed: list[str] = []
    if not vault.wiki_dir.is_dir():
        return entries
    for path in sorted(vault.wiki_dir.rglob("*.md")):
        if path.name in _NON_PAGE_NAMES:
            continue
        rel = path.relative_to(vault.wiki_dir).as_posix()
        try:
            page = parse_page(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            malformed.append(rel)
            continue
        entries.append(
            _Entry(
                relpath=rel,
                page=page,
                path=path,
                in_cold=_COLD_COMPONENT in path.parts,
            )
        )
    report.malformed = sorted(malformed)
    entries.sort(key=lambda e: e.relpath)
    return entries


def _hot_relpath(vault: Vault, page: Page, slug: str) -> str:
    """POSIX relpath (relative to wiki_dir) a hot page of this type lives at."""
    return f"{vault.schema.folder(page.type)}/{slug}.md"


def _cold_relpath(vault: Vault, page: Page, slug: str) -> str:
    """POSIX relpath a cold page of this type lives at."""
    return f"{_COLD_COMPONENT}/{vault.schema.folder(page.type)}/{slug}.md"


class _Occupied:
    """Authoritative map of every live entry's current relpath -> entry.

    The single source of truth for the **no-clobber move invariant** (C1):
    every move phase consults it before computing/committing a destination
    and mutates it as moves happen so it always mirrors on-disk reality for
    surviving (parseable, non-malformed) entries. A destination relpath is
    "free for ``entry``" iff it is unoccupied OR occupied by ``entry``
    itself; if a *different* live entry holds it the move must be deferred,
    never performed (no ``atomic_write_text`` may ever overwrite it).
    """

    def __init__(self, entries: list[_Entry]) -> None:
        self._by_rel: dict[str, _Entry] = {e.relpath: e for e in entries}

    def holder(self, relpath: str) -> _Entry | None:
        return self._by_rel.get(relpath)

    def free_for(self, relpath: str, entry: _Entry) -> bool:
        """True iff ``entry`` may write to ``relpath`` without clobbering."""
        held = self._by_rel.get(relpath)
        return held is None or held is entry

    def move(self, src_rel: str, dst_rel: str, entry: _Entry) -> None:
        """Record that ``entry`` moved ``src_rel`` -> ``dst_rel`` on disk."""
        if self._by_rel.get(src_rel) is entry:
            del self._by_rel[src_rel]
        self._by_rel[dst_rel] = entry

    def drop(self, relpath: str, entry: _Entry) -> None:
        """Record that ``entry``'s file at ``relpath`` was deleted."""
        if self._by_rel.get(relpath) is entry:
            del self._by_rel[relpath]


# --------------------------------------------------------------------------- #
# Phase 2 -- reheat-relocate
# --------------------------------------------------------------------------- #
def _reheat_relocate(
    vault: Vault, entries: list[_Entry], report: LintReport, occ: _Occupied
) -> list[_Entry]:
    """Move every ``status == "hot"`` page still under ``.cold`` out to hot.

    Relocation key exactly: ``status == "hot" and _COLD_COMPONENT in
    path.parts`` (design §4, pinned in Task 2.3). If the hot-location target
    is already held by another live entry (``occ`` -- the authoritative
    no-clobber map), do NOT lose data: leave the cold copy in place (still
    ``status: hot``) and let :func:`_dedup` resolve the collision -- both
    survive into the same ``(type, slug.lower())`` dedup group and are
    merged, never overwritten. Page content is written unchanged.

    Crash-safety ordering contract (M3): write-new (atomic) THEN unlink-old;
    NEVER reorder -- a crash mid-move must leave a recoverable duplicate,
    never a lost page. Task 4.4/4.5 integrators must not reorder this.
    """
    out: list[_Entry] = []
    # Deterministic processing order.
    for entry in sorted(entries, key=lambda e: e.relpath):
        if not (entry.in_cold and entry.page.status == "hot"):
            out.append(entry)
            continue
        try:
            slug = _slug_of(entry.relpath)
            target_rel = _hot_relpath(vault, entry.page, slug)
        except KeyError:
            # Unknown type: cannot compute a hot home; leave it where it is.
            out.append(entry)
            continue
        if not occ.free_for(target_rel, entry):
            # Collision: keep both, defer to dedup. The cold copy survives in
            # place (still status hot) so dedup groups it with the hot one
            # and MERGES them -- no surviving page is ever overwritten.
            out.append(entry)
            continue
        target = vault.wiki_dir / target_rel
        # Content is unchanged (it was reheated in place by wiki_note).
        # write-new THEN unlink-old (crash-safe ordering -- never reorder).
        atomic_write_text(target, serialize_page(entry.page))
        entry.path.unlink(missing_ok=True)
        occ.move(entry.relpath, target_rel, entry)
        report.reheated.append(f"{entry.relpath} -> {target_rel}")
        out.append(
            _Entry(
                relpath=target_rel,
                page=entry.page,
                path=target,
                in_cold=False,
            )
        )
    report.reheated.sort()
    out.sort(key=lambda e: e.relpath)
    return out


# Phase 3 (stale -> cold) is implemented as _stale_to_cold_impl below the
# orchestrator (it needs ``today``); kept there to keep the phase signature
# uniform without threading ``today`` through a module global.


# --------------------------------------------------------------------------- #
# Phase 4 -- dedup / merge
# --------------------------------------------------------------------------- #
def _dedup(
    vault: Vault, entries: list[_Entry], report: LintReport, occ: _Occupied
) -> list[_Entry]:
    """Merge logically-hot pages sharing ``(type, slug.lower())``.

    A page is *logically hot* iff ``page.status == "hot"`` -- this includes a
    deferred reheat collision (a ``status: hot`` page still physically under
    ``.cold/`` because :func:`_reheat_relocate` could not relocate it without
    clobbering an existing hot page) AND a deferred stale->cold collision (a
    stale page :func:`_stale_to_cold_impl` left ``status: hot`` because its
    ``.cold/`` destination was held by another live entry). Such collisions
    MUST be resolved here (design §4 / pinned contract: move collisions are
    deferred to dedup). Genuinely cold pages (``status == "cold"``) are never
    touched.

    keeper = max by ``page.updated`` (ISO string compare); tie-break = POSIX
    relpath ascending (the relpath that sorts first wins). Losers processed
    in deterministic order (relpath ascending). For each loser the keeper's
    body gains ``\\n\\n## merged from {loser_relpath}\\n\\n{loser.body}``
    (skipped if that exact header is already present -- idempotence). The
    keeper's ``updated``/``last_touched`` become the max (ISO string) of
    keeper and loser; ``created`` is preserved. Every loser file is deleted.

    A surviving keeper belongs at its **hot** location: if it was a deferred
    collision (physically under ``.cold/``) it is written to
    ``wiki/{folder}/{slug}.md`` and the cold file removed -- BUT only if that
    hot relpath is free in ``occ`` (the no-clobber move invariant, C1). If a
    *different* surviving entry still holds the hot relpath the relocate is
    deferred (keeper stays under ``.cold/`` this run, still ``status: hot``)
    so a subsequent run / the next dedup pass reconciles it; no surviving
    page is ever overwritten. Single-member groups are returned untouched.

    Crash-safety ordering contract (M3): write-new (atomic) THEN unlink-old
    on the keeper-relocate; NEVER reorder.
    """
    logically_hot = [e for e in entries if e.page.status == "hot"]
    genuinely_cold = [e for e in entries if e.page.status != "hot"]

    groups: dict[tuple[str, str], list[_Entry]] = {}
    for e in logically_hot:
        key = (e.page.type, _slug_of(e.relpath).lower())
        groups.setdefault(key, []).append(e)

    survivors: list[_Entry] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) == 1:
            survivors.append(members[0])
            continue
        # keeper: max ``updated`` (ISO string compare); tie-break = POSIX
        # relpath ascending (the relpath that sorts first wins as keeper).
        max_updated = max(m.page.updated for m in members)
        tied = [m for m in members if m.page.updated == max_updated]
        keeper = sorted(tied, key=lambda e: e.relpath)[0]
        losers = sorted(
            (m for m in members if m is not keeper), key=lambda e: e.relpath
        )

        body = keeper.page.body
        updated = keeper.page.updated
        last_touched = keeper.page.last_touched
        for loser in losers:
            header = f"## merged from {loser.relpath}"
            if header not in body:
                # Single clean separation: exactly one blank line between
                # the keeper body and the merged section. rstrip the keeper
                # body so repeated/varied trailing newlines never produce a
                # different result (idempotence is also guarded by the
                # header-presence check above).
                prefix = body.rstrip("\n")
                lead = f"{prefix}\n\n" if prefix else ""
                body = f"{lead}{header}\n\n{loser.page.body}"
            updated = max(updated, loser.page.updated)
            last_touched = max(last_touched, loser.page.last_touched)
            report.merged.append((keeper.relpath, loser.relpath))
            loser.path.unlink(missing_ok=True)
            # The loser's relpath is now free on disk -- reflect that in the
            # authoritative map so a keeper-relocate onto it is not a clobber.
            occ.drop(loser.relpath, loser)

        keeper.page.body = body
        keeper.page.updated = updated
        keeper.page.last_touched = last_touched

        # The merged keeper belongs at its hot location. If it was a deferred
        # collision (still under .cold/), relocate it out -- but NEVER onto a
        # relpath a different surviving entry still holds (no-clobber, C1).
        slug = _slug_of(keeper.relpath)
        try:
            hot_rel = _hot_relpath(vault, keeper.page, slug)
        except KeyError:  # unknown type: keep it where it is
            hot_rel = keeper.relpath
        hot_path = vault.wiki_dir / hot_rel
        if (
            keeper.in_cold
            and hot_rel != keeper.relpath
            and occ.free_for(hot_rel, keeper)
        ):
            # write-new THEN unlink-old (crash-safe ordering -- never reorder).
            atomic_write_text(hot_path, serialize_page(keeper.page))
            keeper.path.unlink(missing_ok=True)
            occ.move(keeper.relpath, hot_rel, keeper)
            report.reheated.append(f"{keeper.relpath} -> {hot_rel}")
            report.reheated.sort()
            survivors.append(
                _Entry(
                    relpath=hot_rel,
                    page=keeper.page,
                    path=hot_path,
                    in_cold=False,
                )
            )
        else:
            # Either already at its hot home, or the hot relpath is still
            # held by a different surviving entry -> defer the relocate
            # (keeper stays put this run, content still rewritten in place).
            _write_if_changed(keeper.path, serialize_page(keeper.page))
            survivors.append(keeper)

    report.merged.sort()
    survivors.extend(genuinely_cold)
    survivors.sort(key=lambda e: e.relpath)
    return survivors


# --------------------------------------------------------------------------- #
# Phase 5 -- broken links
# --------------------------------------------------------------------------- #
def _broken_links(
    vault: Vault, entries: list[_Entry], report: LintReport
) -> None:
    """Record ``links_out`` refs with no hot OR cold target page.

    A ref like ``"projects/payment-svc"`` is satisfied if
    ``wiki/{ref}.md`` OR ``wiki/.cold/{ref}.md`` exists (a link to a
    now-cold page is NOT broken). Nothing is modified or deleted -- broken
    links are recorded only (design §5).

    Path-containment guard (I2): a ``links_out`` ref like
    ``"../../../etc/hosts"`` resolves OUTSIDE the vault. Without a guard,
    ``(wiki_dir / f"{ref}.md").exists()`` could return True for an unrelated
    out-of-vault file and the escaping ref would be falsely treated as "not
    broken" (and is inconsistent with :meth:`Vault.read_page`, which already
    rejects out-of-vault refs via the same ``is_under`` helper imported from
    ``nanobot.agent.tools.path_utils`` -- the exact import ``vault.py``
    uses). A candidate is only accepted as satisfying the link if it BOTH
    exists AND resolves under ``wiki_dir``; a ref that escapes the vault is
    therefore recorded as BROKEN.
    """
    hot_entries = [e for e in entries if not e.in_cold]
    wiki_root = vault.wiki_dir.resolve()

    def _contained_and_exists(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return is_under(resolved, wiki_root) and resolved.exists()

    broken: list[tuple[str, str]] = []
    for entry in sorted(hot_entries, key=lambda e: e.relpath):
        for ref in entry.page.links_out:
            hot_target = vault.wiki_dir / f"{ref}.md"
            cold_target = vault.wiki_dir / _COLD_COMPONENT / f"{ref}.md"
            if _contained_and_exists(hot_target) or _contained_and_exists(
                cold_target
            ):
                continue
            broken.append((entry.relpath, ref))
    report.broken_links = sorted(broken)


# --------------------------------------------------------------------------- #
# Phase 6 -- regenerate per-type _index.md
# --------------------------------------------------------------------------- #
def _index_content(folder: str, slugs: list[str]) -> str:
    body = f"# {folder} index\n\n"
    for slug in slugs:
        body += f"- [[{folder}/{slug}]]\n"
    return body


def _existing_index_slugs(text: str) -> set[str]:
    """Slugs already present as ``- [[folder/slug]]`` stubs in an index."""
    slugs: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- [[") and line.endswith("]]"):
            ref = line[len("- [[") : -len("]]")]
            slugs.add(ref.rsplit("/", 1)[-1])
    return slugs


def _regenerate_indexes(
    vault: Vault, entries: list[_Entry], report: LintReport
) -> None:
    """Rebuild every type ``_index.md`` deterministically from frontmatter.

    Rule (documented & locked by tests):

    * A type folder with >=1 surviving HOT page: ``_index.md`` =
      ``f"# {folder} index\\n\\n"`` followed by one
      ``- [[{folder}/{slug}]]`` line per hot page sorted by slug ascending.
    * A type folder with 0 hot pages but an EXISTING ``_index.md``:
      regenerated to just ``f"# {folder} index\\n\\n"`` (kept consistent /
      idempotent rather than left stale).
    * A type folder with 0 hot pages and NO ``_index.md``: nothing is done
      (we never conjure an empty index out of thin air).

    A hot page absent from its pre-existing ``_index.md`` is an orphan; it
    is counted in ``report.orphans_fixed`` and now appears.
    """
    hot = [e for e in entries if not e.in_cold]

    # folder -> sorted slug list of its hot pages.
    by_folder: dict[str, list[str]] = {}
    for e in hot:
        folder = e.relpath.split("/", 1)[0]
        by_folder.setdefault(folder, []).append(_slug_of(e.relpath))

    # Every folder that either has hot pages or has an existing _index.md.
    candidate_folders: set[str] = set(by_folder)
    if vault.wiki_dir.is_dir():
        for child in sorted(vault.wiki_dir.iterdir()):
            if child.is_dir() and child.name != _COLD_COMPONENT:
                if (child / "_index.md").exists():
                    candidate_folders.add(child.name)

    regenerated: list[str] = []
    orphans: list[str] = []
    for folder in sorted(candidate_folders):
        slugs = sorted(by_folder.get(folder, []))
        idx_path = vault.wiki_dir / folder / "_index.md"
        if not slugs and not idx_path.exists():
            continue
        old_slugs: set[str] = set()
        if idx_path.exists():
            old_slugs = _existing_index_slugs(
                idx_path.read_text(encoding="utf-8")
            )
        content = _index_content(folder, slugs)
        if _write_if_changed(idx_path, content):
            regenerated.append(f"{folder}/_index.md")
        for slug in slugs:
            if slug not in old_slugs:
                orphans.append(f"{folder}/{slug}")

    report.indexes_regenerated = sorted(regenerated)
    report.orphans_fixed = sorted(orphans)


# --------------------------------------------------------------------------- #
# Phase 7 -- regenerate root MEMORY.md (the MOC)
# --------------------------------------------------------------------------- #
def _moc_content(vault: Vault, entries: list[_Entry]) -> str:
    """Build the deterministic MOC, truncating ``## Recent`` to the cap.

    Layout: ``# Memory`` + blank line; ``## Map`` (one
    ``- [[{folder}/_index]]`` per type with >=1 hot page, folder ascending);
    ``## Recent`` (hot pages by ``last_touched`` DESC then relpath ASC, each
    ``- [[{folder}/{slug}]] — {title}``).

    The page ``title`` is untrusted free text (Ingest, Task 4.4, feeds Lint
    model-/user-shaped titles and this MOC is injected verbatim into the
    system prompt in Task 6.1). It is passed through :func:`_safe_inline`
    so a title containing a newline cannot forge an extra MOC line and one
    containing ``]]`` / ``[[`` cannot forge a spurious wikilink (I1).

    The total file must not exceed ``schema.moc_max_lines`` lines: oldest
    ``## Recent`` entries are dropped until it fits. ``moc_max_lines`` is a
    SOFT target -- if ``# Memory`` + ``## Map`` alone already exceed it, Map
    is still emitted in full (correctness over the soft cap). The
    ``## Recent`` header is omitted ENTIRELY when it would have zero entries
    -- whether because there are no hot pages or truncation dropped them all
    -- so an empty section is never emitted (M2). Deterministic regardless.
    """
    hot = [e for e in entries if not e.in_cold]
    folders = sorted(
        {e.relpath.split("/", 1)[0] for e in hot}
    )
    head = "# Memory\n\n"
    map_lines = ["## Map\n"] + [f"- [[{f}/_index]]\n" for f in folders]

    # Recent: last_touched DESC, then relpath ASC.
    recent_sorted = sorted(
        hot, key=lambda e: (_desc_key(e.page.last_touched), e.relpath)
    )
    recent_lines = []
    for e in recent_sorted:
        folder = e.relpath.split("/", 1)[0]
        slug = _slug_of(e.relpath)
        # _safe_inline neutralizes newline / wikilink prompt-injection (I1).
        title = _safe_inline(e.page.title)
        recent_lines.append(f"- [[{folder}/{slug}]] — {title}\n")

    cap = vault.schema.moc_max_lines

    def assemble(n_recent: int) -> str:
        # M2: emit the Recent header only when it has >=1 entry.
        recent = (
            ["## Recent\n", *recent_lines[:n_recent]] if n_recent > 0 else []
        )
        return "".join([head, *map_lines, *recent])

    n = len(recent_lines)
    content = assemble(n)
    # splitlines() count == number of trailing-newline-terminated lines here.
    while n > 0 and len(content.splitlines()) > cap:
        n -= 1
        content = assemble(n)
    return content


class _DescStr(str):
    """A string whose ordering is REVERSED so it sorts DESCENDING.

    Used purely as a composite sort key (newest ``last_touched`` first).
    The ordering is now TOTAL and self-consistent (M1): all four ordering
    dunders are reversed in lock-step so ``sorted``/``min``/``max`` and any
    ``>``/``>=`` comparison all agree. ``__eq__``/``__hash__`` are
    deliberately left as plain ``str`` semantics: equal strings are equal
    (a stable sort then falls back to the secondary relpath key), and
    reversing only the strict/loose orderings keeps the relation total
    (exactly one of ``<``, ``==``, ``>`` holds for any two values).

    ``functools.total_ordering`` cannot help here: it only fills in dunders
    a class is *missing*, but ``str`` already defines all of them, so each
    reversed operator must be written explicitly.
    """

    def __lt__(self, other):  # type: ignore[override]
        return str.__gt__(self, other)

    def __le__(self, other):  # type: ignore[override]
        return str.__ge__(self, other)

    def __gt__(self, other):  # type: ignore[override]
        return str.__lt__(self, other)

    def __ge__(self, other):  # type: ignore[override]
        return str.__le__(self, other)


def _desc_key(value: str) -> _DescStr:
    return _DescStr(value)


def _regenerate_moc(
    vault: Vault, entries: list[_Entry], report: LintReport
) -> None:
    content = _moc_content(vault, entries)
    if _write_if_changed(vault.root / "MEMORY.md", content):
        report.moc_regenerated = True


# --------------------------------------------------------------------------- #
# Phase 8 -- .lint.log
# --------------------------------------------------------------------------- #
def _log_block(report: LintReport) -> str:
    """Render the ordered ``- `` action lines for this run (no header)."""
    lines: list[str] = []
    for rel in report.malformed:
        lines.append(f"- malformed {rel}")
    for rel in report.reheated:
        lines.append(f"- reheated {rel}")
    for rel in report.cooled:
        lines.append(f"- cooled {rel}")
    for keeper, loser in report.merged:
        lines.append(f"- merged {loser} into {keeper}")
    for src, ref in report.broken_links:
        lines.append(f"- broken-link {src} -> {ref}")
    for rel in report.indexes_regenerated:
        lines.append(f"- regenerated {rel}")
    if report.moc_regenerated:
        lines.append("- regenerated MEMORY.md")
    return "\n".join(lines)


def _append_log(vault: Vault, today: dt.date, report: LintReport) -> None:
    """Append one audit block to ``.lint.log`` iff a MUTATION occurred.

    A run with no mutation writes NOTHING -- this is what preserves
    ``.lint.log`` byte-identity across a no-op second run, even when a
    malformed file or broken link persists (Lint never fixes those, so they
    re-appear every run; auditing them unconditionally would break
    idempotence). When a real mutation does occur, the persistent
    malformed/broken-link findings ARE included in that block for the human
    audit trail.
    """
    if not report.changed:
        return
    block = f"## lint {today.isoformat()}\n{_log_block(report)}\n"
    log_path = vault.root / ".lint.log"
    prefix = ""
    if log_path.exists():
        prefix = log_path.read_text(encoding="utf-8")
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
    atomic_write_text(log_path, prefix + block)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_lint(vault: Vault, today: dt.date) -> LintReport:
    """Run all Lint phases on ``vault`` as of ``today`` (no LLM, no lock).

    Deterministic and idempotent: a second consecutive call makes ZERO
    filesystem changes and returns a report whose ``changed`` is False. The
    caller (Task 4.5) invokes this under the per-vault lock, after Ingest.
    """
    report = LintReport()

    entries = _scan(vault, report)
    # One authoritative no-clobber map, seeded from the post-scan on-disk
    # truth and kept in lock-step through every move phase (C1 invariant:
    # no phase may atomic_write_text onto a relpath a different surviving
    # entry holds; colliding pages are merged or deferred, never overwritten).
    occ = _Occupied(entries)
    entries = _reheat_relocate(vault, entries, report, occ)
    entries = _stale_to_cold_impl(vault, entries, report, today, occ)
    entries = _dedup(vault, entries, report, occ)
    # Post-dedup stale sweep: a stale page whose .cold/ destination was
    # held by another live entry was DEFERRED above (left status: hot) so
    # dedup could MERGE the collision instead of clobbering it. The merged
    # keeper may now itself be stale with a (now-freed) destination -- cool
    # it here so the whole run reaches a fixpoint (run #2 == run #1). The
    # sweep is itself idempotent: on a settled tree nothing is stale-and-hot.
    entries = _stale_to_cold_impl(vault, entries, report, today, occ)
    _broken_links(vault, entries, report)
    _regenerate_indexes(vault, entries, report)
    _regenerate_moc(vault, entries, report)
    _append_log(vault, today, report)
    return report


def _stale_to_cold_impl(
    vault: Vault,
    entries: list[_Entry],
    report: LintReport,
    today: dt.date,
    occ: _Occupied,
) -> list[_Entry]:
    """Move hot-located stale pages into ``.cold/`` (status flipped cold).

    Considers only hot-located pages (``not in_cold``); runs after reheat so
    a freshly reheated page (last_touched == today) is never re-cooled.

    No-clobber move invariant (C1): the ``.cold/`` destination relpath is
    NEVER written if a *different* surviving entry already holds it (``occ``
    -- the authoritative map). The classic data-loss case is a stale hot
    page X at ``people/sam.md`` whose cold home ``.cold/people/sam.md`` is
    already physically occupied by a deferred-reheat page Y (``status: hot``,
    same ``(type, slug.lower())``). Naively cooling X here would
    ``atomic_write_text`` over Y's file and then unlink X's old path --
    silently destroying Y. Instead the cool is **DEFERRED**: X is left
    exactly where it is *with ``status`` unchanged (still ``hot``)* so X and
    Y remain in the same logical-hot dedup group and :func:`_dedup` MERGES
    them (keeper = max ``updated``; loser body appended under ``## merged
    from {relpath}``; loser deleted) -- no page is ever overwritten or lost.
    The post-dedup invocation of this function then cools the merged keeper
    (its destination is now free), so the whole run reaches a fixpoint and
    run #2 makes zero changes (idempotent + deterministic).

    Crash-safety ordering contract (M3): write-new (atomic) THEN unlink-old;
    NEVER reorder -- a crash mid-cool must leave a recoverable duplicate.

    ``report.cooled`` is ACCUMULATED (this function is invoked twice per
    run -- pre- and post-dedup); it is re-sorted in place each call so the
    report stays deterministically ordered regardless of invocation count.
    """
    out: list[_Entry] = []
    cooled: list[str] = list(report.cooled)
    for entry in sorted(entries, key=lambda e: e.relpath):
        if entry.in_cold or not should_cool(entry.page, vault.schema, today):
            out.append(entry)
            continue
        try:
            slug = _slug_of(entry.relpath)
            target_rel = _cold_relpath(vault, entry.page, slug)
        except KeyError:  # pragma: no cover - should_cool gates unknown types
            out.append(entry)
            continue
        if not occ.free_for(target_rel, entry):
            # Collision: the cold destination is held by a different
            # surviving entry. DEFER -- leave the page in place, status
            # UNCHANGED (still hot) so dedup merges the collision rather
            # than this phase clobbering a surviving page. No mutation.
            out.append(entry)
            continue
        entry.page.status = "cold"
        target = vault.wiki_dir / target_rel
        # write-new THEN unlink-old (crash-safe ordering -- never reorder).
        atomic_write_text(target, serialize_page(entry.page))
        entry.path.unlink(missing_ok=True)
        occ.move(entry.relpath, target_rel, entry)
        cooled.append(entry.relpath)
        out.append(
            _Entry(
                relpath=target_rel,
                page=entry.page,
                path=target,
                in_cold=True,
            )
        )
    report.cooled = sorted(set(cooled))
    out.sort(key=lambda e: e.relpath)
    return out
