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


# --------------------------------------------------------------------------- #
# Phase 2 -- reheat-relocate
# --------------------------------------------------------------------------- #
def _reheat_relocate(
    vault: Vault, entries: list[_Entry], report: LintReport
) -> list[_Entry]:
    """Move every ``status == "hot"`` page still under ``.cold`` out to hot.

    Relocation key exactly: ``status == "hot" and _COLD_COMPONENT in
    path.parts`` (design §4, pinned in Task 2.3). If the hot-location target
    already exists as another parseable entry, do NOT lose data: leave the
    cold copy in place and let :func:`_dedup` resolve the collision (both
    survive into the dedup grouping). Page content is written unchanged.
    """
    out: list[_Entry] = []
    by_rel = {e.relpath: e for e in entries}
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
        if target_rel in by_rel and by_rel[target_rel] is not entry:
            # Collision: keep both, defer to dedup. The cold copy survives in
            # place (still status hot) so dedup groups it with the hot one.
            out.append(entry)
            continue
        target = vault.wiki_dir / target_rel
        # Content is unchanged (it was reheated in place by wiki_note).
        atomic_write_text(target, serialize_page(entry.page))
        entry.path.unlink(missing_ok=True)
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
    vault: Vault, entries: list[_Entry], report: LintReport
) -> list[_Entry]:
    """Merge logically-hot pages sharing ``(type, slug.lower())``.

    A page is *logically hot* iff ``page.status == "hot"`` -- this includes a
    deferred reheat collision (a ``status: hot`` page still physically under
    ``.cold/`` because :func:`_reheat_relocate` could not relocate it without
    clobbering an existing hot page). Such collisions MUST be resolved here
    (design §4 / pinned contract: reheat collisions are deferred to dedup).
    Genuinely cold pages (``status == "cold"``) are never touched.

    keeper = max by ``page.updated`` (ISO string compare); tie-break = POSIX
    relpath ascending (the relpath that sorts first wins). Losers processed
    in deterministic order (relpath ascending). For each loser the keeper's
    body gains ``\\n\\n## merged from {loser_relpath}\\n\\n{loser.body}``
    (skipped if that exact header is already present -- idempotence). The
    keeper's ``updated``/``last_touched`` become the max (ISO string) of
    keeper and loser; ``created`` is preserved. Every loser file is deleted.
    A surviving keeper always ends up at its **hot** location: if the keeper
    was a deferred collision (physically under ``.cold/``) it is written to
    ``wiki/{folder}/{slug}.md`` and the cold file removed. Single-member
    groups are returned untouched (no spurious write).
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

        keeper.page.body = body
        keeper.page.updated = updated
        keeper.page.last_touched = last_touched

        # The merged keeper always belongs at its hot location. If it was a
        # deferred reheat collision (still under .cold/), relocate it out.
        slug = _slug_of(keeper.relpath)
        try:
            hot_rel = _hot_relpath(vault, keeper.page, slug)
        except KeyError:  # unknown type: keep it where it is
            hot_rel = keeper.relpath
        hot_path = vault.wiki_dir / hot_rel
        if keeper.in_cold and hot_rel != keeper.relpath:
            atomic_write_text(hot_path, serialize_page(keeper.page))
            keeper.path.unlink(missing_ok=True)
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
    """
    hot_entries = [e for e in entries if not e.in_cold]
    broken: list[tuple[str, str]] = []
    for entry in sorted(hot_entries, key=lambda e: e.relpath):
        for ref in entry.page.links_out:
            hot_target = vault.wiki_dir / f"{ref}.md"
            cold_target = vault.wiki_dir / _COLD_COMPONENT / f"{ref}.md"
            if hot_target.exists() or cold_target.exists():
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

    The total file must not exceed ``schema.moc_max_lines`` lines: oldest
    ``## Recent`` entries are dropped until it fits. ``moc_max_lines`` is a
    SOFT target -- if ``# Memory`` + ``## Map`` alone already exceed it, Map
    is still emitted in full (correctness over the soft cap).
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
        recent_lines.append(f"- [[{folder}/{slug}]] — {e.page.title}\n")

    cap = vault.schema.moc_max_lines

    def assemble(n_recent: int) -> str:
        parts = [head] + map_lines + ["## Recent\n"] + recent_lines[:n_recent]
        return "".join(parts)

    n = len(recent_lines)
    content = assemble(n)
    # splitlines() count == number of trailing-newline-terminated lines here.
    while n > 0 and len(content.splitlines()) > cap:
        n -= 1
        content = assemble(n)
    return content


class _DescStr(str):
    """A string that sorts in DESCENDING order (for stable multi-key sort)."""

    def __lt__(self, other):  # type: ignore[override]
        return str.__gt__(self, other)

    def __le__(self, other):  # type: ignore[override]
        return str.__ge__(self, other)


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
    entries = _reheat_relocate(vault, entries, report)
    entries = _stale_to_cold_impl(vault, entries, report, today)
    entries = _dedup(vault, entries, report)
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
) -> list[_Entry]:
    """Move hot-located stale pages into ``.cold/`` (status flipped cold).

    Considers only hot-located pages (``not in_cold``); runs after reheat so
    a freshly reheated page (last_touched == today) is never re-cooled.
    """
    out: list[_Entry] = []
    cooled: list[str] = []
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
        entry.page.status = "cold"
        target = vault.wiki_dir / target_rel
        atomic_write_text(target, serialize_page(entry.page))
        entry.path.unlink(missing_ok=True)
        cooled.append(entry.relpath)
        out.append(
            _Entry(
                relpath=target_rel,
                page=entry.page,
                path=target,
                in_cold=True,
            )
        )
    report.cooled = sorted(cooled)
    out.sort(key=lambda e: e.relpath)
    return out
