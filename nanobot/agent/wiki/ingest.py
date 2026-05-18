"""Ingest phase: a conversation-history batch -> wiki pages (Task 4.4).

Ingest is the Dream-side *writer* of the wiki tree (design §3). Given a batch
of unprocessed ``history.jsonl`` entries it makes ONE cheap LLM call (no
tools, no AgentRunner — exactly like Dream Phase 1, ``memory.py``) asking the
model to emit a constrained **line protocol**, then *deterministically*
parses that protocol and applies it to the vault. It is paired with Lint
(Task 4.3) which runs immediately AFTER Ingest, under the same per-vault
lock (wired in Task 4.5): Ingest writes hot, schema-typed, round-trippable
pages and Lint then curates/relocates/indexes them into the MOC.

Two properties are load-bearing:

* **Single LLM call, no tools** — mirrors Dream Phase 1's
  ``provider.chat_with_retry(model=..., messages=[system, user],
  tools=None, tool_choice=None)`` and extracts text the same way, but
  *defensively* (I1): ``getattr(response, "content", None) or ""`` (a
  provider returning a bare ``str`` / an object without ``.content`` must
  not raise into the Dream cycle), a non-``str`` content or a
  ``finish_reason == "error"`` response is treated as "no output" (empty
  report, no writes), like ``Consolidator.archive``. An empty history batch
  returns an empty report WITHOUT calling the provider.
* **Deterministic parse + apply** — :func:`_parse_protocol` is a pure
  function of the model's output string; :func:`run_ingest` then applies the
  parsed directives in order against the vault, so the same canned output +
  same starting vault always produces byte-identical files.

Every page Ingest creates or modifies satisfies the Task 4.3 carry-forward
contract so Lint can index it: it round-trips via :func:`serialize_page`
(never hand-rolled frontmatter), is ``status="hot"``, carries ISO
``created``/``updated``/``last_touched`` dates (``date.today().isoformat()``),
has a ``type`` known to ``vault.schema``, is containment-guarded under
``vault.wiki_dir`` (via ``is_under`` — the shared path util ``vault.py``
itself imports), and is NEVER written under ``.cold/`` (only Lint relocates).

Lint's :func:`~nanobot.agent.wiki.lint._dedup` reconciles separate page
*files* that collide on ``(type, slug.lower())`` (keeper = max ``updated``,
loser body appended under ``## merged from``, loser deleted). It does NOT,
and is not designed to, deduplicate *repeated paragraphs WITHIN a single
page body* — that is empirically confirmed: a body containing the same text
three times stays tripled after a Lint pass. Ingest therefore cannot rely on
Lint to absorb a re-delivered batch: if the Dream cycle re-runs the SAME
model output against the SAME vault (a retry, a crash-resume), a naive
append would grow the target body without bound. Two guards make Ingest
convergent under that retry:

* a PAGE onto an existing path falls back to APPEND (prior knowledge is
  never overwritten); and
* every append/contradiction path FIRST checks whether the (clamped,
  stripped) new text is already a substring of the current target body and
  SKIPS the write when it is, so re-delivering the same batch+output is a
  no-op (idempotent on Dream retry — C2). Lint's file-level dedup is
  orthogonal and still does its own (separate-file) job.

Two further deterministic bounds keep a hostile/runaway completion from
bloating the vault under the per-vault Dream lock (C1): at most
``_MAX_DIRECTIVES`` directives are applied (first N by position; the excess
are dropped and counted in ``IngestReport.dropped``) and each directive body
is clamped to ``_MAX_BODY_CHARS`` characters before any Page is built or any
append happens. Both are pure functions of the model output + position, in
the spirit of the existing prompt-side caps
(``_HISTORY_ENTRY_PREVIEW_MAX_CHARS`` / ``_EXISTING_PAGES_MAX``).
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import _COLD_COMPONENT, _NON_PAGE_NAMES, Vault
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.helpers import safe_filename, truncate_text

__all__ = ["IngestReport", "run_ingest"]


# Per-entry preview cap. Replicated from Dream's
# ``Dream._HISTORY_ENTRY_PREVIEW_MAX_CHARS`` (nanobot/agent/memory.py) rather
# than imported: importing ``Dream`` here would pull the whole memory module
# (AgentRunner, GitStore, tiktoken, …) into the lean wiki layer for a single
# int. Keep this value in lockstep with that constant if it ever changes.
_HISTORY_ENTRY_PREVIEW_MAX_CHARS = 4_000

# Cap on the existing-pages listing fed to the model (sorted, first N) so a
# huge vault cannot blow up the prompt.
_EXISTING_PAGES_MAX = 200

# C1 (DoS under the per-vault Dream lock): the model output is fully
# untrusted and ``run_ingest`` runs under that lock every cycle. Without a
# bound a single completion could ask Ingest to write tens of thousands of
# pages or a multi-MB body, all under the lock. These are deterministic
# (pure function of model output + directive position), in the spirit of the
# prompt-side caps above:
#
# * ``_MAX_DIRECTIVES`` — at most this many directives are APPLIED, taken as
#   the first N by document order; the rest are dropped and counted in
#   ``IngestReport.dropped``. 200 mirrors ``_EXISTING_PAGES_MAX`` (a single
#   Dream batch curating > 200 distinct durable pages is already pathological;
#   the surplus is reported, not silently lost-without-trace).
# * ``_MAX_BODY_CHARS`` — every directive body is truncated to this many
#   characters BEFORE a Page/append is built (deterministic prefix). 8000 is
#   2x the per-entry history preview (``_HISTORY_ENTRY_PREVIEW_MAX_CHARS`` =
#   4000): a durable wiki note distilled from history should never need more
#   than a couple of preview windows of prose.
_MAX_DIRECTIVES = 200
_MAX_BODY_CHARS = 8_000

# Heading used to collect contradictions on a page (design §3/§5). The exact
# string is part of the on-disk contract the test asserts.
_CONTRADICTION_HEADING = "## ⚠ contradiction"

# --------------------------------------------------------------------------- #
# Slug policy — a DELIBERATE, independently-maintained local copy of the
# OBSERVABLE rule of ``nanobot.agent.tools.wiki_note._safe_slug`` +
# ``_has_unicode_control`` + ``_is_reserved_slug`` (M1).
#
# It is NOT imported: that would add a tools->wiki layering import (wiki must
# not depend on the agent-tools package). It is a *policy-parity copy*: the
# pipeline below is byte-for-byte the same observable transform + reject
# decision as ``wiki_note``'s, so the agent tool and Ingest produce the SAME
# filename for the SAME entity. (A prior version of this module ASCII-folded
# and lowercased — ``Café``->``cafe``, ``日本語``->`` `` — which silently
# DIVERGED from ``wiki_note`` (``Café``->``Café``); dual-write then created
# two physical files for one entity that Lint's ``(type, slug.lower())``
# dedup could never reconcile (``café`` vs ``cafe``) → a permanent duplicate.
# The two are now pinned together by ``test_slug_policy_parity_with_wiki_note``
# which compares against the REAL ``wiki_note`` helpers and fails on drift.)
# Consolidation path: a future shared ``nanobot.agent.wiki`` (or a neutral
# ``utils``) slug util that BOTH ``wiki_note`` and Ingest import — until then
# this copy + the parity test is the contract.
#
# Pipeline (identical to ``wiki_note._safe_slug``): fold path separators
# (``/ \\``), ASCII control (< 0x20) and any whitespace run to ``-``; then
# ``safe_filename`` (``[<>:"/\\|?*]`` -> ``_``, ``.strip()``); collapse
# ``-`` runs; strip leading/trailing ``-._``; clamp 80 (re-strip ``-._``).
# Benign Unicode letters/digits and CASE are PRESERVED (NOT an ASCII fold).
_SLUG_UNSAFE = re.compile(r"[/\\\x00-\x1f]|\s")
_SLUG_DASH_RUN = re.compile(r"-{2,}")
_SLUG_CONTROL = re.compile(r"[\x00-\x1f]")
_SLUG_MAX_LEN = 80
# Reserved page basenames a slug must not produce (kept in lockstep with the
# authoritative vault constants — no hardcoded duplicate). Mirrors
# ``wiki_note._is_reserved_slug``'s ``_reserved_stems`` exactly (no
# ``.lower()`` fold: case is preserved, so a raw ``SCHEMA`` yields the stem
# ``SCHEMA``, which is in ``_NON_PAGE_NAMES`` via the basename check).
_RESERVED_STEMS = {n[:-3] for n in _NON_PAGE_NAMES if n.endswith(".md")}


def _safe_slug(raw: str) -> str:
    """Sanitize a model-supplied slug to a single safe path component.

    DELIBERATE policy-parity local copy of
    :func:`nanobot.agent.tools.wiki_note._safe_slug` (see the module comment
    above): same observable transform, deliberately NOT imported to avoid a
    tools->wiki layering import, pinned by a parity test. Fold separators /
    ASCII control / whitespace to ``-``, run ``safe_filename``, collapse
    ``-`` runs, strip ``-._``, clamp 80 (re-strip). Benign Unicode
    letters/digits AND case are preserved (this is NOT an ASCII fold).
    Returns ``""`` when nothing safe remains.
    """
    s = _SLUG_UNSAFE.sub("-", raw)
    s = safe_filename(s)
    s = _SLUG_DASH_RUN.sub("-", s)
    s = s.strip("-._")
    if len(s) > _SLUG_MAX_LEN:
        s = s[:_SLUG_MAX_LEN].rstrip("-._")
    return s


def _has_unicode_control(raw: str) -> bool:
    """Whether ``raw`` has a Unicode ``Cc``/``Cf`` codepoint.

    DELIBERATE policy-parity local copy of
    :func:`nanobot.agent.tools.wiki_note._has_unicode_control`: widens the
    ASCII-control reject to the full Unicode control (``Cc``) + format
    (``Cf``) classes (bidi-override / zero-width / BOM). ASCII control is a
    strict ``Cc`` subset, so this subsumes ``_SLUG_CONTROL``. Benign Unicode
    letters/digits (accented latin, CJK) are not ``Cc``/``Cf`` and pass.
    """
    return any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in raw)


def _slug_ok(raw: str, safe: str) -> bool:
    """Whether ``raw``/``safe`` is an acceptable, non-reserved page slug.

    DELIBERATE policy-parity local copy of the ``wiki_note._do_create``
    reject gate (``_SLUG_CONTROL`` / ``_has_unicode_control`` / empty /
    :func:`nanobot.agent.tools.wiki_note._is_reserved_slug`). Rejects: any
    ASCII or Unicode ``Cc``/``Cf`` control char in the RAW slug; an empty
    sanitized slug; a sanitized basename colliding with a reserved/non-page
    file (``SCHEMA.md`` / ``_index.md``) or the cold-archive marker or a
    reserved stem; or a structural/hidden raw intent (leading ``.`` / ``_``,
    or the bare cold marker). Same observable decision as ``wiki_note``'s.
    """
    if _SLUG_CONTROL.search(raw) or _has_unicode_control(raw):
        return False
    if not safe:
        return False
    # _is_reserved_slug parity: sanitized basename collides with a non-page
    # name, OR sanitized form is the cold marker / a reserved stem, OR the
    # raw intent is hidden/structural (leading '.'/'_') or the bare cold
    # marker. No ``.lower()`` fold — case is preserved (parity with
    # wiki_note: a raw ``SCHEMA`` keeps stem ``SCHEMA``).
    if f"{safe}.md" in _NON_PAGE_NAMES:
        return False
    if safe == _COLD_COMPONENT or safe in _RESERVED_STEMS:
        return False
    stripped = raw.strip()
    if stripped.startswith((".", "_")) or stripped == _COLD_COMPONENT:
        return False
    return True


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class IngestReport:
    """Structured outcome of one Ingest run.

    List ordering follows directive application order (a pure deterministic
    function of the model output + starting vault state); it is intentionally
    NOT sorted. ``changed`` is True iff at least one write happened.
    """

    created: list[str] = field(default_factory=list)
    appended: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    unknown: list[tuple[str, str]] = field(default_factory=list)
    malformed_lines: int = 0
    skipped: bool = False
    # C1: directives parsed beyond ``_MAX_DIRECTIVES`` and therefore NOT
    # applied this run (deterministic — the surplus past the first N by
    # document order). Counted, never silently dropped without a trace.
    dropped: int = 0
    # C2: append/contradiction writes that were SKIPPED because the
    # (clamped, stripped) new text was already a substring of the target
    # body — the idempotence guard that makes a re-delivered Dream batch a
    # no-op. A skipped page is NOT added to created/appended/contradictions.
    skipped_duplicate: int = 0

    @property
    def changed(self) -> bool:
        """Whether any page was written this run."""
        return bool(self.created or self.appended or self.contradictions)


# --------------------------------------------------------------------------- #
# Line protocol (pure parser)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Directive:
    """One parsed protocol directive.

    ``kind`` is one of ``PAGE``/``APPEND``/``CONTRADICTION``. ``type`` and
    ``slug`` are the RAW header tokens (sanitization happens at apply time so
    the parser stays a pure string->structure function). ``body`` is the
    verbatim body text (already rstripped of trailing whitespace/newlines).
    """

    kind: str
    type: str
    slug: str
    body: str


# A directive header occupies a whole line, starts at column 0. M4: ALL
# THREE reference directives now use the SAME ``type/slug`` (slash)
# separator so a real model cannot emit the wrong form and silently lose a
# page (the old ``[PAGE <type> <slug>]`` space form diverged from the
# slash-separated APPEND/CONTRADICTION — a model copying the APPEND shape
# for a PAGE produced an unmatched line that was silently dropped):
#   [PAGE <type>/<slug>]
#   [APPEND <type>/<slug>]
#   [CONTRADICTION <type>/<slug>]
#   [SKIP]
# ``type`` is everything up to the first ``/`` (no ``/`` or ``]``);
# ``slug`` is everything after it up to the closing ``]`` — it may still
# contain spaces / punctuation the model emitted (e.g. ``Bad Slug!!``);
# ``_safe_slug`` sanitizes it deterministically at apply time so the parser
# stays a pure string scan.
_RE_PAGE = re.compile(r"^\[PAGE[ \t]+([^/\]]+)/(.+?)\]$")
_RE_REF = re.compile(r"^\[(APPEND|CONTRADICTION)[ \t]+([^/\]]+)/([^/\]]+)\]$")
_RE_SKIP = re.compile(r"^\[SKIP\]$")
# Any line that *looks* like a directive (starts with '[', ends with ']',
# first token uppercase) but did not match the precise grammar above is a
# MALFORMED header (counted), distinct from arbitrary prose.
_RE_DIRECTIVE_SHAPE = re.compile(r"^\[[A-Z][A-Z]*(\b|\]|[ \t]).*\]$")


def _parse_protocol(text: str) -> tuple[list[_Directive], bool, int]:
    """Parse the line protocol. PURE function of ``text``.

    Returns ``(directives, skip, malformed_lines)``:

    * ``directives`` — recognized PAGE/APPEND/CONTRADICTION directives in
      document order, each with its body (the lines after the header up to
      the next header or EOF, trailing whitespace stripped).
    * ``skip`` — True iff a lone ``[SKIP]`` directive appeared anywhere.
    * ``malformed_lines`` — count of lines that looked like a directive
      header but did not match the grammar (a malformed header). Arbitrary
      prose that is consumed as a directive body is NOT counted; stray prose
      OUTSIDE any directive body is also not counted (it is simply ignored)
      — only directive-shaped-but-invalid header lines increment this, which
      is the testable, deterministic signal of a protocol violation.

    Determinism: line-by-line scan, no dict iteration, no clocks.
    """
    directives: list[_Directive] = []
    skip = False
    malformed = 0

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m_page = _RE_PAGE.match(line)
        m_ref = _RE_REF.match(line)
        if _RE_SKIP.match(line):
            skip = True
            i += 1
            continue
        if m_page or m_ref:
            if m_page:
                kind = "PAGE"
                type_, slug = m_page.group(1).strip(), m_page.group(2).strip()
            else:
                kind = m_ref.group(1)
                type_, slug = m_ref.group(2).strip(), m_ref.group(3).strip()
            # Body = subsequent lines until the next header line / EOF.
            # M3: ``_is_header`` includes ``[SKIP]``, so an embedded
            # ``[SKIP]`` line acts as a body TERMINATOR — the directive's
            # body is DELIBERATELY truncated there (silent, deterministic).
            # Consequently a ``[SKIP]`` emitted alongside other directives
            # does NOT suppress them (it only ends the preceding body and
            # sets ``skip``, which only no-ops the run when there are ZERO
            # directives): the template instructs the model to emit
            # ``[SKIP]`` ALONE; mixing it in is off-contract and the SKIP is
            # effectively ignored beyond cutting that one body.
            body_lines: list[str] = []
            j = i + 1
            while j < n and not _is_header(lines[j]):
                body_lines.append(lines[j])
                j += 1
            body = "\n".join(body_lines).strip()
            directives.append(_Directive(kind, type_, slug, body))
            i = j
            continue
        # Not a recognized directive. If it merely *looks* like one (a
        # directive-shaped bracket header) it is a malformed header; plain
        # prose is silently ignored.
        if _RE_DIRECTIVE_SHAPE.match(line.strip()):
            malformed += 1
        i += 1

    return directives, skip, malformed


def _is_header(line: str) -> bool:
    """Whether ``line`` starts a NEW directive (terminates a body block)."""
    return bool(
        _RE_PAGE.match(line)
        or _RE_REF.match(line)
        or _RE_SKIP.match(line)
    )


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def _today() -> str:
    return datetime.date.today().isoformat()


def _resolved_in_vault(vault: Vault, type_: str, slug: str) -> tuple[Any, Any]:
    """Return ``(page_path, resolved)`` where ``resolved`` is the containment-
    guarded absolute path, or ``(path, None)`` if it escapes the vault.
    """
    path = vault.page_path(type_, slug)
    resolved = path.resolve()
    if not is_under(resolved, vault.wiki_dir):
        return path, None
    return path, resolved


# I2: max rendered title length. 120 matches
# ``wiki_note._search_snippet``'s ``_SNIPPET_MAX_LEN`` / ``lint._safe_inline``
# so a derived title never exceeds the width those layers already enforce.
_TITLE_MAX_CHARS = 120
_TITLE_WS_RUN = re.compile(r"\s+")


def _clamp_body(body: str) -> str:
    """Deterministically clamp a directive body to ``_MAX_BODY_CHARS`` (C1).

    A pure prefix truncation (the only deterministic clamp): the model
    output is untrusted and ``run_ingest`` writes under the per-vault Dream
    lock, so an over-long body must be bounded BEFORE any Page is built or
    any append happens. ``str`` slicing is by codepoint, so the result is
    always valid text regardless of where the cut lands.
    """
    if len(body) <= _MAX_BODY_CHARS:
        return body
    return body[:_MAX_BODY_CHARS]


def _derive_title(body: str, slug: str) -> str:
    """First non-empty body line, clamped, falling back to ``slug`` (I2).

    Deterministic single-line title: take the first non-empty body line,
    collapse internal whitespace runs to one space, and hard-cap at
    ``_TITLE_MAX_CHARS``. ``body`` is already ``_clamp_body``-bounded so a
    pathological body cannot make the title scan unbounded, but the explicit
    length cap pins the title regardless (a single 5000-char first line is
    still clamped to a readable, single-line title). Falls back to ``slug``
    when the body has no non-empty line (existing behaviour preserved).
    """
    for raw in body.splitlines():
        if raw.strip():
            line = _TITLE_WS_RUN.sub(" ", raw).strip()
            if len(line) > _TITLE_MAX_CHARS:
                line = line[:_TITLE_MAX_CHARS].rstrip()
            return line or slug
    return slug


def _rel(vault: Vault, type_: str, slug: str) -> str:
    """POSIX relpath (relative to wiki_dir) for reporting."""
    return f"{vault.schema.folder(type_)}/{slug}.md"


def _apply_page(
    vault: Vault, d: _Directive, slug: str, report: IngestReport
) -> None:
    """PAGE: create a new page; if it already exists, fall back to APPEND
    (never overwrite — prior knowledge is preserved; Lint merges if needed).

    ``d.body`` is already ``_clamp_body``-bounded by ``_apply`` (C1).
    """
    _path, resolved = _resolved_in_vault(vault, d.type, slug)
    if resolved is None:
        report.unknown.append((d.type, d.slug))
        return
    rel = _rel(vault, d.type, slug)
    if resolved.exists():
        # Fall back to append so we never clobber an existing page (with the
        # C2 idempotence guard inside _append_body).
        _append_body(vault, resolved, d.body, report, rel)
        return
    today = _today()
    page = Page(
        type=d.type,
        title=_derive_title(d.body, slug),
        status="hot",
        created=today,
        updated=today,
        last_touched=today,
        tags=[],
        links_out=[],
        pinned=None,
        body=d.body,
    )
    atomic_write_text(resolved, serialize_page(page))
    report.created.append(rel)


def _body_already_present(existing_body: str, new_text: str) -> bool:
    """C2 idempotence guard: is ``new_text`` already in ``existing_body``?

    Cheap containment check used before EVERY append/contradiction write so
    re-delivering the SAME model output against the SAME vault on a Dream
    retry / crash-resume is a no-op (the body does not grow). Compared on
    the ``strip()``ed text (the append paths interpolate the directive body
    stripped) so whitespace-only framing differences do not defeat the
    guard. An empty ``new_text`` is treated as "present" (nothing to add).
    Lint's file-level ``(type, slug.lower())`` dedup does NOT dedup repeated
    paragraphs within one body (empirically confirmed) — hence this guard
    lives here, in Ingest, not there.
    """
    needle = new_text.strip()
    if not needle:
        return True
    return needle in existing_body


def _append_body(
    vault: Vault,
    resolved: Any,
    body: str,
    report: IngestReport,
    rel: str,
) -> None:
    """Append ``body`` to an existing page's body with one blank-line
    separator; bump ``updated``/``last_touched``; keep all else.

    A malformed page already on disk is left untouched (Lint owns malformed)
    and counted via ``report.malformed_lines``.

    C2 idempotence: if the (stripped) ``body`` is ALREADY a substring of the
    current page body, SKIP the append entirely (no write, no date bump) and
    count it in ``report.skipped_duplicate`` — re-running the same batch is
    convergent. ``body`` is already ``_clamp_body``-bounded by ``_apply``.
    """
    try:
        page = parse_page(resolved.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        report.malformed_lines += 1
        return
    if _body_already_present(page.body, body):
        # Already present (or empty): re-delivered batch / no-op. Do NOT add
        # to report.appended and do NOT bump dates — keeps the file
        # byte-stable on a Dream retry.
        report.skipped_duplicate += 1
        return
    prefix = page.body.rstrip("\n")
    page.body = f"{prefix}\n\n{body}" if prefix else body
    today = _today()
    page.updated = today
    page.last_touched = today
    atomic_write_text(resolved, serialize_page(page))
    report.appended.append(rel)


def _apply_append(
    vault: Vault, d: _Directive, slug: str, report: IngestReport
) -> None:
    """APPEND: append to the page; if missing, create it (knowledge must not
    be lost — recorded in ``report.created``).
    """
    _path, resolved = _resolved_in_vault(vault, d.type, slug)
    if resolved is None:
        report.unknown.append((d.type, d.slug))
        return
    rel = _rel(vault, d.type, slug)
    if not resolved.exists():
        today = _today()
        page = Page(
            type=d.type,
            title=_derive_title(d.body, slug),
            status="hot",
            created=today,
            updated=today,
            last_touched=today,
            tags=[],
            links_out=[],
            pinned=None,
            body=d.body,
        )
        atomic_write_text(resolved, serialize_page(page))
        report.created.append(rel)
        return
    _append_body(vault, resolved, d.body, report, rel)


def _apply_contradiction(
    vault: Vault, d: _Directive, slug: str, report: IngestReport
) -> None:
    """CONTRADICTION: target MUST exist; append the text under a single
    ``## ⚠ contradiction`` heading. The pre-existing body above is NEVER
    modified. Missing target -> skip + report.unknown.
    """
    _path, resolved = _resolved_in_vault(vault, d.type, slug)
    if resolved is None or not resolved.exists():
        report.unknown.append((d.type, d.slug))
        return
    try:
        page = parse_page(resolved.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        report.malformed_lines += 1
        return
    if _body_already_present(page.body, d.body):
        # C2: the contradiction text is already recorded on this page (a
        # re-delivered Dream batch / crash-resume). SKIP — do NOT re-append
        # the bullet, do NOT bump dates; the file stays byte-stable so the
        # rerun is a no-op. ``d.body`` is _clamp_body-bounded by _apply.
        report.skipped_duplicate += 1
        return
    body = page.body.rstrip("\n")
    if _CONTRADICTION_HEADING in page.body:
        # Section already present: append the new text as a further bullet
        # beneath it (the body above the heading is preserved verbatim).
        new_body = f"{body}\n\n- {d.body}" if body else f"- {d.body}"
    else:
        lead = f"{body}\n\n" if body else ""
        new_body = f"{lead}{_CONTRADICTION_HEADING}\n\n- {d.body}"
    page.body = new_body
    today = _today()
    page.updated = today
    page.last_touched = today
    atomic_write_text(resolved, serialize_page(page))
    report.contradictions.append(_rel(vault, d.type, slug))


def _apply(vault: Vault, directives: list[_Directive], report: IngestReport) -> None:
    """Apply directives IN ORDER against the vault.

    C1 bounds (deterministic, under the per-vault Dream lock):

    * At most ``_MAX_DIRECTIVES`` directives are applied — the FIRST N by
      document order (``directives`` is already in document order from
      ``_parse_protocol``). The surplus is dropped and counted in
      ``report.dropped`` (``len(directives) - _MAX_DIRECTIVES``); it is a
      pure function of position so it is fully deterministic.
    * Each directive body is clamped to ``_MAX_BODY_CHARS`` BEFORE a Page is
      built or any append happens (``_clamp_body``), so a hostile multi-MB
      body cannot be persisted under the lock.
    """
    if len(directives) > _MAX_DIRECTIVES:
        report.dropped += len(directives) - _MAX_DIRECTIVES
        directives = directives[:_MAX_DIRECTIVES]
    for d in directives:
        if not vault.schema.is_known_type(d.type):
            report.unknown.append((d.type, d.slug))
            continue
        slug = _safe_slug(d.slug)
        if not _slug_ok(d.slug, slug):
            report.unknown.append((d.type, d.slug))
            continue
        # C1: clamp the body deterministically before any write. _Directive
        # is frozen, so build a body-clamped copy (parser stays pure; the
        # clamp is an apply-time bound, like the prompt-side caps).
        clamped = d.body if len(d.body) <= _MAX_BODY_CHARS else _clamp_body(d.body)
        if clamped is not d.body:
            d = _Directive(d.kind, d.type, d.slug, clamped)
        if d.kind == "PAGE":
            _apply_page(vault, d, slug, report)
        elif d.kind == "APPEND":
            _apply_append(vault, d, slug, report)
        elif d.kind == "CONTRADICTION":
            _apply_contradiction(vault, d, slug, report)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def _build_history_text(history_entries: list[dict]) -> str:
    """``[{timestamp}] {content}`` per entry, content capped — mirrors Dream
    Phase 1's history-text build (memory.py).
    """
    return "\n".join(
        f"[{e.get('timestamp', '?')}] "
        f"{truncate_text(str(e.get('content', '')), _HISTORY_ENTRY_PREVIEW_MAX_CHARS)}"
        for e in history_entries
    )


def _build_existing_pages(vault: Vault) -> str:
    """``- {type}/{slug}: {title}`` per hot page, sorted, capped."""
    rows: list[str] = []
    for rel, page in vault.iter_pages(include_cold=False):
        # rel is ``{folder}/{slug}.md``; present folder/slug + title.
        stem = rel[: -len(".md")] if rel.endswith(".md") else rel
        title = " ".join(page.title.split())
        rows.append(f"- {stem}: {title}")
    rows.sort()
    return "\n".join(rows[:_EXISTING_PAGES_MAX])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
async def run_ingest(
    vault: Vault,
    history_entries: list[dict],
    provider: Any,
    model: str,
    render_template: Callable[..., str],
) -> IngestReport:
    """Turn a history batch into wiki pages via one LLM call + a deterministic
    parse/apply.

    Returns an :class:`IngestReport`. An empty ``history_entries`` is a no-op
    (the provider is NOT called). The caller (Task 4.5) invokes this BEFORE
    :func:`nanobot.agent.wiki.lint.run_lint`, per vault, under the per-vault
    ``get_vault_lock`` critical section.
    """
    report = IngestReport()
    if not history_entries:
        return report

    history_text = _build_history_text(history_entries)
    existing_pages = _build_existing_pages(vault)
    allowed_types = ", ".join(sorted(vault.schema.types))

    system = render_template(
        "agent/wiki_ingest.md",
        strip=True,
        allowed_types=allowed_types,
    )
    user = (
        f"## Conversation History\n{history_text}\n\n"
        f"## Existing Pages\n{existing_pages or '(none)'}"
    )

    try:
        response = await provider.chat_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
            tool_choice=None,
        )
    except Exception:
        logger.exception("Ingest LLM call failed")
        return report

    if getattr(response, "finish_reason", None) == "error":
        logger.warning(
            "Ingest LLM returned error: {}",
            getattr(response, "content", None),
        )
        return report

    # I1: the response is off-contract-untrusted just like its content. A
    # provider returning a bare ``str`` (or any object without ``.content``)
    # must NOT raise an ``AttributeError`` into the Dream cycle — read
    # ``content`` defensively, symmetric with the ``getattr`` used for
    # ``finish_reason`` just above. A non-LLMResponse / missing-or-None
    # content is treated as empty output → empty report, no writes, no
    # crash (same outcome as a ``finish_reason == "error"`` response).
    output = getattr(response, "content", None) or ""
    if not isinstance(output, str):
        # Content present but not text (e.g. a list of blocks / None-like):
        # do not attempt to parse a non-string; treat as no output.
        logger.warning(
            "Ingest LLM returned non-text content ({}); treating as empty",
            type(output).__name__,
        )
        return report
    directives, skip, malformed = _parse_protocol(output)
    report.malformed_lines += malformed

    if skip and not directives:
        # The model decided nothing is worth persisting: perform NO writes.
        report.skipped = True
        return report

    _apply(vault, directives, report)
    return report
