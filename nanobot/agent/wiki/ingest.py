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
  tools=None, tool_choice=None)`` and extracts text the same way
  (``response.content or ""``; a ``finish_reason == "error"`` response is
  treated as "no output", like ``Consolidator.archive``). An empty history
  batch returns an empty report WITHOUT calling the provider.
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
Colliding pages are losslessly merged by Lint's dedup, so Ingest performs no
dedup of its own; a PAGE onto an existing path falls back to APPEND so prior
knowledge is never overwritten.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import _COLD_COMPONENT, _NON_PAGE_NAMES, Vault
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.helpers import truncate_text

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

# Heading used to collect contradictions on a page (design §3/§5). The exact
# string is part of the on-disk contract the test asserts.
_CONTRADICTION_HEADING = "## ⚠ contradiction"

# Slug sanitization (mirrors the *policy* of
# ``nanobot.agent.tools.wiki_note._safe_slug`` / ``_is_reserved_slug``).
# Deliberately NOT imported: importing the tool would create a tools->wiki
# layering import (wiki must not depend on the agent-tools package); the two
# are kept independent. A future task may consolidate them — do NOT import
# the tools one. Same observable rule: fold separators / control / whitespace
# to '-', keep only [a-z0-9._-] after lowercasing, collapse '-', strip
# leading/trailing '-._', cap at 80; reject empty / reserved / dot-leading.
_SLUG_UNSAFE = re.compile(r"[^a-z0-9._-]")
_SLUG_DASH_RUN = re.compile(r"-{2,}")
_SLUG_MAX_LEN = 80
# Reserved page basenames a slug must not produce (kept in lockstep with the
# authoritative vault constants — no hardcoded duplicate). Compared
# case-insensitively: ``_safe_slug`` lowercases, so a raw ``SCHEMA`` slug
# yields ``schema`` — still a reserved/structural name and must be refused
# (it would otherwise become a phantom ``schema.md`` page).
_RESERVED_STEMS = {n[:-3].lower() for n in _NON_PAGE_NAMES if n.endswith(".md")}


def _safe_slug(raw: str) -> str:
    """Sanitize a model-supplied slug to a single safe path component.

    Conservative & deterministic: lowercase, replace every char outside
    ``[a-z0-9._-]`` (incl. whitespace, separators, control chars) with ``-``,
    collapse ``-`` runs, strip leading/trailing ``-._``, cap at 80 chars.
    Returns ``""`` when nothing safe remains. This intentionally mirrors the
    policy of :func:`nanobot.agent.tools.wiki_note._safe_slug` but is kept
    independent to avoid a tools->wiki layering import.
    """
    s = _SLUG_UNSAFE.sub("-", raw.strip().lower())
    s = _SLUG_DASH_RUN.sub("-", s)
    s = s.strip("-._")
    if len(s) > _SLUG_MAX_LEN:
        s = s[:_SLUG_MAX_LEN].rstrip("-._")
    return s


def _slug_ok(raw: str, safe: str) -> bool:
    """Whether ``safe`` is a usable, non-reserved, non-hidden page slug.

    Rejects: empty (nothing safe remained), a basename colliding with a
    reserved/non-page file (``SCHEMA.md`` / ``_index.md``), the cold-archive
    marker, or a hidden/structural raw intent (leading ``.`` or ``_``).
    """
    if not safe:
        return False
    if safe.lower() in _RESERVED_STEMS:
        return False
    if safe == _COLD_COMPONENT:
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


# A directive header occupies a whole line, starts at column 0:
#   [PAGE <type> <slug>]            -> type/slug space-separated
#   [APPEND <type>/<slug>]          -> type/slug slash-separated
#   [CONTRADICTION <type>/<slug>]
#   [SKIP]
# PAGE: ``type`` is the first whitespace-delimited token; everything after it
# (up to the closing ``]``) is the RAW slug — it may contain spaces / punct
# the model emitted (e.g. ``Bad Slug!!``); ``_safe_slug`` sanitizes it
# deterministically at apply time so the parser stays a pure string scan.
_RE_PAGE = re.compile(r"^\[PAGE[ \t]+(\S+)[ \t]+(.+?)\]$")
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
                kind, type_, slug = "PAGE", m_page.group(1), m_page.group(2)
            else:
                kind = m_ref.group(1)
                type_, slug = m_ref.group(2).strip(), m_ref.group(3).strip()
            # Body = subsequent lines until the next header line / EOF.
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


def _derive_title(body: str, slug: str) -> str:
    """First non-empty trimmed body line, falling back to ``slug``."""
    for raw in body.splitlines():
        if raw.strip():
            return raw.strip()
    return slug


def _rel(vault: Vault, type_: str, slug: str) -> str:
    """POSIX relpath (relative to wiki_dir) for reporting."""
    return f"{vault.schema.folder(type_)}/{slug}.md"


def _apply_page(
    vault: Vault, d: _Directive, slug: str, report: IngestReport
) -> None:
    """PAGE: create a new page; if it already exists, fall back to APPEND
    (never overwrite — prior knowledge is preserved; Lint merges if needed).
    """
    _path, resolved = _resolved_in_vault(vault, d.type, slug)
    if resolved is None:
        report.unknown.append((d.type, d.slug))
        return
    rel = _rel(vault, d.type, slug)
    if resolved.exists():
        # Fall back to append so we never clobber an existing page.
        _append_body(vault, resolved, d.body, report, rel, created_ok=False)
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


def _append_body(
    vault: Vault,
    resolved: Any,
    body: str,
    report: IngestReport,
    rel: str,
    *,
    created_ok: bool,
) -> None:
    """Append ``body`` to an existing page's body with one blank-line
    separator; bump ``updated``/``last_touched``; keep all else.

    A malformed page already on disk is left untouched (Lint owns malformed)
    and counted via ``report.malformed_lines``.
    """
    try:
        page = parse_page(resolved.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        report.malformed_lines += 1
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
    _append_body(vault, resolved, d.body, report, rel, created_ok=True)


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
    """Apply directives IN ORDER against the vault."""
    for d in directives:
        if not vault.schema.is_known_type(d.type):
            report.unknown.append((d.type, d.slug))
            continue
        slug = _safe_slug(d.slug)
        if not _slug_ok(d.slug, slug):
            report.unknown.append((d.type, d.slug))
            continue
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
        logger.warning("Ingest LLM returned error: {}", response.content)
        return report

    output = response.content or ""
    directives, skip, malformed = _parse_protocol(output)
    report.malformed_lines += malformed

    if skip and not directives:
        # The model decided nothing is worth persisting: perform NO writes.
        report.skipped = True
        return report

    _apply(vault, directives, report)
    return report
