"""``wiki_note`` agent tool: read/write pen into the per-user wiki memory.

This is the agent's interface onto the long-term wiki tree (design §3/§4).
Each request is routed to a per-session *vault* under
``<workspace>/memory/users/<slug>/wiki/`` so concurrent users never share
memory (report P3). Session isolation uses a per-instance
:class:`~contextvars.ContextVar`, mirroring :mod:`nanobot.agent.tools.spawn`.

Task 2.1 implements ``operation=read`` and a minimal ``operation=create``.
Task 2.2 hardens ``create`` with the SCHEMA admission gate (unknown-type +
frontmatter validation, design §3/P7), refuses duplicates, scaffolds the
per-type ``_index.md`` Map-of-Content, and runs the page-write + index-append
as one critical section under the per-vault async lock (design H2/Task 0.3).
Task 2.3 adds ``append`` + reheat-on-``read``. Task 2.4 adds ``search``: a
read-only keyword/tag/recency scan over hot AND cold pages (no embeddings,
no SQLite — design §4) that deliberately does NOT reheat (only ``read``
does). This completes Milestone 2's read/write agent pen. The parameter
schema here is intentionally the full stable set so it does not churn
across tasks.
"""

from __future__ import annotations

import asyncio
import datetime
import re
import unicodedata
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.filesystem import _FsTool
from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.paths import vault_dir, vault_slug
from nanobot.agent.wiki.vault import _COLD_COMPONENT, _NON_PAGE_NAMES, Vault
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.helpers import safe_filename
from nanobot.utils.vault_lock import get_vault_lock

_FALLBACK_SESSION_KEY = "unified:default"

# An ASCII control char (codepoint < 0x20 — incl. \n \r \t). Its presence in
# a model-supplied slug is never legitimate: it signals corruption/injection
# and is exactly what would split an ``_index.md`` stub across two physical
# lines (the I1 vector). ``_do_create`` rejects any slug containing one
# outright rather than mangling it into a different page name — that is the
# "do NOT silently truncate at a newline" contract, extended: a control-char
# slug must never silently become a *different* (sanitized) filename either.
_SLUG_CONTROL = re.compile(r"[\x00-\x1f]")
# A path separator (`/ \\`), an ASCII control char, or any whitespace →
# folded to a single `-`. Applied to the RAW slug (before safe_filename, so
# 'a/b c' → 'a-b-c' not 'a_b-c'); safe_filename maps `/ \\` to `_`, which we
# want to avoid for the separator case.
_SLUG_UNSAFE = re.compile(r"[/\\\x00-\x1f]|\s")
_SLUG_DASH_RUN = re.compile(r"-{2,}")

# Conservative cross-platform cap on the sanitized slug *component* (R2).
# A page filename is ``{slug}.md`` plus the atomic-write ``.tmp`` suffix;
# 80 keeps the basename well under every common limit (NAME_MAX 255, and
# Windows' practical per-component ceiling) regardless of the vault's
# absolute path depth, so a long slug is a deterministic *trim* on every
# OS rather than a Linux-accepted / Windows-``WinError 123`` divergence.
_SLUG_MAX_LEN = 80


def _safe_slug(raw: str) -> str:
    """Sanitize a model-supplied slug to a single filename-safe component.

    Pipeline (deterministic, per the I1 rule):

    1. Fold every path separator (``/ \\``), ASCII control char (codepoint
       < 0x20, incl. ``\\n \\r \\t``), and whitespace run to ``-``.
    2. ``safe_filename`` (strips ``[<>:"/\\|?*]`` and trims ends).
    3. Collapse consecutive ``-`` and strip leading/trailing ``-._``.
    4. Clamp to ``_SLUG_MAX_LEN`` chars (truncate, then re-strip trailing
       ``-._`` so a cut never lands on a dangling separator) — R2: a long
       slug is benign, so trim it deterministically instead of letting it
       diverge cross-platform (Linux-accept / Windows ``WinError 123``).

    Returns the sanitized slug, or ``""`` if nothing safe remains (incl.
    a slug that clamps to empty — the existing empty→error path in
    ``_do_create`` handles that). Benign Unicode letters/digits (accented
    latin, CJK) are preserved: this is NOT an ASCII-only fold; the Cc/Cf
    control/format *rejection* (and the empty→error decision) live in
    ``_do_create``, not here. Pure and reused by Task 2.3's append so page
    and append agree on the on-disk filename.
    """
    s = _SLUG_UNSAFE.sub("-", raw)
    s = safe_filename(s)
    s = _SLUG_DASH_RUN.sub("-", s)
    s = s.strip("-._")
    if len(s) > _SLUG_MAX_LEN:
        s = s[:_SLUG_MAX_LEN].rstrip("-._")
    return s


def _has_unicode_control(raw: str) -> bool:
    """Whether ``raw`` contains a Unicode category ``Cc`` or ``Cf`` codepoint.

    R3: widens the ASCII-control reject (``_SLUG_CONTROL``, ``[\\x00-\\x1f]``)
    to the full Unicode control (``Cc``) and format (``Cf``) classes so
    bidi-override / zero-width / BOM chars (U+202E, U+200E/200F, U+2066-2069,
    U+FEFF, U+0085, U+00A0…) can never survive ``_safe_slug`` into a filename
    or an ``_index.md`` wikilink. ASCII control is a strict subset of ``Cc``,
    so this fully subsumes the prior behaviour. Benign Unicode letters/digits
    (accented latin, CJK) are *not* in ``Cc``/``Cf`` and pass through.
    """
    return any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in raw)


def _is_reserved_slug(raw: str, safe_slug: str) -> bool:
    """Whether a slug would yield a non-page / structural / hidden file.

    R1: a page is written at ``<folder>/{safe_slug}.md`` but
    :meth:`Vault.iter_pages` filters by *basename* against
    :data:`nanobot.agent.wiki.vault._NON_PAGE_NAMES` regardless of folder,
    so ``slug="SCHEMA"`` produces a ``SCHEMA.md`` Ingest/Lint/``is_empty``
    can never see — while a dangling stub is still appended to ``_index.md``
    (a silently-broken phantom page reachable from normal model operation).

    Two checks, because ``_safe_slug`` strips leading ``-._`` (so a raw
    ``"_index"`` / ``".cold"`` / ``".hidden"`` has already lost its
    structural prefix by the time we see ``safe_slug``):

    * the *sanitized* basename ``f"{safe_slug}.md"`` colliding with the
      authoritative reserved set (catches ``SCHEMA`` → ``SCHEMA.md``); and
    * the *raw* slug being a structural/hidden/cold name — it starts with
      ``.`` or ``_`` (hidden / ``_index``-class) or its sanitized form is
      the cold-archive marker component or any reserved stem.

    The reserved set is *referenced* from the authoritative ``vault``
    module constants (no hardcoded duplicate) so it stays in lockstep if
    ``iter_pages``' filter ever changes.
    """
    _reserved_stems = {n[:-3] for n in _NON_PAGE_NAMES if n.endswith(".md")}
    stripped = raw.strip()
    if f"{safe_slug}.md" in _NON_PAGE_NAMES:
        return True
    if safe_slug == _COLD_COMPONENT or safe_slug in _reserved_stems:
        return True
    # Hidden / structural raw intent (leading '.' or '_'), and the cold
    # marker spelled with its leading dot.
    if stripped.startswith((".", "_")):
        return True
    if stripped == _COLD_COMPONENT:
        return True
    return False


# Hard cap on search results returned to the model. More than this and a
# single overflow note is appended (the model should refine its query rather
# than be flooded). Kept small + deterministic so search stays model-readable.
_SEARCH_CAP = 20

# Target width of a per-result snippet (title + first body line, single line,
# internal whitespace collapsed). ~120 chars keeps each result compact.
_SNIPPET_MAX_LEN = 120

# Collapses any whitespace run (incl. newlines/tabs) to a single space so a
# multi-line body folds into one readable snippet line.
_WS_RUN = re.compile(r"\s+")


def _search_snippet(page: Page) -> str:
    """One compact line: page ``title`` + first non-empty body line.

    Internal whitespace/newlines are collapsed to single spaces and the
    whole thing is trimmed to ``_SNIPPET_MAX_LEN`` chars (single line) so a
    result list stays model-readable regardless of body shape.
    """
    title = _WS_RUN.sub(" ", page.title).strip()
    first_body = ""
    for raw_line in page.body.splitlines():
        line = _WS_RUN.sub(" ", raw_line).strip()
        if line:
            first_body = line
            break
    snippet = f"{title} — {first_body}" if first_body else title
    snippet = _WS_RUN.sub(" ", snippet).strip()
    if len(snippet) > _SNIPPET_MAX_LEN:
        snippet = snippet[: _SNIPPET_MAX_LEN - 1].rstrip() + "…"
    return snippet


def _terminated_line_set(text: str) -> set[str]:
    """Existing ``_index.md`` lines, each re-terminated with a single ``\\n``.

    Returns a *set* (membership-test only) for the idempotent stub check:
    comparing whole, newline-terminated lines (not a raw substring) so
    ``[[people/al]]`` does not falsely match an existing ``[[people/alice]]``
    stub.
    """
    return {f"{line}\n" for line in text.splitlines()}


@tool_parameters(
    tool_parameters_schema(
        operation=StringSchema(
            "What to do: 'read' a page by path, 'create' a new leaf page, "
            "'append' text to a page, or 'search' pages by keyword/tag/recency."
        ),
        path=StringSchema(
            "For read: page path relative to wiki/, e.g. 'people/alice.md'."
        ),
        type=StringSchema(
            "For create: the page type, e.g. 'people', 'projects', 'concepts'."
        ),
        slug=StringSchema(
            "For create: the filename stem (no extension), e.g. 'alice'."
        ),
        title=StringSchema("For create: the human-readable page title."),
        body=StringSchema("For create: the Markdown body of the page."),
        text=StringSchema("Reserved for the append operation (Task 2.3)."),
        query=StringSchema(
            "For search: a keyword/phrase, a 'tag:NAME' filter, or empty "
            "to list the most recently touched pages."
        ),
        required=["operation"],
    )
)
class WikiNoteTool(_FsTool, ContextAware):
    """Read and create pages in the per-user long-term wiki memory."""

    _scopes = {"core"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-instance ContextVar so concurrent sessions sharing one tool
        # instance do not bleed routing into each other (spawn.py pattern).
        self._session_key_var: ContextVar[str] = ContextVar(
            "wiki_note_session_key", default=_FALLBACK_SESSION_KEY
        )

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        # Master switch (I-1): only register — and therefore only emit the
        # tool's JSON schema into the provider ``tools=`` array — when the
        # resolved ``dream.wiki_enabled`` is true. ``getattr`` default False
        # so a ToolContext that predates / omits the field (subagents, unit
        # tests building a SimpleNamespace ctx) keeps the tool gated OFF,
        # preserving byte-identity with pre-wiki nanobot.
        return bool(getattr(ctx, "wiki_enabled", False))

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        # Reuse _FsTool.create verbatim so _workspace / allowed_dir / sandbox
        # are wired exactly like every other filesystem tool. _FsTool.create
        # ends in ``return cls(...)``, so calling its underlying function with
        # this subclass constructs a fully-initialised WikiNoteTool (our
        # __init__ then sets up the session ContextVar).
        return _FsTool.create.__func__(cls, ctx)

    def set_context(self, ctx: RequestContext) -> None:
        self._session_key_var.set(ctx.session_key or f"{ctx.channel}:{ctx.chat_id}")

    def _session_key(self) -> str:
        return self._session_key_var.get()

    def _vault(self) -> Vault:
        root = vault_dir(Path(self._workspace), self._session_key())
        # Ensure the wiki/ subtree exists so writes (and a freshly created
        # vault's reads) work. The schema still falls back to the bundled
        # master — copying SCHEMA.md into the vault is a later task.
        (root / "wiki").mkdir(parents=True, exist_ok=True)
        return Vault(root)

    def _vault_lock(self) -> asyncio.Lock:
        """The per-vault async lock for this session's vault.

        Lock-key derivation is centralised here so Task 2.3's append
        acquires the *identical* lock as create (same critical section).
        """
        return get_vault_lock(vault_slug(self._session_key()))

    def _resolved_in_vault(self, path: Path, vault: Vault) -> Path | None:
        """Resolve ``path`` and return it iff it stays inside ``vault.wiki_dir``.

        The single containment guard, used for BOTH the page path and the
        ``_index.md`` path (and reused by Task 2.3). Returns ``None`` when
        the resolved path escapes the vault so the caller can refuse without
        ever touching the filesystem (no TOCTOU: the checked path is the one
        handed to ``atomic_write_text``).
        """
        resolved = path.resolve()
        return resolved if is_under(resolved, vault.wiki_dir) else None

    @property
    def name(self) -> str:
        return "wiki_note"

    @property
    def description(self) -> str:
        return (
            "Manage your long-term wiki memory (durable notes about people, "
            "projects, concepts, decisions). Operations:\n"
            "- read: fetch a page by its path relative to wiki/ "
            "(e.g. path='people/alice.md'). Returns frontmatter + body. "
            "Reading a page that had gone cold automatically reheats it "
            "(marks it hot again) so it stays in active memory.\n"
            "- create: add a new leaf page (args: type, slug, title, "
            "optional body). The type is validated against the wiki schema "
            "(unknown types are refused); pages are filed by type and an "
            "index of each type is kept up to date automatically.\n"
            "- append: add text to an existing page (args: path, text). The "
            "text is appended to the page body and the page's freshness is "
            "bumped (it is treated as recently touched). The page must "
            "already exist — use create first for a new page.\n"
            "- search: find pages (arg: query). Searches across BOTH active "
            "(hot) and archived (cold) pages so older knowledge stays "
            "discoverable. 'query' may be: a keyword/phrase (case-insensitive "
            "substring over title, tags, and body, ranked title > tag > "
            "body); a 'tag:NAME' filter (exact, case-insensitive tag match); "
            "or empty/omitted to list the most recently touched pages. "
            "Returns up to 20 result lines, each starting with the page's "
            "path relative to wiki/ — pass that path to operation='read' to "
            "open a result. Search is plain keyword/tag/recency (no "
            "embeddings or semantic search) and is READ-ONLY: unlike read it "
            "does NOT reheat a cold page — only an explicit read reheats.\n"
            "You may only read, create, append to, and search leaf pages. "
            "You cannot move pages to cold storage, merge pages, or rewrite "
            "indexes/MOC files — those are automatic and Dream-only."
        )

    async def execute(self, operation: str | None = None, **kw: Any) -> str:
        if operation == "read":
            return await self._do_read(kw.get("path"))
        if operation == "create":
            return await self._do_create(
                kw.get("type"),
                kw.get("slug"),
                kw.get("title"),
                kw.get("body") or "",
            )
        if operation == "append":
            return await self._do_append(kw.get("path"), kw.get("text"))
        if operation == "search":
            # search is read-only and synchronous (no lock/write/reheat); call
            # the sync helper directly without await.
            return self._do_search(kw.get("query"))
        return f"Error: unknown operation {operation!r}"

    async def _do_read(self, path: str | None) -> str:
        """Read a page; reheat it (cold→hot) iff it is currently cold.

        Refactored async (Task 2.3): the COLD branch must take the per-vault
        lock and persist the reheated page, so the method is ``async`` and
        awaited from ``execute``. The HOT branch is a deliberate pure read —
        NO lock acquisition and NO write — so the common case stays cheap and
        lock-free (an earlier review flagged read-path side effects; only a
        cold page pays the write+lock cost). Idempotent: a second read of a
        now-hot page changes nothing. This never raises to the caller — every
        failure is returned as a model-readable string (Task 2.1 contract).
        """
        if not path:
            return "Error: 'path' is required for read"
        vault = self._vault()
        try:
            page = vault.read_page(path)
        except FileNotFoundError:
            return f"Page not found: {path}"
        except ValueError as e:
            return f"Error: {e}"

        # read_page already enforced wiki-dir containment (vault.py:86); reuse
        # its resolved path so the sandbox check and the vault-traversal guard
        # cannot disagree (no second, divergent resolve).
        target = (vault.wiki_dir / path).resolve()

        # HOT (the common case): pure read, no lock, no write. Return the
        # original bytes verbatim — byte-identical to pre-Task-2.3 behaviour.
        if page.status != "cold":
            try:
                return target.read_text(encoding="utf-8")
            except OSError as e:
                return f"Error: {e}"

        # COLD: reheat under the per-vault lock. Re-read + re-parse inside the
        # lock (the file may have changed/been reheated concurrently); only
        # flip + persist if it is STILL cold, otherwise just return current
        # content. Double-check keeps the write idempotent under concurrency.
        async with self._vault_lock():
            try:
                current = target.read_text(encoding="utf-8")
            except OSError as e:
                return f"Error: {e}"
            try:
                page = parse_page(current)
            except ValueError as e:
                return f"Error: {e}"
            if page.status != "cold":
                return current
            page.status = "hot"
            page.last_touched = datetime.date.today().isoformat()
            # M2: serialize once so the persisted bytes and the returned
            # bytes are guaranteed identical (no drift if serialization
            # ever becomes nondeterministic).
            out = serialize_page(page)
            try:
                atomic_write_text(target, out)
            except OSError as e:
                return f"Error: {e}"
            return out

    async def _do_append(self, path: str | None, text: str | None) -> str:
        """Append ``text`` to an existing page's body and bump its freshness.

        Whole operation runs under the per-vault async lock (same critical
        section as create) so a concurrent Dream-Lint pass / another turn
        cannot interleave a half-written page. Refuses (writing nothing) for
        a missing page, a non-page / structural basename (``SCHEMA.md`` /
        ``_index.md``), any ``.cold`` path component (cold pages reheat via
        ``read``; Dream owns ``.cold/``), a containment escape, or a
        malformed page (Lint repairs malformed pages — do not overwrite).
        Never touches ``_index.md`` (the stub already exists from create —
        append must not re-scaffold the MOC).
        """
        if not path:
            return "Error: 'path' is required for append"
        # M3: reject missing/empty AND whitespace-only text. Appending
        # ``"\n"`` / ``"   "`` would only inject blank lines into the body.
        if text is None or not text.strip():
            return "Error: 'text' is required for append"

        vault = self._vault()
        target = vault.wiki_dir / path
        resolved = self._resolved_in_vault(target, vault)
        if resolved is None:
            return (
                f"Error: refusing to append {path} "
                "— resolves outside the vault"
            )

        # Non-page / structural basename, or any cold-archive path component.
        # Reference the authoritative vault constants (no hardcoded dup) so
        # this stays in lockstep with iter_pages' filter / the cold marker.
        if resolved.name in _NON_PAGE_NAMES:
            return (
                f"Error: refusing to append {path} "
                "— not a content page (structural/index file)"
            )
        if _COLD_COMPONENT in resolved.parts:
            return (
                f"Error: refusing to append {path} "
                "— cold pages are reheated via read, not appended"
            )

        async with self._vault_lock():
            if not resolved.exists():
                return (
                    f"Error: Page not found: {path}. "
                    "Use operation='create' first."
                )
            try:
                current = resolved.read_text(encoding="utf-8")
            except OSError as e:
                return f"Error: {e}"
            try:
                page = parse_page(current)
            except ValueError:
                return f"Error: cannot append — {path} is malformed"

            # Exactly one separating newline between old body and new text,
            # and a trailing newline so subsequent appends stay clean.
            # Deliberate: pre-existing trailing blank lines in the body are
            # NOT collapsed here — body normalization is Lint's job (Task
            # 4.3); append must not rewrite prior body bytes.
            body = page.body
            if body and not body.endswith("\n"):
                body += "\n"
            body += text
            if not body.endswith("\n"):
                body += "\n"
            page.body = body

            today = datetime.date.today().isoformat()
            page.updated = today
            page.last_touched = today
            # created / type / title / status / tags / links_out / pinned
            # are left UNCHANGED.

            try:
                atomic_write_text(resolved, serialize_page(page))
            except OSError as e:
                return f"Error: {e}"

        return f"Appended to {path}"

    def _do_search(self, query: str | None) -> str:
        """Keyword / tag / recency search over hot AND cold pages.

        Read-only by deliberate design (Task 2.3 boundary): NO lock, NO
        write, NO reheat — only an explicit ``read`` reheats a cold page.
        ``search`` is the agent's discovery mechanism for pages not linked
        from the MOC (design §4): no embeddings, no SQLite — pure
        keyword/substring + recency over ``Vault.iter_pages`` so cold
        knowledge stays findable (the agent then ``read``s a hit, which
        reheats it). Fully deterministic ordering so behaviour is testable
        and Lint/regeneration stays predictable. Never raises — every path
        returns a model-readable string.

        Three modes (mutually exclusive):

        * **empty/whitespace/missing query** → the most-recently-touched
          pages (``last_touched`` desc, relpath asc tiebreak);
        * **``tag:`` prefix** → exact case-insensitive tag filter (a page
          matches iff one of its ``tags`` equals the requested tag,
          case-insensitively), ordered ``last_touched`` desc / relpath asc;
        * **otherwise** → case-insensitive substring search scored by field
          presence (title 3, tag 2, body 1; presence per field, NOT
          occurrence count), score>0 only, sorted score desc /
          ``last_touched`` desc / relpath asc.

        Capped at :data:`_SEARCH_CAP`; an overflow note is appended when
        more matched than were shown.
        """
        vault = self._vault()
        # iter_pages already skips SCHEMA.md/_index.md + malformed pages and,
        # with include_cold=True, walks .cold/ too — search inherits all of
        # that (no extra filtering needed).
        pages = list(vault.iter_pages(include_cold=True))

        q = (query or "").strip()

        def _ordered_recent(
            items: list[tuple[str, Page]],
        ) -> list[tuple[str, Page]]:
            # Deterministic: relpath ASC tiebreak, last_touched DESC primary.
            # Stable sort applied twice (least-significant key first).
            by_rel = sorted(items, key=lambda it: it[0])
            return sorted(
                by_rel, key=lambda it: it[1].last_touched, reverse=True
            )

        header: str
        results: list[tuple[str, Page]]

        if not q:
            results = _ordered_recent(pages)
            header_kind = "most recently touched"
        elif q[:4].lower() == "tag:":
            wanted = q[4:].strip().lower()
            matched = [
                (rel, page)
                for rel, page in pages
                if any(tag.lower() == wanted for tag in page.tags)
            ]
            results = _ordered_recent(matched)
            header_kind = f"tagged {wanted!r}"
        else:
            needle = q.lower()
            scored: list[tuple[int, str, Page]] = []
            for rel, page in pages:
                score = 0
                if needle in page.title.lower():
                    score += 3
                if any(needle in tag.lower() for tag in page.tags):
                    score += 2
                if needle in page.body.lower():
                    score += 1
                if score > 0:
                    scored.append((score, rel, page))
            # Deterministic: score DESC, last_touched DESC, relpath ASC.
            # Apply stable sorts least-significant key first.
            scored.sort(key=lambda t: t[1])
            scored.sort(key=lambda t: t[2].last_touched, reverse=True)
            scored.sort(key=lambda t: t[0], reverse=True)
            results = [(rel, page) for _, rel, page in scored]
            header_kind = f"matching {q!r}"

        if not results:
            return f"No matching pages for {query!r}."

        total = len(results)
        shown = results[:_SEARCH_CAP]
        header = f"Found {len(shown)} page(s) ({header_kind}):"
        lines = [header]
        for rel, page in shown:
            cold = " (cold)" if page.status == "cold" else ""
            lines.append(f"- {rel}{cold} — {_search_snippet(page)}")
        if total > _SEARCH_CAP:
            lines.append(
                f"… ({total - _SEARCH_CAP} more not shown; refine the query)"
            )
        return "\n".join(lines)

    async def _do_create(
        self,
        type: str | None,
        slug: str | None,
        title: str | None,
        body: str,
    ) -> str:
        if not type:
            return "Error: 'type' is required for create"
        if not slug:
            return "Error: 'slug' is required for create"
        if not title:
            return "Error: 'title' is required for create"

        vault = self._vault()
        schema = vault.schema

        # (1) Unknown-type admission gate (friendly). Refuse before building
        # anything so nothing is written for an unrecognised type.
        if not schema.is_known_type(type):
            allowed = ", ".join(sorted(schema.types))
            return f"Error: unknown type '{type}'. Allowed: {allowed}"

        # (1b) Slug sanitization (review I1). The model controls ``slug``.
        # A control char (incl. \n \r \t) would (a) defeat the containment
        # check (it only guards traversal), (b) corrupt ``_index.md`` into a
        # two-physical-line stub that breaks the whole-line idempotency check
        # Lint 4.3 / context 6.1 rely on, and (c) diverge cross-platform
        # (``alice\nbob.md`` is Linux-accepted / Windows-rejected). A
        # control-char slug is never legitimate input: reject it outright
        # rather than silently mangling it into a different page name (the
        # "do NOT silently truncate at a newline" contract). Separators and
        # whitespace are benign and are folded to ``-`` by _safe_slug.
        #
        # R3 widens the control-char reject from ASCII (`_SLUG_CONTROL`,
        # kept as a fast subset path) to the full Unicode Cc/Cf classes so
        # bidi-override / zero-width / BOM chars cannot survive into a
        # filename or MOC wikilink. Benign Unicode letters/digits (accented
        # latin, CJK) are not Cc/Cf and pass through _safe_slug unchanged.
        safe_slug = _safe_slug(slug)
        if (
            _SLUG_CONTROL.search(slug)
            or _has_unicode_control(slug)
            or not safe_slug
        ):
            return (
                f"Error: invalid slug {slug!r} — must contain "
                "filename-safe characters"
            )

        # R1: refuse a slug whose resulting basename collides with a
        # vault-structural / non-page name (``SCHEMA.md``, ``_index.md``),
        # the cold-archive marker, or a hidden/structural (leading
        # ``.``/``_``) name. Such a page is permanently invisible to
        # iter_pages (basename filter, any folder) yet still gets a
        # dangling ``_index.md`` stub — a silently-broken phantom page.
        # Rejected here, BEFORE any filesystem touch, so nothing is
        # written and no stub is appended.
        if _is_reserved_slug(slug, safe_slug):
            return (
                f"Error: reserved/invalid slug {slug!r} — collides with a "
                "structural or hidden vault filename"
            )

        today = datetime.date.today().isoformat()
        page = Page(
            type=type,
            title=title,
            status="hot",
            created=today,
            updated=today,
            last_touched=today,
            tags=[],
            links_out=[],
            pinned=None,
            body=body,
        )

        # (2) Frontmatter admission gate (design §3/P7). Derive ``fm`` from
        # the SAME bytes that get written: serialize the page, parse the
        # frontmatter back, and project its fields. This makes the gate
        # validate the *real* serialized frontmatter instead of a parallel
        # hand-built dict that could silently drift from serialize_page.
        serialized = serialize_page(page)
        parsed_back = parse_page(serialized)
        fm = {
            "type": parsed_back.type,
            "title": parsed_back.title,
            "status": parsed_back.status,
            "created": parsed_back.created,
            "updated": parsed_back.updated,
            "last_touched": parsed_back.last_touched,
            "tags": parsed_back.tags,
            "links_out": parsed_back.links_out,
        }
        fm_errors = schema.validate_frontmatter(fm)
        if fm_errors:
            return "Error: " + "; ".join(fm_errors)

        try:
            target = vault.page_path(type, safe_slug)
        except (KeyError, ValueError) as e:  # pragma: no cover - gated above
            return f"Error: {e}"

        # Enforce vault containment symmetrically with read_page's is_under
        # guard (Task 2.1 security fix — DO NOT remove). Schema.folder(type)
        # is whatever the per-vault SCHEMA.md declares with no single-component
        # validation; a folder of '../...' or an absolute path would let
        # atomic_write_text (parent.mkdir(parents=True)) write OUTSIDE the
        # vault. _resolved_in_vault resolves the destination and only returns
        # it when it stays inside the vault, so the checked path and the
        # written path are identical (no TOCTOU).
        resolved = self._resolved_in_vault(target, vault)
        if resolved is None:
            return (
                f"Error: refusing to write {type}/{slug} "
                "— resolves outside the vault"
            )

        folder = schema.folder(type)
        index_path = vault.wiki_dir / folder / "_index.md"
        resolved_index = self._resolved_in_vault(index_path, vault)
        if resolved_index is None:
            return (
                f"Error: refusing to write the {type} index "
                "— resolves outside the vault"
            )

        # (5) Page-write + index-append are ONE critical section under the
        # per-vault async lock so a concurrent Dream-Lint pass or another
        # turn cannot interleave a half-written page/index (design H2).
        async with self._vault_lock():
            # (3) Duplicate refusal: never overwrite an existing page; the
            # existing body must survive byte-for-byte. Checked inside the
            # lock so two concurrent creates of the same slug can't race.
            if resolved.exists():
                return (
                    f"Page already exists: {type}/{safe_slug}. "
                    "Use operation='append' to add to it."
                )

            try:
                # Write the exact bytes the frontmatter gate validated.
                atomic_write_text(resolved, serialized)
            except OSError as e:
                return f"Error: {e}"

            # (4) Append the wikilink stub to the type's _index.md MOC.
            # APPEND-ONLY: read current content, ensure the stub is present
            # exactly once, and write back all existing bytes unchanged plus
            # the stub if missing. Dream-Lint owns rewrites/reordering.
            stub = f"- [[{folder}/{safe_slug}]]\n"
            try:
                if resolved_index.exists():
                    current = resolved_index.read_text(encoding="utf-8")
                    # Idempotent re-run after a partial failure: if the exact
                    # stub line is already present, leave the index untouched.
                    if stub not in _terminated_line_set(current):
                        new_text = current
                        if new_text and not new_text.endswith("\n"):
                            new_text += "\n"
                        atomic_write_text(resolved_index, new_text + stub)
                else:
                    # M3: header uses ``folder`` (not ``type``) so a schema
                    # where folder != type stays coherent and matches what
                    # Lint 4.3 regenerates ("- [[{folder}/...]]" stubs).
                    header = f"# {folder} index\n\n"
                    atomic_write_text(resolved_index, header + stub)
            except OSError as e:
                # The page is written; surface the index failure honestly so
                # a retry (idempotent above) can repair the MOC.
                return (
                    f"Created page {type}/{safe_slug}.md but failed to update "
                    f"the index: {e}"
                )

        return f"Created page {type}/{safe_slug}.md"
