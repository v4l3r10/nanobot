"""Tests for the Ingest phase (Task 4.4).

Ingest turns a batch of conversation-history entries into wiki pages via one
LLM call + a deterministic line-protocol parser. The provider is always
mocked (NO real LLM); the vault is a real :class:`Vault` on ``tmp_path`` with
the bundled SCHEMA fallback and the production parse/serialize path.

The line-protocol parser is a *pure* function of (model_output, vault state)
and is exercised both end-to-end (via :func:`run_ingest`) and indirectly via
the public report so the determinism guarantee is locked.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.tools import wiki_note as _wn
from nanobot.agent.wiki.ingest import (
    _MAX_BODY_CHARS,
    _MAX_DIRECTIVES,
    IngestReport,
    _safe_slug,
    _slug_ok,
    run_ingest,
)
from nanobot.agent.wiki.lint import run_lint
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import Vault
from nanobot.providers.base import LLMResponse
from nanobot.utils.prompt_templates import render_template

TODAY = date.today().isoformat()


def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


def _vault(tmp_path, *, name="u1") -> Vault:
    root = tmp_path / "memory" / "users" / name
    wiki = root / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "SCHEMA.md").write_text(_bundled_schema_text(), encoding="utf-8")
    return Vault(root)


def _provider(output: str) -> MagicMock:
    """A provider whose chat_with_retry yields a canned protocol string."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content=output, finish_reason="stop")
    )
    return provider


def _entries(*contents: str) -> list[dict]:
    return [
        {"cursor": i, "timestamp": "2026-05-18 10:0%d" % i, "content": c}
        for i, c in enumerate(contents)
    ]


def _write_page(vault: Vault, type_: str, slug: str, page: Page) -> Path:
    p = vault.page_path(type_, slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(serialize_page(page), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# empty history -> no provider call
# --------------------------------------------------------------------------- #
async def test_empty_history_skips_provider(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE people/alice]\nshould never run")
    report = await run_ingest(vault, [], provider, "m", render_template)

    provider.chat_with_retry.assert_not_called()
    assert isinstance(report, IngestReport)
    assert report.changed is False
    assert report.created == []
    assert report.appended == []
    assert report.contradictions == []
    assert list(vault.iter_pages(include_cold=True)) == []


# --------------------------------------------------------------------------- #
# PAGE directive + the 4.3-contract end-to-end (ingest -> lint -> indexed)
# --------------------------------------------------------------------------- #
async def test_page_directive_creates_lint_ingestible_page(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE people/alice]\nAlice leads payments.")
    report = await run_ingest(
        vault, _entries("alice runs the payments team"), provider, "m",
        render_template,
    )

    page_path = vault.page_path("people", "alice")
    assert page_path.exists()
    page = parse_page(page_path.read_text(encoding="utf-8"))
    assert page.type == "people"
    assert page.status == "hot"
    assert "Alice leads payments." in page.body
    assert page.created == TODAY
    assert page.updated == TODAY
    assert page.last_touched == TODAY
    assert "people/alice.md" in report.created
    assert report.changed is True

    # 4.3 contract end-to-end: Lint must index the page and NOT cold it.
    run_lint(vault, date.today())
    idx = (vault.wiki_dir / "people" / "_index.md").read_text(encoding="utf-8")
    assert "[[people/alice]]" in idx
    assert not (vault.wiki_dir / ".cold").exists()


# --------------------------------------------------------------------------- #
# APPEND to an existing page
# --------------------------------------------------------------------------- #
async def test_append_grows_body_and_bumps_dates(tmp_path):
    vault = _vault(tmp_path)
    original = Page(
        type="people",
        title="Alice",
        status="hot",
        created="2020-01-01",
        updated="2020-01-01",
        last_touched="2020-01-01",
        tags=["team"],
        links_out=["projects/pay"],
        pinned=True,
        body="Alice is the lead.\n",
    )
    _write_page(vault, "people", "alice", original)

    provider = _provider("[APPEND people/alice]\nAlice now owns billing too.")
    report = await run_ingest(
        vault, _entries("alice took over billing"), provider, "m",
        render_template,
    )

    page = parse_page(vault.page_path("people", "alice").read_text("utf-8"))
    assert "Alice is the lead." in page.body
    assert "Alice now owns billing too." in page.body
    assert page.updated == TODAY
    assert page.last_touched == TODAY
    # other frontmatter unchanged
    assert page.created == "2020-01-01"
    assert page.type == "people"
    assert page.title == "Alice"
    assert page.status == "hot"
    assert page.tags == ["team"]
    assert page.links_out == ["projects/pay"]
    assert page.pinned is True
    assert "people/alice.md" in report.appended


async def test_append_to_missing_page_creates_it(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[APPEND people/bob]\nBob is the new hire.")
    report = await run_ingest(
        vault, _entries("bob joined"), provider, "m", render_template
    )

    page = parse_page(vault.page_path("people", "bob").read_text("utf-8"))
    assert "Bob is the new hire." in page.body
    assert page.status == "hot"
    assert page.created == TODAY
    assert "people/bob.md" in report.created


# --------------------------------------------------------------------------- #
# CONTRADICTION never overwrites prior body
# --------------------------------------------------------------------------- #
async def test_contradiction_preserves_body_adds_section(tmp_path):
    vault = _vault(tmp_path)
    original = Page(
        type="people",
        title="Alice",
        status="hot",
        created="2020-01-01",
        updated="2020-01-01",
        last_touched="2020-01-01",
        tags=[],
        links_out=[],
        pinned=None,
        body="Alice leads payments.\n",
    )
    _write_page(vault, "people", "alice", original)

    provider = _provider(
        "[CONTRADICTION people/alice]\nAlice actually left payments in May."
    )
    report = await run_ingest(
        vault, _entries("alice left payments"), provider, "m", render_template
    )

    page = parse_page(vault.page_path("people", "alice").read_text("utf-8"))
    assert "Alice leads payments." in page.body  # original intact
    assert "## ⚠ contradiction" in page.body
    assert "Alice actually left payments in May." in page.body
    # original body must come BEFORE the contradiction section
    assert page.body.index("Alice leads payments.") < page.body.index(
        "## ⚠ contradiction"
    )
    assert page.updated == TODAY
    assert page.last_touched == TODAY
    assert "people/alice.md" in report.contradictions


async def test_contradiction_missing_target_is_unknown(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[CONTRADICTION people/ghost]\nGhost is wrong.")
    report = await run_ingest(
        vault, _entries("x"), provider, "m", render_template
    )
    assert not vault.page_path("people", "ghost").exists()
    assert ("people", "ghost") in report.unknown
    assert report.changed is False


# --------------------------------------------------------------------------- #
# [SKIP]
# --------------------------------------------------------------------------- #
async def test_skip_writes_nothing(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[SKIP]")
    report = await run_ingest(
        vault, _entries("nothing durable here"), provider, "m", render_template
    )
    assert report.skipped is True
    assert report.changed is False
    assert list(vault.iter_pages(include_cold=True)) == []


# --------------------------------------------------------------------------- #
# garbage / malformed lines
# --------------------------------------------------------------------------- #
async def test_garbage_lines_counted_valid_applied(tmp_path):
    vault = _vault(tmp_path)
    out = (
        "here is some prose the model should not have emitted\n"
        "[GARBAGE not a directive]\n"
        "[PAGE people/carol]\nCarol is an engineer.\n"
        "[PAGE]\n"  # malformed header
        "trailing junk\n"
    )
    provider = _provider(out)
    report = await run_ingest(
        vault, _entries("carol"), provider, "m", render_template
    )
    assert vault.page_path("people", "carol").exists()
    assert "people/carol.md" in report.created
    assert report.malformed_lines > 0


# --------------------------------------------------------------------------- #
# unknown type
# --------------------------------------------------------------------------- #
async def test_unknown_type_not_created(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE aliens/x]\nThey exist.")
    report = await run_ingest(
        vault, _entries("aliens"), provider, "m", render_template
    )
    assert ("aliens", "x") in report.unknown
    assert report.created == []
    assert report.changed is False


# --------------------------------------------------------------------------- #
# slug sanitization
# --------------------------------------------------------------------------- #
async def test_slug_sanitized_to_safe_component(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE people/Bad Slug!!]\nMessy slug person.")
    report = await run_ingest(
        vault, _entries("messy"), provider, "m", render_template
    )
    # M1: case/Unicode are now PRESERVED (parity with wiki_note._safe_slug,
    # which folds whitespace to '-' but does NOT lowercase): 'Bad Slug!!'
    # -> 'Bad-Slug!!' deterministically (same filename the agent tool would
    # produce for the same entity, so dual-write cannot diverge).
    assert vault.page_path("people", "Bad-Slug!!").exists()
    assert "people/Bad-Slug!!.md" in report.created


async def test_reserved_slug_skipped(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE people/SCHEMA]\nReserved.")
    report = await run_ingest(
        vault, _entries("reserved"), provider, "m", render_template
    )
    assert ("people", "SCHEMA") in report.unknown
    assert report.created == []


async def test_control_char_slug_skipped(tmp_path):
    vault = _vault(tmp_path)
    # M1: a raw slug containing an ASCII/Unicode control char is REJECTED
    # outright (parity with wiki_note: never silently mangled into a
    # different page name) -> recorded in report.unknown, no write, no crash.
    provider = _provider("[PAGE people/\x01\x02]\nControl only.")
    report = await run_ingest(
        vault, _entries("ctrl"), provider, "m", render_template
    )
    assert report.created == []
    assert report.unknown  # recorded, no crash


# --------------------------------------------------------------------------- #
# single provider call, no tools
# --------------------------------------------------------------------------- #
async def test_single_provider_call_no_tools(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE people/dave]\nDave.")
    await run_ingest(vault, _entries("dave"), provider, "m", render_template)
    assert provider.chat_with_retry.await_count == 1
    _, kwargs = provider.chat_with_retry.call_args
    assert kwargs.get("tools") is None
    assert kwargs.get("tool_choice") is None
    assert kwargs.get("model") == "m"
    # No AgentRunner / tool loop: chat_with_retry is the ONLY provider
    # interaction (mirrors Dream Phase 1, not Phase 2).
    assert provider.mock_calls and all(
        c[0] in ("chat_with_retry", "") for c in provider.mock_calls
    )
    msgs = kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]


# --------------------------------------------------------------------------- #
# determinism: same canned output + same starting vault -> identical bytes
# --------------------------------------------------------------------------- #
async def test_determinism_identical_bytes(tmp_path):
    out = (
        "[PAGE people/erin]\nErin is the SRE lead.\n"
        "[APPEND projects/migration]\nMigration kicked off.\n"
        "[PAGE concepts/slo]\nSLO = service level objective.\n"
    )
    entries = _entries("erin", "migration", "slo")

    v1 = _vault(tmp_path, name="a")
    v2 = _vault(tmp_path, name="b")
    await run_ingest(v1, entries, _provider(out), "m", render_template)
    await run_ingest(v2, entries, _provider(out), "m", render_template)

    def snapshot(v: Vault) -> dict[str, bytes]:
        snap: dict[str, bytes] = {}
        for p in sorted(v.wiki_dir.rglob("*.md")):
            snap[p.relative_to(v.wiki_dir).as_posix()] = p.read_bytes()
        return snap

    assert snapshot(v1) == snapshot(v2)


# --------------------------------------------------------------------------- #
# provider error -> no writes, no crash
# --------------------------------------------------------------------------- #
async def test_provider_error_no_writes(tmp_path):
    vault = _vault(tmp_path)
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="boom", finish_reason="error")
    )
    report = await run_ingest(
        vault, _entries("x"), provider, "m", render_template
    )
    assert report.changed is False
    assert list(vault.iter_pages(include_cold=True)) == []


# --------------------------------------------------------------------------- #
# C1 — directive count is bounded under the per-vault Dream lock
# --------------------------------------------------------------------------- #
async def test_directive_count_capped(tmp_path):
    vault = _vault(tmp_path)
    n = _MAX_DIRECTIVES + 50
    out = "".join(
        f"[PAGE people/p{i:05d}]\nPerson number {i}.\n" for i in range(n)
    )
    report = await run_ingest(
        vault, _entries("many"), _provider(out), "m", render_template
    )
    pages = [
        rel for rel, _ in vault.iter_pages(include_cold=True)
    ]
    assert len(report.created) == _MAX_DIRECTIVES
    assert len(pages) == _MAX_DIRECTIVES
    assert report.dropped == 50
    # Deterministic: the FIRST _MAX_DIRECTIVES by document order survived;
    # the surplus tail (p00200..) was dropped.
    assert vault.page_path("people", "p00000").exists()
    assert vault.page_path(
        "people", f"p{_MAX_DIRECTIVES - 1:05d}"
    ).exists()
    assert not vault.page_path(
        "people", f"p{_MAX_DIRECTIVES:05d}"
    ).exists()


# --------------------------------------------------------------------------- #
# C1 — body size is clamped (deterministic prefix truncation)
# --------------------------------------------------------------------------- #
async def test_body_size_clamped(tmp_path):
    vault = _vault(tmp_path)
    huge = "X" * (_MAX_BODY_CHARS * 3)
    out = f"[PAGE people/big]\n{huge}"
    await run_ingest(
        vault, _entries("big"), _provider(out), "m", render_template
    )
    page = parse_page(vault.page_path("people", "big").read_text("utf-8"))
    assert len(page.body) <= _MAX_BODY_CHARS
    # Deterministic prefix truncation: the body is exactly the first
    # _MAX_BODY_CHARS chars of the directive body.
    assert page.body == "X" * _MAX_BODY_CHARS


# --------------------------------------------------------------------------- #
# C2 — re-running the SAME output against the SAME vault is idempotent
# --------------------------------------------------------------------------- #
async def test_rerun_same_output_is_idempotent(tmp_path):
    # (a) PAGE-onto-existing fallback path: a 2nd PAGE of the same slug
    # falls back to APPEND; the C2 guard must skip the duplicate append.
    vault = _vault(tmp_path, name="page")
    out = "[PAGE people/alice]\nAlice leads payments."
    await run_ingest(
        vault, _entries("a"), _provider(out), "m", render_template
    )
    b1 = vault.page_path("people", "alice").read_bytes()
    r2 = await run_ingest(
        vault, _entries("a"), _provider(out), "m", render_template
    )
    b2 = vault.page_path("people", "alice").read_bytes()
    r3 = await run_ingest(
        vault, _entries("a"), _provider(out), "m", render_template
    )
    b3 = vault.page_path("people", "alice").read_bytes()
    assert b1 == b2 == b3  # triple delivery -> stable, no growth
    assert r2.skipped_duplicate == 1 and r2.appended == []
    assert r3.skipped_duplicate == 1 and r3.appended == []

    # (b) APPEND-onto-existing: pre-create, then APPEND the same text twice.
    vault = _vault(tmp_path, name="app")
    _write_page(
        vault,
        "people",
        "bob",
        Page(
            type="people", title="Bob", status="hot",
            created="2020-01-01", updated="2020-01-01",
            last_touched="2020-01-01", tags=[], links_out=[],
            pinned=None, body="Bob is here.\n",
        ),
    )
    aout = "[APPEND people/bob]\nBob now owns billing."
    await run_ingest(
        vault, _entries("b"), _provider(aout), "m", render_template
    )
    a1 = vault.page_path("people", "bob").read_bytes()
    ar = await run_ingest(
        vault, _entries("b"), _provider(aout), "m", render_template
    )
    a2 = vault.page_path("people", "bob").read_bytes()
    assert a1 == a2
    assert ar.skipped_duplicate == 1 and ar.appended == []

    # (c) CONTRADICTION-onto-existing: same contradiction text twice.
    # (vault name avoids the Windows reserved device name "con".)
    vault = _vault(tmp_path, name="contra")
    _write_page(
        vault,
        "people",
        "carl",
        Page(
            type="people", title="Carl", status="hot",
            created="2020-01-01", updated="2020-01-01",
            last_touched="2020-01-01", tags=[], links_out=[],
            pinned=None, body="Carl leads infra.\n",
        ),
    )
    cout = "[CONTRADICTION people/carl]\nCarl actually left infra in May."
    await run_ingest(
        vault, _entries("c"), _provider(cout), "m", render_template
    )
    c1 = vault.page_path("people", "carl").read_bytes()
    cr = await run_ingest(
        vault, _entries("c"), _provider(cout), "m", render_template
    )
    c2 = vault.page_path("people", "carl").read_bytes()
    assert c1 == c2
    assert cr.skipped_duplicate == 1 and cr.contradictions == []


# --------------------------------------------------------------------------- #
# I1 — an off-contract response must not crash the Dream cycle
# --------------------------------------------------------------------------- #
async def test_offcontract_response_no_crash(tmp_path):
    # (a) provider returns a bare str (no .content attribute at all).
    vault = _vault(tmp_path, name="bare")
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value="just a string")
    report = await run_ingest(
        vault, _entries("x"), provider, "m", render_template
    )
    assert report.changed is False
    assert list(vault.iter_pages(include_cold=True)) == []

    # (b) an object with content=None and finish_reason="error".
    vault = _vault(tmp_path, name="none")
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content=None, finish_reason="error")
    )
    report = await run_ingest(
        vault, _entries("x"), provider, "m", render_template
    )
    assert report.changed is False
    assert list(vault.iter_pages(include_cold=True)) == []

    # (c) an object with content=None but finish_reason="stop" (not an
    # error) — still must not crash and must write nothing.
    vault = _vault(tmp_path, name="nonestop")
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content=None, finish_reason="stop")
    )
    report = await run_ingest(
        vault, _entries("x"), provider, "m", render_template
    )
    assert report.changed is False
    assert list(vault.iter_pages(include_cold=True)) == []


# --------------------------------------------------------------------------- #
# I2 — a derived title is clamped to a single bounded line
# --------------------------------------------------------------------------- #
async def test_title_clamped(tmp_path):
    vault = _vault(tmp_path)
    first_line = "T" * 5000
    out = f"[PAGE people/long]\n{first_line}\nrest of body"
    await run_ingest(
        vault, _entries("t"), _provider(out), "m", render_template
    )
    page = parse_page(vault.page_path("people", "long").read_text("utf-8"))
    assert len(page.title) <= 120
    assert "\n" not in page.title
    assert page.title == "T" * 120


# --------------------------------------------------------------------------- #
# M1 — Ingest's slug+accept/reject policy is pinned to wiki_note's
# --------------------------------------------------------------------------- #
def test_slug_policy_parity_with_wiki_note():
    """Ingest's _safe_slug + accept/reject decision MUST equal wiki_note's
    _safe_slug + its _do_create reject gate for the SAME inputs.

    This pins the two independent policy copies together so the dual-write
    duplicate-page bug (Ingest ASCII-folded while wiki_note preserved
    Unicode/case) can never regress: a future drift fails HERE.
    """
    samples = [
        "Café",
        "日本語",
        "A B",
        "Über",
        "payment-svc",
        "SCHEMA",
        "  spaced  ",
        "\x01\x02bad\x7f",
        ".cold",
        "_index",
        "Bad Slug!!",
        "",
    ]

    def _wn_accepts(raw: str) -> tuple[str, bool]:
        s = _wn._safe_slug(raw)
        rejected = bool(
            _wn._SLUG_CONTROL.search(raw)
            or _wn._has_unicode_control(raw)
            or not s
            or _wn._is_reserved_slug(raw, s)
        )
        return s, not rejected

    for raw in samples:
        ig_slug = _safe_slug(raw)
        ig_accept = _slug_ok(raw, ig_slug)
        wn_slug, wn_accept = _wn_accepts(raw)
        assert ig_slug == wn_slug, (
            f"slug divergence for {raw!r}: ingest={ig_slug!r} "
            f"wiki_note={wn_slug!r}"
        )
        assert ig_accept == wn_accept, (
            f"accept/reject divergence for {raw!r}: ingest={ig_accept} "
            f"wiki_note={wn_accept}"
        )


# --------------------------------------------------------------------------- #
# Ingest linking (increment 2.5): the prompt instructs body wikilinks, and an
# ingest-written [[folder/slug]] is reconciled into links_out in one pass.
# --------------------------------------------------------------------------- #
def test_template_instructs_body_linking():
    rendered = render_template(
        "agent/wiki_ingest.md", strip=True, allowed_types="people, projects, concepts"
    )
    assert "[[" in rendered  # teaches linking existing pages in the body


async def test_ingest_body_link_reconciled_into_links_out(tmp_path):
    vault = _vault(tmp_path)
    # An existing page the ingest body will link to.
    _write_page(
        vault, "people", "alice",
        Page(
            type="people", title="Alice", status="hot",
            created="2020-01-01", updated="2020-01-01", last_touched="2020-01-01",
            body="Alice leads payments.\n",
        ),
    )
    # First non-empty body line ("Bob") becomes the title; the link sits in the
    # body below it.
    provider = _provider("[PAGE people/bob]\nBob\nWorks with [[people/alice]].")
    await run_ingest(
        vault, _entries("bob works with alice"), provider, "m", render_template
    )
    # Reconcile (increment 1) runs in the same Dream pass right after ingest.
    run_lint(vault, date.today())

    bob = parse_page(vault.page_path("people", "bob").read_text(encoding="utf-8"))
    assert bob.links_out == ["people/alice"]
