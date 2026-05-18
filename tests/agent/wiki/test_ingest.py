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

from nanobot.agent.wiki.ingest import IngestReport, run_ingest
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
    provider = _provider("[PAGE people alice]\nshould never run")
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
    provider = _provider("[PAGE people alice]\nAlice leads payments.")
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
        "[PAGE people carol]\nCarol is an engineer.\n"
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
    provider = _provider("[PAGE aliens x]\nThey exist.")
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
    provider = _provider("[PAGE people Bad Slug!!]\nMessy slug person.")
    report = await run_ingest(
        vault, _entries("messy"), provider, "m", render_template
    )
    # deterministic outcome: sanitized to 'bad-slug'
    assert vault.page_path("people", "bad-slug").exists()
    assert "people/bad-slug.md" in report.created


async def test_reserved_slug_skipped(tmp_path):
    vault = _vault(tmp_path)
    provider = _provider("[PAGE people SCHEMA]\nReserved.")
    report = await run_ingest(
        vault, _entries("reserved"), provider, "m", render_template
    )
    assert ("people", "SCHEMA") in report.unknown
    assert report.created == []


async def test_control_char_slug_skipped(tmp_path):
    vault = _vault(tmp_path)
    # tab inside the slug header -> token still parses but sanitizes; an
    # all-control slug sanitizes to empty -> skipped to report.unknown.
    provider = _provider("[PAGE people \x01\x02]\nControl only.")
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
    provider = _provider("[PAGE people dave]\nDave.")
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
        "[PAGE people erin]\nErin is the SRE lead.\n"
        "[APPEND projects/migration]\nMigration kicked off.\n"
        "[PAGE concepts slo]\nSLO = service level objective.\n"
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
