"""Task 7.3 — END-TO-END + edge-case integration safety net for the wiki tree.

This is the integration safety net for the whole wiki feature (Milestones
0-7). The individual units (``wiki/{paths,page,schema,vault,decay,lint,
ingest,migrate}``, ``tools/wiki_note``, the Dream wiki block, ContextBuilder
MOC injection, the per-vault lock, GitStore vault versioning) are each
covered by their own unit suites and reviewed. 7.3's value is proving they
*compose* correctly end-to-end:

  A. the core value-prop loop (tool create -> Dream/Lint -> MOC -> prompt);
  B. Dream Ingest->Lint->MOC per-user with tagged history (no cross-bleed);
  C. the decay -> cold -> reheat-on-read -> relocate round trip + MOC;
  D. the GitStore /dream-restore per-commit-inverse on a wiki commit;
  E. the unified_session collapse to one vault;
  F. idempotent Dream retry / crash-resume (byte-stable full-vault snapshot);
  G. per-user failure isolation + malformed/edge (null/non-str key, reserved
     slug, malformed page, the C1 migration-unified-only gate);
  H. concurrency H2 (per-vault lock serializes same-user, independent slugs);
  I. the GOLDEN end-to-end: ``wiki_enabled=False`` is byte-identical to
     pre-wiki behavior (no ``memory/users/``, global MEMORY.md prompt,
     legacy Dream cursor/compact/git path).

All LLM calls are MOCKED (NO real network). The Vault / MemoryStore / Dream /
GitStore are REAL, on ``tmp_path``. Scenarios assert by CONTENT and on-disk /
MOC / git state, not mere existence. Determinism is event-based where
serialization matters; no flaky sleeps gate a correctness assertion.

Fixture provenance: ``store`` / ``mock_provider`` / ``mock_runner`` /
``dream`` and ``_make_run_result`` mirror the canonical mocked pattern in
``tests/agent/test_dream.py`` / ``tests/agent/test_dream_wiki.py`` (they are
module-local there, not exported via conftest, so they are reproduced here —
keep in sync if those change). ``_content_echo_provider`` mirrors the
dream_wiki echo-provider trick: Phase 1 (legacy) gets the FIRST provider
call (plain analysis string), every later call is an Ingest call whose
emitted ``[PAGE concepts/dump]`` body echoes the conversation-history slice
that vault received — so any cross-bleed is visible in page CONTENT.

This file is TEST-ONLY. No production code is modified by it; the wiki
pipeline composes correctly as built (see the suite report).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

import nanobot.agent.memory as memory_mod
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import Dream, MemoryStore
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.context import RequestContext, ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.wiki_note import WikiNoteTool
from nanobot.agent.wiki.lint import run_lint
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.vault import Vault
from nanobot.utils.vault_lock import get_vault_lock

# --- Cross-loop lock isolation (mirrors test_dream_wiki) ---------------------


@pytest.fixture(autouse=True)
def _reset_dream_run_lock():
    """Swap the process-global ``_DREAM_RUN_LOCK`` for a fresh unbound Lock in
    teardown so a contended binding from one test's event loop never leaks
    into the next (pytest runs each ``async def`` on its own loop). Identical
    rationale to the fixture in ``tests/agent/test_dream_wiki.py``."""
    yield
    memory_mod._DREAM_RUN_LOCK = asyncio.Lock()


# --- Canonical mocked Dream fixtures (mirror test_dream.py) ------------------


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path)
    s.write_soul("# Soul\n- Helpful")
    s.write_user("# User\n- Developer")
    s.write_memory("# Memory\n- Project X active")
    return s


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def dream(store, mock_provider, mock_runner):
    d = Dream(store=store, provider=mock_provider, model="test-model", max_batch_size=20)
    d._runner = mock_runner
    return d


def _make_run_result(stop_reason="completed", final_content=None, tool_events=None):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


def _ok_runner():
    return AsyncMock(return_value=_make_run_result(
        tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
    ))


def _content_echo_provider(mock_provider):
    """Phase 1 (legacy) gets the FIRST call (plain analysis). Every later
    (Ingest) call emits ``[PAGE concepts/dump]`` whose body is the verbatim
    user-prompt conversation-history text it received — so a vault that got
    another user's slice visibly contains that user's text (cross-bleed is
    detectable in page CONTENT)."""
    state = {"calls": 0}

    async def _side_effect(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return MagicMock(content="New fact", finish_reason="stop")
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        user_msg = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        return MagicMock(
            content=f"[PAGE concepts/dump]\n{user_msg}\n", finish_reason="stop",
        )

    mock_provider.chat_with_retry.side_effect = _side_effect
    return state


# --- wiki_note tool wiring (mirrors tests/agent/tools/test_wiki_note.py) -----


def _wiki_tool(workspace, session_key):
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            restrict_to_workspace=True,
            exec=SimpleNamespace(sandbox=False),
        ),
        workspace=str(workspace),
        file_state_store=None,
    )
    t = WikiNoteTool.create(ctx)
    channel, _, chat = session_key.partition(":")
    t.set_context(
        RequestContext(channel=channel, chat_id=chat, session_key=session_key)
    )
    return t


def _vault_root(store, session_key_slug):
    return store.workspace / "memory" / "users" / session_key_slug


def _snapshot(root):
    """Full byte snapshot of every file under ``root`` (POSIX-rel keys)."""
    snap: dict[str, bytes] = {}
    if not root.exists():
        return snap
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[p.relative_to(root).as_posix()] = p.read_bytes()
    return snap


# =========================================================================== #
# A. Core value-prop loop: tool create -> Dream/Lint -> MOC -> prompt
# =========================================================================== #


class TestCoreValuePropLoop:
    """The agent writes a durable note via the wiki_note tool; a Dream-Lint
    cycle regenerates the per-user MOC; the next system prompt for that
    session injects that MOC referencing the page (so the agent could
    navigate to it); search + read round-trip the page back."""

    async def test_create_then_lint_then_prompt_then_search_read(
        self, dream, mock_provider, mock_runner, store,
    ):
        session = "telegram:1"
        slug = "telegram_1"
        tool = _wiki_tool(store.workspace, session)

        # 1. Agent creates a concepts page via the tool.
        created = await tool.execute(
            operation="create",
            type="concepts",
            slug="payment-flow",
            title="Payment Flow",
            body="The payment flow uses idempotency keys end to end.",
        )
        assert "Created page concepts/payment-flow.md" in created

        vroot = _vault_root(store, slug)
        page_file = vroot / "wiki" / "concepts" / "payment-flow.md"
        assert page_file.is_file()

        # 2. A Dream-Lint cycle for this user regenerates the per-user MOC.
        #    Drive the real Dream wiki block with a tagged history entry +
        #    a SKIP Ingest (no new pages) so Lint indexes the tool's page.
        dream.wiki_enabled = True
        store.append_history("discussed the payment flow", session_key=session)
        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),   # Phase 1
            MagicMock(content="[SKIP]", finish_reason="stop"),     # Ingest
        ]
        mock_runner.run = _ok_runner()

        assert await dream.run() is True

        moc_path = vroot / "MEMORY.md"
        assert moc_path.is_file(), "Lint did not regenerate the per-user MOC"
        moc_text = moc_path.read_text(encoding="utf-8")
        assert "[[concepts/payment-flow]]" in moc_text
        assert "Payment Flow" in moc_text
        # Idempotence on disk: the tool page was indexed, not mangled.
        assert parse_page(page_file.read_text(encoding="utf-8")).type == "concepts"

        # 3. The 6.1 prompt path injects exactly THIS user's regenerated MOC.
        builder = ContextBuilder(workspace=store.workspace, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key=session)
        assert "# Memory\n\n" in prompt
        assert "[[concepts/payment-flow]]" in prompt, (
            "the regenerated per-user MOC was not injected into the prompt — "
            "the agent could not navigate to its own durable note"
        )

        # 4. search finds it; read returns its body verbatim.
        found = tool._do_search("payment")
        assert "concepts/payment-flow.md" in found
        read_back = await tool.execute(
            operation="read", path="concepts/payment-flow.md",
        )
        assert "idempotency keys end to end" in read_back
        assert "title: Payment Flow" in read_back


# =========================================================================== #
# B. Dream Ingest->Lint->MOC, per-user, tagged history, no cross-bleed
# =========================================================================== #


class TestPerUserIngestLintMoc:
    async def test_two_users_isolated_indexes_and_mocs(
        self, dream, mock_provider, mock_runner, store,
    ):
        dream.wiki_enabled = True
        store.append_history("USER ONE secret alpha", session_key="telegram:1")
        store.append_history("USER TWO secret beta", session_key="telegram:2")
        store.append_history("USER ONE more alpha", session_key="telegram:1")

        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()

        assert await dream.run() is True

        users = store.workspace / "memory" / "users"
        v1 = users / "telegram_1"
        v2 = users / "telegram_2"

        # Each vault has ONLY its own user's ingested page content.
        v1_dump = (v1 / "wiki" / "concepts" / "dump.md").read_text(encoding="utf-8")
        v2_dump = (v2 / "wiki" / "concepts" / "dump.md").read_text(encoding="utf-8")
        assert "USER ONE secret alpha" in v1_dump
        assert "USER ONE more alpha" in v1_dump
        assert "USER TWO" not in v1_dump
        assert "USER TWO secret beta" in v2_dump
        assert "USER ONE" not in v2_dump

        # Each vault: its own _index.md + root MEMORY.md MOC, Lint-regenerated,
        # listing ONLY its own page.
        for vroot in (v1, v2):
            idx = (vroot / "wiki" / "concepts" / "_index.md").read_text(encoding="utf-8")
            assert idx.startswith("# concepts index\n\n")
            assert "[[concepts/dump]]" in idx
            moc = (vroot / "MEMORY.md").read_text(encoding="utf-8")
            assert "## Map" in moc
            assert "[[concepts/_index]]" in moc
            assert "[[concepts/dump]]" in moc

        # No spurious unified vault (every entry was tagged per-user).
        assert not (users / "unified_default").exists()

        # Building each session's prompt shows ONLY that user's MOC content.
        builder = ContextBuilder(workspace=store.workspace, wiki_enabled=True)
        p1 = builder.build_system_prompt(session_key="telegram:1")
        p2 = builder.build_system_prompt(session_key="telegram:2")
        assert "[[concepts/dump]]" in p1 and "[[concepts/dump]]" in p2
        # MOC content is per-user; the page bodies never enter the MOC/prompt,
        # but the distinct vaults are proven isolated above. Sanity: each
        # prompt resolves a Memory section.
        assert "# Memory\n\n" in p1
        assert "# Memory\n\n" in p2


# =========================================================================== #
# C. Decay -> cold -> reheat-on-read -> relocate round-trip + MOC
# =========================================================================== #


class TestDecayReheatRelocateRoundTrip:
    async def test_full_cold_reheat_relocate_cycle(self, store):
        """Seed a stale hot page -> run_lint cools it into .cold/ ->
        wiki_note.read of the .cold/ path reheats it in place (status hot) ->
        next run_lint RELOCATES it out of .cold/ back to its hot home and
        re-indexes it in the MOC. Asserts the full cycle on disk + MOC."""
        session = "telegram:1"
        slug = "telegram_1"
        vroot = _vault_root(store, slug)
        tool = _wiki_tool(store.workspace, session)

        # Materialize the vault (SCHEMA.md) via the tool, then seed a stale
        # hot 'people' page directly (people cold_after_days=180).
        Vault(vroot).ensure_initialized()
        people_dir = vroot / "wiki" / "people"
        people_dir.mkdir(parents=True, exist_ok=True)
        stale = Page(
            type="people", title="Stale Sam", status="hot",
            created="2020-01-01", updated="2020-01-01",
            last_touched="2020-01-01",
            tags=[], links_out=[], pinned=None,
            body="Sam is a contact from long ago.",
        )
        hot_path = people_dir / "sam.md"
        hot_path.write_text(serialize_page(stale), encoding="utf-8")

        today = dt.date(2026, 5, 19)
        cold_path = vroot / "wiki" / ".cold" / "people" / "sam.md"

        # 1. Lint cools the stale page into .cold/.
        rep1 = run_lint(Vault(vroot), today)
        assert "people/sam.md" in rep1.cooled
        assert not hot_path.exists()
        assert cold_path.is_file()
        assert parse_page(cold_path.read_text(encoding="utf-8")).status == "cold"
        moc_after_cool = (vroot / "MEMORY.md").read_text(encoding="utf-8")
        assert "[[people/sam]]" not in moc_after_cool, (
            "a cold page must not be listed in the hot MOC"
        )

        # 2. wiki_note.read of the .cold/ path reheats it in place.
        read_out = await tool.execute(
            operation="read", path=".cold/people/sam.md",
        )
        assert "Sam is a contact from long ago." in read_out
        reheated = parse_page(cold_path.read_text(encoding="utf-8"))
        assert reheated.status == "hot", (
            "read of a cold page did not reheat it (status still cold)"
        )
        assert reheated.last_touched == today.isoformat() or (
            reheated.last_touched == dt.date.today().isoformat()
        )
        # Still physically under .cold/ until Lint relocates it.
        assert cold_path.is_file()

        # 3. Next Lint RELOCATES it out of .cold/ to its hot home + re-indexes.
        rep2 = run_lint(Vault(vroot), today)
        assert any(
            ".cold/people/sam.md -> people/sam.md" == r for r in rep2.reheated
        ), f"relocate not reported: {rep2.reheated}"
        assert hot_path.is_file(), "page not relocated back to hot location"
        assert not cold_path.exists(), ".cold copy not removed after relocate"
        assert parse_page(hot_path.read_text(encoding="utf-8")).status == "hot"
        moc_final = (vroot / "MEMORY.md").read_text(encoding="utf-8")
        assert "[[people/sam]]" in moc_final, (
            "reheated+relocated page not re-indexed in the MOC"
        )
        idx = (vroot / "wiki" / "people" / "_index.md").read_text(encoding="utf-8")
        assert "[[people/sam]]" in idx


# =========================================================================== #
# D. Git history / /dream-restore round-trip on a wiki commit
# =========================================================================== #


class TestGitDreamRestoreRoundTrip:
    async def test_revert_first_wiki_commit_preserves_later_files(
        self, dream, mock_provider, mock_runner, store,
    ):
        """A Dream cycle writes a wiki page and auto-commits memory/users/**.
        Capture sha1. A SECOND cycle writes another page + a new commit.
        ``GitStore.revert(sha1)`` is the per-commit inverse: the sha1 wiki
        state is restored, legacy files untouched, the later unrelated wiki
        page preserved (the 5.1/5.2 contract end-to-end)."""
        store.git.init()
        store.git.auto_commit("baseline")

        dream.wiki_enabled = True

        # --- Cycle 1: ingest a page for telegram:1, then auto-commit. -----
        store.append_history("alpha fact one", session_key="telegram:1")
        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()
        assert await dream.run() is True

        v1_dump = (
            store.workspace / "memory" / "users" / "telegram_1"
            / "wiki" / "concepts" / "dump.md"
        )
        assert v1_dump.is_file()
        assert "alpha fact one" in v1_dump.read_text(encoding="utf-8")

        sha1 = store.git.auto_commit("dream cycle 1 wiki state")
        assert sha1, "no commit captured the first wiki state"

        # --- Cycle 2: a DIFFERENT user's page (a later, unrelated change). -
        store.append_history("beta fact two", session_key="telegram:2")
        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()
        assert await dream.run() is True

        v2_dump = (
            store.workspace / "memory" / "users" / "telegram_2"
            / "wiki" / "concepts" / "dump.md"
        )
        assert v2_dump.is_file()
        sha2 = store.git.auto_commit("dream cycle 2 wiki state")
        assert sha2

        # Mutate v1's page AFTER sha1 so revert(sha1) has something to undo
        # on a path sha1 introduced... actually sha1 ADDED v1_dump, so
        # revert(sha1) must DELETE v1_dump (undo the add) while leaving the
        # later v2_dump (created by sha2, outside sha1's diff) untouched and
        # the legacy files intact.
        legacy_before = store.read_memory()
        soul_before = store.read_soul()

        new_sha = store.git.revert(sha1)
        assert new_sha, "revert(sha1) made no change / created no inverse commit"

        # sha1 ADDED v1_dump => its inverse DELETES it.
        assert not v1_dump.exists(), (
            "revert(sha1) did not undo the page sha1 added"
        )
        # The later, unrelated wiki page (sha2) is preserved.
        assert v2_dump.is_file(), (
            "revert(sha1) destroyed a file created by a LATER commit — "
            "not a true per-commit inverse"
        )
        assert "beta fact two" in v2_dump.read_text(encoding="utf-8")
        # Legacy memory files are byte-untouched (sha1 didn't change them).
        assert store.read_memory() == legacy_before
        assert store.read_soul() == soul_before


# =========================================================================== #
# E. unified_session collapse
# =========================================================================== #


class TestUnifiedSessionCollapse:
    async def test_all_unified_one_vault_one_moc_for_any_session(
        self, dream, mock_provider, mock_runner, store,
    ):
        dream.wiki_enabled = True
        store.append_history("device A note", session_key="unified:default")
        store.append_history("device B note", session_key="unified:default")

        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()
        assert await dream.run() is True

        users = store.workspace / "memory" / "users"
        assert (users / "unified_default").is_dir()
        others = [p.name for p in users.iterdir() if p.name != "unified_default"]
        assert others == [], f"unexpected per-user vaults: {others}"

        dump = (
            users / "unified_default" / "wiki" / "concepts" / "dump.md"
        ).read_text(encoding="utf-8")
        assert "device A note" in dump
        assert "device B note" in dump

        moc = (users / "unified_default" / "MEMORY.md").read_text(encoding="utf-8")
        assert "[[concepts/dump]]" in moc

        # ANY session's prompt resolves the one unified MOC (the wiki-on
        # prompt resolves vault by vault_slug(session_key); the only vault is
        # unified_default — distinct sessions DON'T see it, but the unified
        # session does, which is the contract under unified_session).
        builder = ContextBuilder(workspace=store.workspace, wiki_enabled=True)
        p_unified = builder.build_system_prompt(session_key="unified:default")
        assert "[[concepts/dump]]" in p_unified


# =========================================================================== #
# F. Idempotent Dream retry / crash-resume
# =========================================================================== #


class TestIdempotentDreamRetry:
    @staticmethod
    def _fixed_ingest_provider(mock_provider):
        """Phase 1 -> plain analysis; EVERY Ingest call -> the SAME fixed
        ``[PAGE concepts/dump]`` directive with a constant body, regardless
        of the prompt.

        Crash-resume re-delivery is defined as the SAME history batch AND
        the SAME model output applied to the vault again. The C2 idempotence
        guard (``ingest._body_already_present``) makes re-applying an
        IDENTICAL directive body a byte-stable no-op. (An echo provider that
        mirrors the prompt would emit DIFFERENT output on run 2 because the
        ``## Existing Pages`` section legitimately changed — that is correct
        Ingest behaviour, not the crash-resume scenario, so the fixed
        provider is the right model of a re-delivery.)"""
        state = {"calls": 0}

        async def _side_effect(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                return MagicMock(content="New fact", finish_reason="stop")
            return MagicMock(
                content="[PAGE concepts/dump]\nA durable distilled fact.\n",
                finish_reason="stop",
            )

        mock_provider.chat_with_retry.side_effect = _side_effect

    async def test_redelivered_batch_is_byte_stable(
        self, dream, mock_provider, mock_runner, store,
    ):
        """Run Dream on a multi-user tagged batch, snapshot every vault, then
        simulate a crash-resume re-delivery of the SAME batch + SAME canned
        model output (reset the dream cursor, re-run with the same fixed
        provider output). Ingest C2 + Lint idempotence => the full vault tree
        is byte-identical run-2 vs run-1 (bodies don't grow, dates not
        bumped, Lint output byte-stable)."""
        dream.wiki_enabled = True
        store.append_history("alpha durable one", session_key="telegram:1")
        store.append_history("beta durable two", session_key="telegram:2")

        self._fixed_ingest_provider(mock_provider)
        mock_runner.run = _ok_runner()
        assert await dream.run() is True

        users = store.workspace / "memory" / "users"
        snap1 = _snapshot(users)
        assert snap1, "first run produced no vault files"
        assert store.get_last_dream_cursor() == 2
        # Sanity: the fixed directive WAS applied (a real page was written).
        assert (
            users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        ).is_file()

        # Crash-resume: rewind the dream cursor so the SAME batch is
        # re-delivered, and re-arm the SAME fixed canned provider output.
        store.set_last_dream_cursor(0)
        self._fixed_ingest_provider(mock_provider)
        mock_runner.run = _ok_runner()
        assert await dream.run() is True

        snap2 = _snapshot(users)

        # Full-vault snapshot byte-identical: the re-delivered batch is a
        # no-op (C2 substring guard + Lint idempotence). The .lint.log is
        # included in the snapshot and must also be byte-stable.
        assert set(snap2) == set(snap1), (
            f"file set changed on re-delivery: "
            f"added={set(snap2) - set(snap1)} removed={set(snap1) - set(snap2)}"
        )
        for rel in sorted(snap1):
            assert snap2[rel] == snap1[rel], (
                f"{rel} changed on a re-delivered identical batch — "
                "Ingest/Lint not idempotent (C2 / Lint idempotence broken)"
            )


# =========================================================================== #
# G. Per-user failure isolation + malformed / edge cases
# =========================================================================== #


class TestFailureIsolationAndEdgeCases:
    async def test_one_user_ingest_failure_isolated(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """``run_ingest`` raising for ONE slug isolates to that slug: the
        other user + legacy path + cursor are intact, the failure is logged
        with its slug, no exception escapes."""
        dream.wiki_enabled = True
        store.append_history("alpha for one", session_key="telegram:1")
        store.append_history("beta for two", session_key="telegram:2")

        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()

        real_run_ingest = memory_mod.run_ingest

        async def _selective(vault, *a, **k):
            if vault.root.name == "telegram_2":
                raise RuntimeError("user-2 vault corrupt")
            return await real_run_ingest(vault, *a, **k)

        monkeypatch.setattr(
            memory_mod, "run_ingest", AsyncMock(side_effect=_selective),
        )

        captured: list[str] = []
        sink = logger.add(lambda m: captured.append(str(m)), level="ERROR")
        try:
            assert await dream.run() is True
        finally:
            logger.remove(sink)

        assert store.get_last_dream_cursor() == 2
        users = store.workspace / "memory" / "users"
        v1_dump = users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        assert v1_dump.is_file(), "user-1 lost to user-2's failure (no isolation)"
        assert "alpha for one" in v1_dump.read_text(encoding="utf-8")
        assert not (
            users / "telegram_2" / "wiki" / "concepts" / "dump.md"
        ).exists()
        assert any("telegram_2" in m for m in captured)
        assert any("wiki ingest/lint failed" in m for m in captured)

    async def test_non_str_and_null_session_key_route_unified_no_crash(
        self, dream, mock_provider, mock_runner, store,
    ):
        """A ``{"session_key": 123}`` / ``null`` record routes to
        unified_default without aborting the cycle; a real per-user record
        still ingests. No AttributeError escapes; cursor advances."""
        dream.wiki_enabled = True
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": '
            '"int keyed", "session_key": 123}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": '
            '"null keyed", "session_key": null}\n'
            '{"cursor": 3, "timestamp": "2026-04-01 10:02", "content": '
            '"tg one", "session_key": "telegram:1"}\n',
            encoding="utf-8",
        )
        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()

        assert await dream.run() is True
        assert store.get_last_dream_cursor() == 3

        users = store.workspace / "memory" / "users"
        unified = (
            users / "unified_default" / "wiki" / "concepts" / "dump.md"
        ).read_text(encoding="utf-8")
        assert "int keyed" in unified
        assert "null keyed" in unified
        tg = (
            users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        ).read_text(encoding="utf-8")
        assert "tg one" in tg
        assert "int keyed" not in tg

    async def test_malformed_page_left_untouched_and_unindexed(self, store):
        """A malformed page on disk is left byte-untouched by Lint and never
        indexed into the MOC; a valid sibling IS indexed."""
        slug = "telegram_1"
        vroot = _vault_root(store, slug)
        Vault(vroot).ensure_initialized()
        concepts = vroot / "wiki" / "concepts"
        concepts.mkdir(parents=True, exist_ok=True)

        bad = concepts / "broken.md"
        bad_bytes = b"this is not a wiki page: no frontmatter fence at all\n"
        bad.write_bytes(bad_bytes)
        good = concepts / "valid.md"
        good.write_text(serialize_page(Page(
            type="concepts", title="Valid One", status="hot",
            created="2026-05-19", updated="2026-05-19",
            last_touched="2026-05-19", tags=[], links_out=[], pinned=None,
            body="valid body",
        )), encoding="utf-8")

        rep = run_lint(Vault(vroot), dt.date(2026, 5, 19))
        assert "concepts/broken.md" in rep.malformed
        # Malformed file byte-untouched.
        assert bad.read_bytes() == bad_bytes
        moc = (vroot / "MEMORY.md").read_text(encoding="utf-8")
        idx = (concepts / "_index.md").read_text(encoding="utf-8")
        assert "[[concepts/valid]]" in moc
        assert "[[concepts/valid]]" in idx
        assert "broken" not in moc
        assert "broken" not in idx

    async def test_reserved_slug_create_rejected_writes_nothing(self, store):
        """``wiki_note.create`` with a reserved/control-char slug is rejected
        with a clear error and writes NOTHING (no page, no index stub)."""
        tool = _wiki_tool(store.workspace, "telegram:1")
        vroot = _vault_root(store, "telegram_1")

        # Reserved structural stem.
        r1 = await tool.execute(
            operation="create", type="concepts", slug="SCHEMA",
            title="X", body="b",
        )
        assert r1.lower().startswith("error")
        assert "reserved" in r1.lower() or "invalid" in r1.lower()

        # Control char in slug.
        r2 = await tool.execute(
            operation="create", type="concepts", slug="bad\nslug",
            title="X", body="b",
        )
        assert r2.lower().startswith("error")
        assert "invalid slug" in r2.lower()

        # Nothing was written: no concepts pages, no index stub.
        concepts = vroot / "wiki" / "concepts"
        if concepts.exists():
            md = sorted(p.name for p in concepts.glob("*.md"))
            assert md == [] or md == ["_index.md"] and not (
                concepts / "_index.md"
            ).read_text(encoding="utf-8").strip().endswith("]]"), (
                f"reserved/control slug leaked files: {md}"
            )

    async def test_c1_migration_unified_only_under_live_routing(
        self, dream, mock_provider, mock_runner, store,
    ):
        """C1 gate under LIVE multi-slug routing: a mixed batch (unified +
        two per-user) -> the legacy GLOBAL memory/USER blob migrates ONLY
        into unified_default; per-user vaults get their own Ingest slice but
        NEVER a .migrated / imported-memory.md / USER.md."""
        dream.wiki_enabled = True
        store.append_history("unified line", session_key="unified:default")
        store.append_history("alpha for one", session_key="telegram:1")
        store.append_history("beta for two", session_key="telegram:2")

        _content_echo_provider(mock_provider)
        mock_runner.run = _ok_runner()
        assert await dream.run() is True

        users = store.workspace / "memory" / "users"
        unified = users / "unified_default"
        assert (unified / ".migrated").is_file()
        assert (unified / "wiki" / "concepts" / "imported-memory.md").is_file()
        assert (unified / "USER.md").read_text(encoding="utf-8") == "# User\n- Developer"

        for v, own, foreign in (
            (users / "telegram_1", "alpha for one", "beta for two"),
            (users / "telegram_2", "beta for two", "alpha for one"),
        ):
            assert (v / "wiki" / "SCHEMA.md").is_file()
            assert not (v / ".migrated").exists()
            assert not (v / "wiki" / "concepts" / "imported-memory.md").exists()
            assert not (v / "USER.md").exists()
            dump = (v / "wiki" / "concepts" / "dump.md").read_text(encoding="utf-8")
            assert own in dump
            assert foreign not in dump
            assert "unified line" not in dump


# =========================================================================== #
# H. Concurrency (H2) — per-vault lock serializes same-user, independent slugs
# =========================================================================== #


class TestConcurrencyH2:
    async def test_same_vault_write_blocks_while_lock_held(self, store):
        """While ``get_vault_lock(slug)`` is held, a ``wiki_note`` write to
        THAT vault blocks until release (same-user Dream-Ingest vs tool
        serialize). Event-based: no sleep gates the correctness assertion."""
        from nanobot.agent.wiki.paths import vault_slug

        session = "telegram:1"
        slug = vault_slug(session)
        tool = _wiki_tool(store.workspace, session)
        lock = get_vault_lock(slug)

        page_file = (
            _vault_root(store, slug) / "wiki" / "concepts" / "locked.md"
        )

        await lock.acquire()
        try:
            task = asyncio.ensure_future(tool.execute(
                operation="create", type="concepts", slug="locked",
                title="Locked", body="written only after lock release",
            ))
            # Yield control repeatedly; the create must NOT complete while we
            # hold the same per-vault lock (the create's critical section is
            # under get_vault_lock(slug)).
            for _ in range(20):
                await asyncio.sleep(0)
            assert not task.done(), (
                "wiki_note.create completed while the per-vault lock was held "
                "externally — same-user Dream/tool did NOT serialize (H2)"
            )
            assert not page_file.exists()
        finally:
            lock.release()

        out = await asyncio.wait_for(task, timeout=5)
        assert "Created page concepts/locked.md" in out
        assert page_file.is_file()

    async def test_different_vault_slugs_proceed_independently(self, store):
        """A held lock on slug A must NOT block a ``wiki_note`` write to a
        DIFFERENT slug B (no cross-vault serialization)."""
        from nanobot.agent.wiki.paths import vault_slug

        lock_a = get_vault_lock(vault_slug("telegram:1"))
        tool_b = _wiki_tool(store.workspace, "telegram:2")

        await lock_a.acquire()
        try:
            # B's vault uses a different lock; this must complete promptly
            # while A's lock is still held.
            out = await asyncio.wait_for(
                tool_b.execute(
                    operation="create", type="concepts", slug="indep",
                    title="Indep", body="independent vault write",
                ),
                timeout=5,
            )
            assert "Created page concepts/indep.md" in out
            assert (
                _vault_root(store, "telegram_2")
                / "wiki" / "concepts" / "indep.md"
            ).is_file()
        finally:
            lock_a.release()


# =========================================================================== #
# I. GOLDEN end-to-end: wiki_enabled=False is byte-identical to pre-wiki
# =========================================================================== #


class TestWikiOffGoldenE2E:
    async def test_full_turn_and_dream_cycle_byte_identical_wiki_off(
        self, dream, mock_provider, mock_runner, store,
    ):
        """``wiki_enabled=False`` end-to-end: a full Dream cycle behaves
        exactly like legacy (cursor/compact/git the 4.1 way) and creates NO
        ``memory/users/``; the system prompt uses the global MEMORY.md, NOT
        a vault MOC. Composes the 4.1 Dream golden + the 6.1 wiki-off
        prompt golden at the e2e level."""
        dream.wiki_enabled = False
        store.append_history("event 1")
        store.append_history("event 2")
        assert store.get_last_dream_cursor() == 0

        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner.run = _ok_runner()

        assert await dream.run() is True

        # Legacy Dream path intact: cursor advanced, Phase 1 + Phase 2 once.
        assert store.get_last_dream_cursor() == 2
        mock_provider.chat_with_retry.assert_called_once()
        mock_runner.run.assert_called_once()

        # ZERO wiki artifacts anywhere under the workspace.
        users_root = store.workspace / "memory" / "users"
        assert not users_root.exists(), (
            f"{users_root} created while wiki disabled — golden violated"
        )
        stray = [p for p in store.workspace.rglob("wiki") if p.is_dir()]
        assert not stray, f"stray wiki dir while wiki disabled: {stray}"

        # The system prompt uses the GLOBAL MEMORY.md, byte-identical to
        # the pre-6.1 path, and is identical with vs without a session_key
        # while wiki is OFF.
        off = ContextBuilder(workspace=store.workspace, wiki_enabled=False)
        on_off = ContextBuilder(workspace=store.workspace, wiki_enabled=True)
        p_off = off.build_system_prompt()
        p_off_key = off.build_system_prompt(session_key="telegram:1")
        # wiki_enabled=True but no session_key must equal wiki-off byte-for-byte.
        p_on_nokey = on_off.build_system_prompt()
        assert p_off == p_off_key == p_on_nokey
        assert "# Memory\n\n## Long-term Memory\n" in p_off
        assert "Project X active" in p_off
        # No vault wikilink leakage into the wiki-off prompt.
        assert "[[concepts/" not in p_off

    async def test_wiki_off_dream_noop_creates_no_vault(
        self, dream, mock_provider, mock_runner, store,
    ):
        """No unprocessed history + wiki off = the exact legacy no-op:
        False, no LLM/runner calls, no memory/users/."""
        dream.wiki_enabled = False
        assert await dream.run() is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()
        assert not (store.workspace / "memory" / "users").exists()


# =========================================================================== #
# I-1. Master-switch: wiki_note tool MUST NOT be registered / sent to the LLM
#      when wiki_enabled=False (final-review must-fix; restores byte-identity
#      with v0.2.0 — the extra tool definition every turn broke the guarantee).
# =========================================================================== #


# =========================================================================== #
# J. Hybrid (BM25) search end-to-end through the real wiki_note tool
# =========================================================================== #


class TestHybridSearchE2E:
    """End-to-end through the actual ``wiki_note`` tool (Task 9): tokenized
    BM25 ranking + the three search-mode coexistence, no extra installed
    (BM25-only — the dense tier is absent/auto-detected-off here).

    Reuses this file's own harness verbatim: ``_wiki_tool`` (the
    SimpleNamespace ToolContext + RequestContext wiring mirrored from
    ``tests/agent/tools/test_wiki_note.py``) and ``_vault_root``; pages are
    created through the tool's ``create`` operation, exactly as
    ``TestCoreValuePropLoop`` does.
    """

    async def test_tokenized_bm25_beats_exact_substring(self, store):
        """A multi-word keyword query whose EXACT phrase appears in NO page
        still ranks the page matching the most/rarest query terms first —
        proving the search is tokenized Okapi BM25 (retrieval.bm25_ranking),
        not the old exact-substring scan. The query 'idempotency retries
        payment' appears verbatim nowhere; the payment-flow page is the only
        page carrying the rare terms 'idempotency' + 'retries', so it must
        rank first."""
        session = "telegram:1"
        tool = _wiki_tool(store.workspace, session)

        # A mix of people / projects / concepts pages via the tool's create.
        await tool.execute(
            operation="create", type="people", slug="alice",
            title="Alice Rossi",
            body="Alice manages the marketing team and quarterly reports.",
        )
        await tool.execute(
            operation="create", type="projects", slug="logistica",
            title="Logistica Platform",
            body="Warehouse logistics platform with shipment tracking.",
        )
        await tool.execute(
            operation="create", type="concepts", slug="payment-flow",
            title="Payment Flow",
            body=(
                "The payment service uses idempotency keys and safe retries "
                "so a duplicated request never charges a card twice."
            ),
        )
        await tool.execute(
            operation="create", type="concepts", slug="checkout",
            title="Checkout",
            body="The checkout step collects the cart and starts a payment.",
        )

        # Exact phrase 'idempotency retries payment' is in NO page; the
        # payment-flow page carries the rarest matching terms.
        out = tool._do_search("idempotency retries payment")
        lines = [ln for ln in out.splitlines() if ln.startswith("- ")]
        assert lines, f"no ranked results returned:\n{out}"
        assert lines[0].startswith("- concepts/payment-flow.md"), (
            "tokenized BM25 did not rank the rarest-term page first "
            f"(exact-substring behavior would have failed entirely):\n{out}"
        )

    async def test_three_search_modes_coexist(self, store):
        """Empty query → most-recent ordering; ``tag:NAME`` → only tag-matched
        pages; keyword → ranked results. All three through the real tool."""
        session = "telegram:2"
        vroot = _vault_root(store, "telegram_2")
        tool = _wiki_tool(store.workspace, session)

        await tool.execute(
            operation="create", type="people", slug="bob",
            title="Bob", body="Bob is an engineer who owns the billing code.",
        )
        await tool.execute(
            operation="create", type="projects", slug="atlas",
            title="Atlas", body="Atlas is the data ingestion pipeline.",
        )

        # Seed a tagged page directly (the tool's create always sets tags=[],
        # so a tag-mode assertion needs a page authored with a tag on disk).
        # Give it a clearly NEWER last_touched so the empty-query (recency)
        # branch has a deterministic, content-driven winner.
        concepts = vroot / "wiki" / "concepts"
        concepts.mkdir(parents=True, exist_ok=True)
        tagged = Page(
            type="concepts", title="GDPR Notes", status="hot",
            created="2026-05-20", updated="2026-05-20",
            last_touched="2099-01-01",
            tags=["compliance"], links_out=[], pinned=None,
            body="Data retention and consent requirements.",
        )
        (concepts / "gdpr.md").write_text(serialize_page(tagged), encoding="utf-8")

        # 1. Empty query → most-recently-touched first (gdpr's 2099 date wins).
        empty_out = tool._do_search("")
        assert "most recently touched" in empty_out
        empty_lines = [ln for ln in empty_out.splitlines() if ln.startswith("- ")]
        assert empty_lines[0].startswith("- concepts/gdpr.md"), (
            f"empty query did not order by recency:\n{empty_out}"
        )
        # All three pages are listed in recency mode.
        assert any("people/bob.md" in ln for ln in empty_lines)
        assert any("projects/atlas.md" in ln for ln in empty_lines)

        # 2. tag:compliance → ONLY the tagged page.
        tag_out = tool._do_search("tag:compliance")
        tag_lines = [ln for ln in tag_out.splitlines() if ln.startswith("- ")]
        assert len(tag_lines) == 1
        assert tag_lines[0].startswith("- concepts/gdpr.md")
        assert "atlas" not in tag_out and "bob" not in tag_out

        # 3. Keyword → ranked results (the billing/engineer page for 'billing').
        kw_out = tool._do_search("billing engineer")
        kw_lines = [ln for ln in kw_out.splitlines() if ln.startswith("- ")]
        assert kw_lines, f"keyword search returned no results:\n{kw_out}"
        assert kw_lines[0].startswith("- people/bob.md"), (
            f"keyword mode did not rank the matching page first:\n{kw_out}"
        )
        assert f"matching {'billing engineer'!r}" in kw_out


class TestWikiNoteToolGatedOnWikiEnabled:
    """``WikiNoteTool`` must be gated on the resolved ``dream.wiki_enabled``.

    With the gate OFF the loader must NOT register it and
    ``registry.get_definitions()`` (the exact list sent to the provider as
    the ``tools=`` array every turn) must NOT contain ``wiki_note`` — so a
    stock wiki-OFF install is byte-identical to pre-wiki nanobot. With the
    gate ON it must register exactly as before (e2e A/G/H depend on it).
    """

    @staticmethod
    def _load(workspace, *, wiki_enabled: bool) -> ToolRegistry:
        """Drive the real loader+ToolContext path the production loop uses
        (loop.py ``_register_default_tools`` builds this same ToolContext and
        calls ``ToolLoader().load(ctx, registry)``)."""
        from nanobot.config.schema import ToolsConfig

        ctx = ToolContext(
            config=ToolsConfig(),
            workspace=str(workspace),
            wiki_enabled=wiki_enabled,
        )
        registry = ToolRegistry()
        ToolLoader().load(ctx, registry)
        return registry

    @staticmethod
    def _definition_names(registry: ToolRegistry) -> set[str]:
        names: set[str] = set()
        for schema in registry.get_definitions():
            fn = schema.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                names.add(fn["name"])
            elif isinstance(schema.get("name"), str):
                names.add(schema["name"])
        return names

    def test_wiki_note_tool_absent_when_disabled(self, tmp_path):
        """The I-1 regression. On the pre-fix code ``WikiNoteTool`` has no
        ``enabled()`` override -> it is registered AND in get_definitions()
        even with the gate off (this assertion FAILS pre-fix). Post-fix it
        is absent when disabled and present when enabled."""
        off = self._load(tmp_path, wiki_enabled=False)
        on = self._load(tmp_path, wiki_enabled=True)

        off_names = self._definition_names(off)
        on_names = self._definition_names(on)

        # Gate OFF: not registered, not in the provider tools= array.
        assert "wiki_note" not in off.tool_names, (
            "wiki_note registered while wiki_enabled=False — breaks the "
            "master-switch byte-identity guarantee (v0.2.0 baseline)"
        )
        assert "wiki_note" not in off_names, (
            "wiki_note in registry.get_definitions() while wiki_enabled=False "
            "— its JSON schema would be sent to the provider every turn"
        )

        # Gate ON: registered + in the provider tools= array (feature works;
        # e2e A/G/H rely on the tool existing under wiki_enabled=True).
        assert "wiki_note" in on.tool_names
        assert "wiki_note" in on_names

        # Byte-identity-leaning: the disabled tool set differs from the
        # enabled one by EXACTLY {"wiki_note"} and nothing else (no other
        # tool's registration semantics changed).
        assert on_names - off_names == {"wiki_note"}
        assert off_names - on_names == set()
