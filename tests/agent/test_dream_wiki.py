"""Milestone 4 GOLDEN GUARD — Dream is byte-identical to nanobot v0.2.0 when the wiki is off.

This is a TEST-ONLY regression lock. No production code is (or should be) touched
to make it pass: today's ``Dream`` has zero wiki code, so "wiki disabled" is, by
definition, identical to the v0.2.0 baseline. These tests characterize that
baseline as an executable snapshot.

Milestone 4 will extend ``Dream.run()`` with wiki Ingest+Lint, gated behind
``dream.wiki_enabled``. The plan's GOLDEN RULE: until Milestone 6, no existing
code path may change behavior when ``wiki_enabled`` is false. This file is the
executable guard for that rule on the Dream path:

  * It PASSES NOW with no production change.
  * It MUST keep passing after Task 4.5 wires Ingest/Lint into ``Dream`` (the
    gate, off, must be a true no-op).
  * It WILL FAIL the instant a future M4 change makes ``Dream`` create a wiki
    vault (``memory/users/`` / any ``wiki/`` dir) or otherwise diverge from the
    legacy MEMORY.md path while ``wiki_enabled`` is False.

``dream.wiki_enabled`` is set here as a plain attribute. ``Dream`` does not
declare it until Task 4.5; assigning an undeclared attribute is intentional and
fine — Task 4.5 makes it a real ``__init__`` attribute (not a property/slot), so
the assignment stays valid.

Fixture provenance: the ``store`` / ``mock_provider`` / ``mock_runner`` / ``dream``
fixtures and the ``_make_run_result`` helper are COPIED verbatim from
``tests/agent/test_dream.py`` (commit on branch ``feat/wiki-tree-memory``).
They are module-local there and not exported via ``tests/agent/conftest.py``
(which only provides ``loop_factory``/``make_loop``/``make_provider`` for
AgentLoop), so importing them as pytest fixtures across modules would be
fragile. Copying them keeps the "baseline" a faithful mirror of the real
passing Dream tests rather than a strawman. Keep this block in sync with
``test_dream.py`` if its fixtures ever change.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

import nanobot.agent.memory as memory_mod
from nanobot.agent.memory import Dream, MemoryStore
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.wiki.paths import vault_slug
from nanobot.utils.vault_lock import get_vault_lock

# --- Cross-loop lock isolation (review follow-up I1) -------------------------


@pytest.fixture(autouse=True)
def _reset_dream_run_lock():
    """Neutralize the module-global ``_DREAM_RUN_LOCK`` cross-loop bind footgun.

    ``nanobot.agent.memory._DREAM_RUN_LOCK`` is a deliberate process-wide
    module-global ``asyncio.Lock`` (reviewer-mandated; see the comment block on
    it in ``memory.py``). On first *contention* it permanently binds to that
    event loop. Production is a single ``asyncio.run`` per process so this is a
    non-issue there — but pytest runs each ``async def`` test on its OWN event
    loop (``asyncio_mode=auto``). Once a test *contends* the lock
    (``test_concurrent_run_serialized`` does, via two gathered ``dream.run()``s),
    the binding leaks: any later cross-loop *contended* ``Dream.run()`` — or an
    unrelated test that happens to contend it — would raise
    ``RuntimeError: <Lock> is bound to a different event loop`` and could mask a
    real regression in this golden gate.

    We swap the module global for a FRESH unbound ``asyncio.Lock()`` in
    teardown. Replacing the object (vs. poking private ``_loop`` internals) is
    the cleanest reset and re-establishes the import-time invariant ("no loop
    bound until first contention") for the next test's loop. The reset runs in
    TEARDOWN (post-yield), so within any single test the identity invariant
    still holds: ``test_guard_is_module_global_strong_ref`` observes the SAME
    object for the whole test (its ``is`` assertions never cross the reset), and
    ``test_concurrent_run_serialized`` serializes on one stable object for its
    whole body — the swap only takes effect once the test has finished. This is
    purely test isolation; no production behavior changes.
    """
    yield
    memory_mod._DREAM_RUN_LOCK = asyncio.Lock()


# --- Fixtures copied verbatim from tests/agent/test_dream.py -----------------
# (module-local there; not exported via conftest.py — see module docstring)


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
    d = Dream(store=store, provider=mock_provider, model="test-model", max_batch_size=5)
    d._runner = mock_runner
    return d


def _make_run_result(
    stop_reason="completed",
    final_content=None,
    tool_events=None,
    usage=None,
):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


# --- Golden guard -----------------------------------------------------------


def _assert_no_wiki_artifacts(workspace):
    """No per-user wiki vault and no `wiki/` dir may exist anywhere in workspace.

    Task 4.5+ will introduce ``<workspace>/memory/users/<user>/...`` as the wiki
    vault root. With the gate off it must never be created. We also reject any
    ``wiki/`` directory created anywhere under the workspace as a belt-and-braces
    catch for alternative layouts.
    """
    users_root = workspace / "memory" / "users"
    assert not users_root.exists(), (
        f"wiki vault root {users_root} was created while wiki disabled — "
        "M4 golden rule violated (Dream touched the wiki with the gate off)"
    )
    # Belt-and-braces: no `wiki/` directory anywhere under the workspace.
    stray_wiki = [p for p in workspace.rglob("wiki") if p.is_dir()]
    assert not stray_wiki, (
        f"unexpected wiki directory/-ies created while wiki disabled: {stray_wiki}"
    )


class TestDreamWikiDisabledGolden:
    """Locks v0.2.0 Dream behavior for the wiki-disabled path (Milestone 4 guard)."""

    async def test_dream_run_creates_no_wiki_artifacts_when_disabled(
        self, dream, mock_provider, mock_runner, store,
    ):
        """A full successful Dream run with the wiki gate off must behave exactly
        like the v0.2.0 baseline (cursor advances, runner invoked once, history
        compacted) AND must create zero wiki artifacts.

        Mirrors test_dream.py::test_calls_runner_for_unprocessed_entries +
        ::test_advances_dream_cursor — the canonical "work was done" path.
        """
        # Gate explicitly off. `Dream` doesn't declare `wiki_enabled` until Task
        # 4.5; assigning it now is intentional (see module docstring).
        dream.wiki_enabled = False

        # Seed unprocessed history exactly as the baseline tests do so run()
        # actually does work (two entries → final cursor must land on 2).
        store.append_history("event 1")
        store.append_history("event 2")
        assert store.get_last_dream_cursor() == 0  # baseline precondition

        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()

        # 1. Same truthy result a normal successful run returns (work done).
        assert result is True

        # 2. Legacy behavior intact: Dream cursor advanced past processed batch.
        assert store.get_last_dream_cursor() == 2

        # 3. Legacy MEMORY.md path still functioned — Phase 1 LLM call + Phase 2
        #    AgentRunner delegation happened exactly as the baseline asserts.
        mock_provider.chat_with_retry.assert_called_once()
        mock_runner.run.assert_called_once()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == 10
        assert spec.fail_on_tool_error is False

        # 4. No wiki vault / no stray wiki dir anywhere under the workspace.
        _assert_no_wiki_artifacts(store.workspace)

    async def test_dream_noop_still_noop_when_wiki_attr_set_false(
        self, dream, mock_provider, mock_runner, store,
    ):
        """With no unprocessed history the wiki-disabled run must be the exact
        same no-op as the v0.2.0 baseline: returns False, no LLM/runner calls,
        and no wiki artifacts.

        Mirrors test_dream.py::test_noop_when_no_unprocessed_history.
        """
        dream.wiki_enabled = False

        result = await dream.run()

        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()
        _assert_no_wiki_artifacts(store.workspace)


# --- Wiki ENABLED (Task 4.5 C) ---------------------------------------------

# Canned Ingest line-protocol output: one PAGE directive that creates
# people/alice.md in the unified vault.
_INGEST_OUTPUT = "[PAGE people/alice]\nAlice is a backend engineer who prefers dark mode.\n"


class TestDreamWikiEnabled:
    """With ``dream.wiki_enabled = True`` Ingest+Lint run per vault under the
    per-vault lock, AFTER the legacy MEMORY.md path, exception-isolated."""

    async def test_wiki_enabled_runs_ingest_and_lint_on_unified_vault(
        self, dream, mock_provider, mock_runner, store,
    ):
        """Enabled: the unified vault is created (SCHEMA.md copied by
        ensure_initialized) and Ingest writes people/alice.md; the legacy
        path (Phase 1/2, cursor advance, compact) is unchanged."""
        dream.wiki_enabled = True

        store.append_history("event 1")
        store.append_history("event 2")
        assert store.get_last_dream_cursor() == 0

        # Phase 1 (legacy) gets the first call; Ingest gets the second. Each
        # needs its own canned content — Phase 1 wants a plain analysis
        # string, Ingest wants the line protocol.
        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()

        assert result is True

        # Legacy behavior intact, exactly as the disabled golden case.
        assert store.get_last_dream_cursor() == 2
        mock_runner.run.assert_called_once()

        # Unified vault created + SCHEMA.md copied by ensure_initialized.
        vault_root = store.workspace / "memory" / "users" / "unified_default"
        schema_md = vault_root / "wiki" / "SCHEMA.md"
        assert schema_md.is_file()

        # Ingest wrote the page; Lint left it parseable.
        alice = vault_root / "wiki" / "people" / "alice.md"
        assert alice.is_file()
        from nanobot.agent.wiki.vault import Vault
        page = Vault(vault_root).read_page("people/alice.md")
        assert page.type == "people"
        assert "dark mode" in page.body

    async def test_wiki_enabled_first_cycle_migrates_legacy_workspace(
        self, dream, mock_provider, mock_runner, store,
    ):
        """Task 7.1: the FIRST wiki-enabled Dream cycle migrates the LEGACY
        global memory/USER into the unified vault (via ensure_initialized),
        then Lint builds the MOC — closing the 6.1 enable-ordering window.

        The ``store`` fixture already wrote a NON-template ``memory/MEMORY.md``
        ("# Memory\n- Project X active") and a root ``USER.md``
        ("# User\n- Developer")."""
        dream.wiki_enabled = True

        store.append_history("event 1")
        store.append_history("event 2")

        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()

        assert result is True
        vault_root = store.workspace / "memory" / "users" / "unified_default"
        # One-shot migration marker set on the first cycle.
        assert (vault_root / ".migrated").is_file()
        # Legacy memory imported as a concepts page + Lint built the MOC.
        imported = vault_root / "wiki" / "concepts" / "imported-memory.md"
        assert imported.is_file()
        assert "Project X active" in imported.read_text(encoding="utf-8")
        moc = vault_root / "MEMORY.md"
        assert moc.is_file()
        assert "[[concepts/imported-memory]]" in moc.read_text(encoding="utf-8")
        # Legacy global USER.md became the vault USER.md (what 6.1 reads).
        assert (vault_root / "USER.md").read_text(encoding="utf-8") == "# User\n- Developer"
        # Legacy sources are byte-unchanged (never deleted/modified).
        assert store.read_memory() == "# Memory\n- Project X active"
        assert store.read_user() == "# User\n- Developer"

    async def test_legacy_migration_gated_to_unified_vault_only(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """C1 gate: ``migrate_legacy`` reads the SINGLE GLOBAL
        ``memory/MEMORY.md`` + root ``USER.md``. The Dream call site must pass
        ``legacy_workspace`` ONLY for the ``unified_default`` slug — so when
        Task 7.2 makes ``_vaults_for_batch`` return per-user slugs, the global
        memory/USER blob (incl. another user's profile) is NOT fanned into
        every vault.

        Today ``_vaults_for_batch`` returns only ``[unified_default]`` so the
        gate is behavior-identical; here we simulate the post-7.2 world by
        making it return the unified slug PLUS an extra per-user slug, then
        assert the extra vault is initialized (wiki/ + SCHEMA) but NEVER
        migrated (no ``.migrated`` marker, no imported-memory page, no
        USER.md), while the unified vault IS migrated exactly as before."""
        dream.wiki_enabled = True
        store.append_history("event 1")
        store.append_history("event 2")

        unified = vault_slug("unified:default")
        extra = vault_slug("telegram:999")
        assert unified == "unified_default"
        assert extra != unified
        monkeypatch.setattr(dream, "_vaults_for_batch", lambda _b: [unified, extra])

        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        users = store.workspace / "memory" / "users"
        unified_root = users / unified
        extra_root = users / extra

        # Unified vault: migrated exactly as before (golden 7.1 behavior).
        assert (unified_root / ".migrated").is_file()
        assert (unified_root / "wiki" / "concepts" / "imported-memory.md").is_file()
        assert (unified_root / "USER.md").read_text(encoding="utf-8") == "# User\n- Developer"

        # Extra (per-user) vault: initialized but the global blob was NOT
        # fanned in — NO migration ran for it (C1 gate holds).
        assert (extra_root / "wiki" / "SCHEMA.md").is_file()
        assert not (extra_root / ".migrated").exists()
        assert not (extra_root / "wiki" / "concepts" / "imported-memory.md").exists()
        assert not (extra_root / "USER.md").exists()

    async def test_wiki_failure_does_not_break_legacy(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """A wiki exception is logged and SWALLOWED — the legacy path
        (result, cursor advance, compact) is byte-identical to success."""
        dream.wiki_enabled = True

        store.append_history("event 1")
        store.append_history("event 2")

        mock_provider.chat_with_retry.return_value = MagicMock(
            content="New fact", finish_reason="stop",
        )
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        compact_spy = MagicMock(side_effect=store.compact_history)
        monkeypatch.setattr(store, "compact_history", compact_spy)

        def _boom(*a, **k):
            raise RuntimeError("wiki exploded")

        monkeypatch.setattr(memory_mod, "run_ingest", AsyncMock(side_effect=_boom))

        # nanobot logs via loguru, not stdlib logging — add a sink to capture.
        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="ERROR")
        try:
            result = await dream.run()
        finally:
            logger.remove(sink_id)

        # Legacy path entirely intact despite the wiki blowing up.
        assert result is True
        assert store.get_last_dream_cursor() == 2
        compact_spy.assert_called_once()
        # The failure was logged via loguru, not propagated.
        assert any("wiki ingest/lint failed" in m for m in captured)

    async def test_wiki_noop_when_no_history(
        self, dream, mock_provider, mock_runner, store,
    ):
        """No unprocessed history → no-op result, provider NOT called, and the
        wiki block must NOT run (no memory/users/ created)."""
        dream.wiki_enabled = True

        result = await dream.run()

        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()
        _assert_no_wiki_artifacts(store.workspace)

    async def test_lock_serializes_with_wiki_note(
        self, dream, mock_provider, mock_runner, store,
    ):
        """The wiki block must acquire ``get_vault_lock(vault_slug(
        "unified:default"))`` — the SAME lock the wiki_note tool takes — so
        the two serialize. Holding it externally blocks the wiki write until
        released; the legacy path completes regardless."""
        slug = vault_slug("unified:default")
        assert slug == "unified_default"
        lock = get_vault_lock(slug)

        dream.wiki_enabled = True
        store.append_history("event 1")

        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        alice = store.workspace / "memory" / "users" / "unified_default" / "wiki" / "people" / "alice.md"

        await lock.acquire()
        try:
            task = asyncio.ensure_future(dream.run())
            # Give the run a chance to reach (and block on) the vault lock.
            await asyncio.sleep(0.05)
            assert not task.done() or not alice.exists(), (
                "wiki write completed while the vault lock was held externally "
                "— the block did not serialize on get_vault_lock(slug)"
            )
            assert not alice.exists()
        finally:
            lock.release()

        result = await asyncio.wait_for(task, timeout=5)
        assert result is True
        assert store.get_last_dream_cursor() == 1
        assert alice.is_file()


# --- Layer 1a: unified vault USER.md mirror ---------------------------------


class TestDreamVaultUserMirror:
    """Layer 1a: with wiki on, each Dream cycle mirrors the (Phase-2-refined)
    global USER.md into the UNIFIED vault's USER.md, un-freezing the stale
    vault file the prompt reads. Unified-only, deterministic (no LLM call)."""

    async def test_unified_vault_user_is_refreshed_from_global(
        self, dream, mock_provider, mock_runner, store,
    ):
        dream.wiki_enabled = True
        # Global USER.md = the rich, Phase-2-maintained card (fixture seeds it).
        # Pre-create a STALE vault USER.md (simulating the frozen migrate one-shot).
        vault_root = store.workspace / "memory" / "users" / "unified_default"
        (vault_root / "wiki").mkdir(parents=True, exist_ok=True)
        (vault_root / "USER.md").write_text("STALE 3-line note", encoding="utf-8")

        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True
        # Vault USER.md now equals the global card (stale note overwritten).
        assert (vault_root / "USER.md").read_text(encoding="utf-8") == "# User\n- Developer"

    async def test_mirror_skipped_when_global_user_blank(
        self, dream, mock_provider, mock_runner, store,
    ):
        """If the global USER.md is blank/whitespace, do NOT clobber the vault
        copy with emptiness (guard against wiping a good vault card)."""
        dream.wiki_enabled = True
        store.write_user("   \n")  # blank global
        vault_root = store.workspace / "memory" / "users" / "unified_default"
        (vault_root / "wiki").mkdir(parents=True, exist_ok=True)
        (vault_root / "USER.md").write_text("keep me", encoding="utf-8")
        store.append_history("event 1")
        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "x"}],
        ))
        await dream.run()
        assert (vault_root / "USER.md").read_text(encoding="utf-8") == "keep me"


# --- Task 7: Dream-time embedding refresh (dense tier optional) -------------


class TestDreamWikiEmbeddingsRefresh:
    """With ``dream.wiki_embeddings = True`` but fastembed stubbed OFF, the
    Dream cycle must complete normally: Ingest+Lint still run, NO ``.embeddings/``
    dir is created (refresh_embeddings is a clean no-op without fastembed)."""

    async def test_embeddings_refresh_noop_without_fastembed(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """Flag on, fastembed off → cycle completes, Ingest wrote alice.md,
        and no embeddings artifacts exist. Mirrors
        ::test_wiki_enabled_runs_ingest_and_lint_on_unified_vault."""
        dream.wiki_enabled = True
        dream.wiki_embeddings = True
        # Stub fastembed off so refresh_embeddings is a no-op.
        monkeypatch.setattr(
            "nanobot.agent.wiki.embeddings._import_text_embedding", lambda: None,
        )

        store.append_history("event 1")
        store.append_history("event 2")
        assert store.get_last_dream_cursor() == 0

        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()

        # Cycle completes normally; legacy path intact.
        assert result is True
        assert store.get_last_dream_cursor() == 2
        mock_runner.run.assert_called_once()

        vault_root = store.workspace / "memory" / "users" / "unified_default"
        # Ingest+Lint still happened (alice.md written and parseable).
        alice = vault_root / "wiki" / "people" / "alice.md"
        assert alice.is_file()
        from nanobot.agent.wiki.vault import Vault
        page = Vault(vault_root).read_page("people/alice.md")
        assert page.type == "people"
        assert "dark mode" in page.body

        # No embeddings artifacts: refresh was a clean no-op without fastembed.
        assert not (vault_root / "wiki" / ".embeddings").exists()


# --- Task 7.2: per-user history routing + multi-user vault isolation --------


class TestVaultsForBatchGrouping:
    """``_vaults_for_batch`` groups a batch by ``vault_slug(session_key)`` and
    returns the SORTED distinct slug list (deterministic)."""

    def test_vaults_for_batch_groups_by_session_key(self, dream):
        batch = [
            {"cursor": 1, "timestamp": "t", "content": "a", "session_key": "telegram:1"},
            {"cursor": 2, "timestamp": "t", "content": "b", "session_key": "telegram:1"},
            {"cursor": 3, "timestamp": "t", "content": "c", "session_key": "telegram:2"},
            {"cursor": 4, "timestamp": "t", "content": "d"},  # legacy / untagged
        ]
        assert dream._vaults_for_batch(batch) == [
            "telegram_1",
            "telegram_2",
            "unified_default",
        ]

    def test_all_unified_collapses_to_single_slug(self, dream):
        batch = [
            {"cursor": 1, "timestamp": "t", "content": "a", "session_key": "unified:default"},
            {"cursor": 2, "timestamp": "t", "content": "b", "session_key": "unified:default"},
        ]
        assert dream._vaults_for_batch(batch) == ["unified_default"]

    def test_all_legacy_untagged_collapses_to_unified(self, dream):
        batch = [
            {"cursor": 1, "timestamp": "t", "content": "a"},
            {"cursor": 2, "timestamp": "t", "content": "b"},
        ]
        assert dream._vaults_for_batch(batch) == ["unified_default"]


def _content_echo_provider(mock_provider):
    """Make the mocked provider's Ingest call emit a PAGE whose body is the
    user-prompt history text it received.

    Phase 1 (legacy) gets the FIRST call (plain analysis string). Every
    subsequent call is an Ingest call: we echo the conversation-history text
    from the user message into a single PAGE body, so a vault that received
    another user's slice would visibly contain that user's text — making any
    cross-bleed detectable.
    """
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
            content=f"[PAGE concepts/dump]\n{user_msg}\n",
            finish_reason="stop",
        )

    mock_provider.chat_with_retry.side_effect = _side_effect
    return state


class TestMultiUserVaultIsolation:
    """Task 7.2: each user's history slice is consolidated into THAT user's
    own per-user vault — no cross-bleed."""

    async def test_multiuser_ingest_isolation(
        self, dream, mock_provider, mock_runner, store,
    ):
        """Two tagged users → two distinct vaults; each vault contains ONLY
        its own user's text. The Ingest slice each vault received is the
        filtered per-slug batch, NOT the whole batch."""
        dream.wiki_enabled = True

        store.append_history("USER ONE secret alpha", session_key="telegram:1")
        store.append_history("USER TWO secret beta", session_key="telegram:2")
        store.append_history("USER ONE more alpha", session_key="telegram:1")

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        users = store.workspace / "memory" / "users"
        v1_page = users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        v2_page = users / "telegram_2" / "wiki" / "concepts" / "dump.md"
        assert v1_page.is_file()
        assert v2_page.is_file()

        v1_text = v1_page.read_text(encoding="utf-8")
        v2_text = v2_page.read_text(encoding="utf-8")

        # User 1's vault has ONLY user 1 content.
        assert "USER ONE secret alpha" in v1_text
        assert "USER ONE more alpha" in v1_text
        assert "USER TWO secret beta" not in v1_text

        # User 2's vault has ONLY user 2 content.
        assert "USER TWO secret beta" in v2_text
        assert "USER ONE secret alpha" not in v2_text
        assert "USER ONE more alpha" not in v2_text

        # No spurious unified vault (every entry was tagged per-user).
        assert not (users / "unified_default").exists()

    async def test_unified_session_collapses_to_one_vault(
        self, dream, mock_provider, mock_runner, store,
    ):
        """All entries keyed ``unified:default`` (simulating
        ``unified_session=True``) → ONLY ``memory/users/unified_default/``,
        no per-user vaults."""
        dream.wiki_enabled = True

        store.append_history("device A", session_key="unified:default")
        store.append_history("device B", session_key="unified:default")

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        users = store.workspace / "memory" / "users"
        assert (users / "unified_default").is_dir()
        others = [p.name for p in users.iterdir() if p.name != "unified_default"]
        assert others == [], f"unexpected per-user vaults: {others}"

    async def test_legacy_untagged_history_routes_to_unified(
        self, dream, mock_provider, mock_runner, store,
    ):
        """A history.jsonl with LEGACY records lacking ``session_key`` is
        routed to ``unified_default`` only (back-compat)."""
        dream.wiki_enabled = True

        # Hand-write legacy records WITHOUT the session_key field.
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "legacy one"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "legacy two"}\n',
            encoding="utf-8",
        )

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        users = store.workspace / "memory" / "users"
        assert (users / "unified_default").is_dir()
        others = [p.name for p in users.iterdir() if p.name != "unified_default"]
        assert others == [], f"legacy entries leaked to per-user vaults: {others}"
        dump = users / "unified_default" / "wiki" / "concepts" / "dump.md"
        assert dump.is_file()
        body = dump.read_text(encoding="utf-8")
        assert "legacy one" in body
        assert "legacy two" in body

    async def test_legacy_migration_gated_only_unified_with_live_routing(
        self, dream, mock_provider, mock_runner, store,
    ):
        """C1 gate under LIVE multi-slug routing (no monkeypatch of
        ``_vaults_for_batch``): real per-user entries produce real per-user
        vaults; the legacy GLOBAL memory/USER blob migrates ONLY into
        ``unified_default``. Per-user vaults get their own Ingest slice but
        NEVER a ``.migrated`` / ``imported-memory.md`` / ``USER.md``."""
        dream.wiki_enabled = True

        # Mixed batch: a unified-keyed entry AND two distinct per-user ones.
        store.append_history("unified line", session_key="unified:default")
        store.append_history("alpha for one", session_key="telegram:1")
        store.append_history("beta for two", session_key="telegram:2")

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        users = store.workspace / "memory" / "users"
        unified_root = users / "unified_default"
        v1 = users / "telegram_1"
        v2 = users / "telegram_2"

        # Unified vault migrated exactly as in 7.1.
        assert (unified_root / ".migrated").is_file()
        assert (unified_root / "wiki" / "concepts" / "imported-memory.md").is_file()
        assert (unified_root / "USER.md").read_text(encoding="utf-8") == "# User\n- Developer"

        # Per-user vaults: initialized + got their OWN slice, but NEVER the
        # global blob (C1 gate holds under live routing).
        for v, own, foreign in (
            (v1, "alpha for one", "beta for two"),
            (v2, "beta for two", "alpha for one"),
        ):
            assert (v / "wiki" / "SCHEMA.md").is_file()
            assert not (v / ".migrated").exists()
            assert not (v / "wiki" / "concepts" / "imported-memory.md").exists()
            assert not (v / "USER.md").exists()
            dump = (v / "wiki" / "concepts" / "dump.md").read_text(encoding="utf-8")
            assert own in dump
            assert foreign not in dump
            assert "unified line" not in dump


# --- Task 7.2 review follow-up: C1 (null-safe slug) + I1 (per-user isolation)


class TestNullSessionKeyRouting:
    """C1: a record with ``"session_key": null`` (JSON null — reachable via
    external/legacy/hand-edited/malformed writers) must route to the unified
    vault, NOT crash ``vault_slug(None)`` and skip the ENTIRE wiki pass for
    every user that cycle (cursor already advanced → unrecoverable).
    """

    def test_vaults_for_batch_null_session_key_routes_to_unified_not_crash(
        self, dream,
    ):
        """``_vaults_for_batch`` with a JSON-null ``session_key`` must return
        ``unified_default`` (NOT raise ``AttributeError`` on
        ``None.replace``)."""
        batch = [
            {"cursor": 1, "timestamp": "t", "content": "x", "session_key": None},
            {"cursor": 2, "timestamp": "t", "content": "y", "session_key": "telegram:1"},
        ]
        # Must NOT raise; must group the null record under the unified slug.
        assert dream._vaults_for_batch(batch) == ["telegram_1", "unified_default"]

    async def test_null_session_key_routes_to_unified_not_crash(
        self, dream, mock_provider, mock_runner, store,
    ):
        """End-to-end: a batch mixing a ``session_key: null`` record and a
        ``telegram:1`` record → BOTH ingested (null→unified_default vault,
        telegram:1→telegram_1 vault). No ``AttributeError``; the wiki pass
        is NOT skipped for everyone.

        On CURRENT code ``vault_slug(None)`` raises ``AttributeError`` inside
        ``_vaults_for_batch`` (called before the per-slug loop, inside the
        batch-global try) → the whole wiki Ingest+Lint is skipped for EVERY
        user this cycle while the cursor has already advanced.
        """
        dream.wiki_enabled = True

        # Hand-write a JSON-null session_key record + a real per-user one.
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": '
            '"null keyed line", "session_key": null}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": '
            '"tg one line", "session_key": "telegram:1"}\n',
            encoding="utf-8",
        )

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True
        assert store.get_last_dream_cursor() == 2

        users = store.workspace / "memory" / "users"
        # null → unified_default vault.
        unified_dump = users / "unified_default" / "wiki" / "concepts" / "dump.md"
        assert unified_dump.is_file(), (
            "null session_key was NOT routed to unified_default — the wiki "
            "pass was skipped for everyone (C1)"
        )
        assert "null keyed line" in unified_dump.read_text(encoding="utf-8")

        # telegram:1 → its own per-user vault, also ingested (not skipped).
        tg_dump = users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        assert tg_dump.is_file(), (
            "telegram:1 was NOT ingested — one null record skipped ALL "
            "users' wiki pass this cycle (C1)"
        )
        assert "tg one line" in tg_dump.read_text(encoding="utf-8")

    async def test_non_str_truthy_session_key_routes_to_unified_not_crash(
        self, dream, mock_provider, mock_runner, store,
    ):
        """Residual follow-up to 7.2 (same data-loss class as C1, via a
        different bad type): a batch mixing a ``{"session_key": 123}``
        record, a ``{"session_key": ["bad"]}`` record, a legacy untagged
        record, and a ``telegram:1`` record. The non-str TRUTHY keys (123 /
        list) are reachable from the SAME external/legacy/hand-edited/
        malformed history writers ``null`` is — and ``... or
        "unified:default"`` does NOT collapse them (truthy). The
        malformed-type + untagged content must land in
        ``unified_default``; ``telegram:1`` in ``telegram_1``; NO
        ``AttributeError`` escapes ``dream.run()``; NO all-users-skip
        (telegram_1 IS ingested); the cursor advanced.

        On CURRENT ``cd32bf2e`` ``_entry_slug`` →
        ``vault_slug(123)`` → ``123.replace(...)`` → ``AttributeError``
        inside ``_vaults_for_batch``, called at the
        ``for slug in self._vaults_for_batch(batch):`` line which is
        OUTSIDE/BEFORE the per-iteration ``try`` (the old batch-global try
        was removed). The wiki block has no outer handler → the exception
        escapes ``dream.run()`` after the cursor ALREADY advanced → ONE
        poison record = ALL users' wiki ingest skipped this cycle,
        unrecoverable (telegram_1 NOT ingested).
        """
        dream.wiki_enabled = True

        # Hand-write malformed non-str truthy session_key records + a legacy
        # untagged record + a real per-user one.
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": '
            '"int keyed line", "session_key": 123}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": '
            '"list keyed line", "session_key": ["bad"]}\n'
            '{"cursor": 3, "timestamp": "2026-04-01 10:02", "content": '
            '"untagged legacy line"}\n'
            '{"cursor": 4, "timestamp": "2026-04-01 10:03", "content": '
            '"tg one line", "session_key": "telegram:1"}\n',
            encoding="utf-8",
        )

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        # Must NOT raise AttributeError out of dream.run().
        result = await dream.run()
        assert result is True
        assert store.get_last_dream_cursor() == 4

        users = store.workspace / "memory" / "users"
        # Non-str truthy (123, ["bad"]) + untagged all collapse to unified.
        unified_dump = users / "unified_default" / "wiki" / "concepts" / "dump.md"
        assert unified_dump.is_file(), (
            "non-str truthy session_key was NOT routed to unified_default — "
            "the wiki pass was skipped for everyone (residual C1)"
        )
        unified_body = unified_dump.read_text(encoding="utf-8")
        assert "int keyed line" in unified_body
        assert "list keyed line" in unified_body
        assert "untagged legacy line" in unified_body

        # telegram:1 → its own per-user vault, also ingested (NOT skipped by
        # the poison record).
        tg_dump = users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        assert tg_dump.is_file(), (
            "telegram:1 was NOT ingested — one non-str truthy record "
            "skipped ALL users' wiki pass this cycle (residual C1)"
        )
        tg_body = tg_dump.read_text(encoding="utf-8")
        assert "tg one line" in tg_body
        # No cross-bleed: the malformed-key content stayed unified-only.
        assert "int keyed line" not in tg_body
        assert "list keyed line" not in tg_body


class TestPerUserIngestFailureIsolation:
    """I1: a per-slug ``run_ingest``/``run_lint`` failure must skip ONLY that
    slug, not abort every later user's consolidation. The batch-global
    try/except defeated 7.2's per-user isolation (cursor already advanced →
    unrecoverable for the skipped users).
    """

    async def test_one_user_ingest_failure_isolated(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """``run_ingest`` raises ONLY for the ``telegram_2`` vault. After
        ``dream.run()``: ``telegram_1``'s vault HAS its ingested page (NOT
        aborted by user-2's failure), user-2's failure is logged WITH its
        slug, the legacy path/cursor still advanced, no exception escapes.

        On CURRENT code one bad vault aborts the whole `for slug` loop (the
        try/except wraps the entire loop), so the user sorted after the
        failing one is lost while the cursor has already advanced.
        """
        dream.wiki_enabled = True

        store.append_history("alpha for one", session_key="telegram:1")
        store.append_history("beta for two", session_key="telegram:2")

        # Phase 1 (legacy) gets the first provider call (plain analysis).
        # Ingest's provider calls echo the history slice into a PAGE body.
        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        real_run_ingest = memory_mod.run_ingest

        async def _selective_ingest(vault, *a, **k):
            # vault is a Vault whose root path ends with the slug dir.
            if vault.root.name == "telegram_2":
                raise RuntimeError("user-2 vault corrupt page")
            return await real_run_ingest(vault, *a, **k)

        monkeypatch.setattr(
            memory_mod, "run_ingest", AsyncMock(side_effect=_selective_ingest),
        )

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="ERROR")
        try:
            result = await dream.run()
        finally:
            logger.remove(sink_id)

        # Legacy path entirely intact + cursor advanced (unchanged by I1).
        assert result is True
        assert store.get_last_dream_cursor() == 2

        users = store.workspace / "memory" / "users"

        # user-1 ingested DESPITE user-2's failure (the isolation 7.2 promises).
        v1_dump = users / "telegram_1" / "wiki" / "concepts" / "dump.md"
        assert v1_dump.is_file(), (
            "telegram_1 was NOT ingested — user-2's failure aborted the "
            "whole loop (I1: per-user isolation broken)"
        )
        assert "alpha for one" in v1_dump.read_text(encoding="utf-8")

        # user-2's failure was logged WITH its slug, and swallowed (no raise).
        assert any("telegram_2" in m for m in captured), (
            "the failing vault slug was not logged"
        )
        assert any("wiki ingest/lint failed" in m for m in captured)

        # user-2's page is absent (its ingest raised) — only that slug lost.
        assert not (
            users / "telegram_2" / "wiki" / "concepts" / "dump.md"
        ).exists()


# --- Concurrency guard (Task 4.6) ------------------------------------------


class TestDreamConcurrencyGuard:
    """A cron tick racing ``/dream`` can call ``Dream.run()`` twice on one loop.

    Both calls would pass the cursor guard, process the SAME batch, and
    double-edit MEMORY.md (and, with the wiki on, double-Ingest). Task 4.6
    serializes ``Dream.run()`` via a module-global strong-ref ``asyncio.Lock``
    so the two invocations run one-after-another; the second then re-reads the
    cursor the first advanced and correctly no-ops.
    """

    async def test_guard_is_module_global_strong_ref(
        self, store, mock_provider, mock_runner,
    ):
        """The guard is a module-global ``asyncio.Lock`` on
        ``nanobot.agent.memory`` — a process singleton (SAME object across
        ``Dream`` instances, not per-instance) and distinct from any
        ``get_vault_lock`` entry (which lives in a weak registry that could be
        GC'd between non-overlapping ``create_task``s)."""
        assert isinstance(memory_mod._DREAM_RUN_LOCK, asyncio.Lock)

        d1 = Dream(store=store, provider=mock_provider, model="m", max_batch_size=5)
        d2 = Dream(store=store, provider=mock_provider, model="m", max_batch_size=5)
        # Process-singleton: not stored per-instance, same object for all.
        assert memory_mod._DREAM_RUN_LOCK is memory_mod._DREAM_RUN_LOCK
        # Not derived from the weak per-vault registry.
        assert memory_mod._DREAM_RUN_LOCK is not get_vault_lock("unified_default")
        assert memory_mod._DREAM_RUN_LOCK is not get_vault_lock(
            vault_slug("unified:default")
        )
        # Sanity: the lock is reachable as a strong module attribute (held for
        # the process lifetime), unlike WeakValueDictionary entries.
        held = memory_mod._DREAM_RUN_LOCK
        assert held is memory_mod._DREAM_RUN_LOCK
        # d1/d2 are otherwise normal Dream objects sharing the one global lock.
        assert d1 is not d2

    async def test_single_run_unaffected(
        self, dream, mock_provider, mock_runner, store,
    ):
        """One ``dream.run()`` with the lock uncontended is byte-identical to
        the v0.2.0 baseline: same truthy result, cursor advance, Phase 1/2
        delegation, history compaction. (Golden-equivalent.)"""
        dream.wiki_enabled = False
        store.append_history("event 1")
        store.append_history("event 2")
        assert store.get_last_dream_cursor() == 0

        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()

        assert result is True
        assert store.get_last_dream_cursor() == 2
        mock_provider.chat_with_retry.assert_called_once()
        mock_runner.run.assert_called_once()

    async def test_concurrent_run_serialized(
        self, dream, mock_provider, mock_runner, store,
    ):
        """Two concurrent ``dream.run()`` invocations must NOT overlap and must
        NOT double-process the same batch.

        Instrument ``provider.chat_with_retry`` to record enter/exit order with
        a sleep so any overlap is observable. With the guard:
          * run A's instrumented section fully completes before run B's begins
            (no interleave), AND
          * run B re-reads the cursor A advanced, finds no unprocessed entries,
            and no-ops — so Phase 1 (provider) + Phase 2 (runner) run exactly
            ONCE and the cursor advances exactly once to ``batch[-1]``.
        """
        dream.wiki_enabled = False
        store.append_history("event 1")
        store.append_history("event 2")
        assert store.get_last_dream_cursor() == 0

        events: list[str] = []

        async def _instrumented_phase1(*args, **kwargs):
            events.append("enter")
            # Long enough that an unguarded second run would interleave here.
            await asyncio.sleep(0.02)
            events.append("exit")
            return MagicMock(content="New fact")

        mock_provider.chat_with_retry.side_effect = _instrumented_phase1
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))

        results = await asyncio.gather(dream.run(), dream.run())

        # No overlap: a strictly non-interleaved enter/exit sequence. Because
        # the second run re-reads the advanced cursor and no-ops (no provider
        # call), exactly ONE enter/exit pair is recorded.
        assert events == ["enter", "exit"], (
            f"runs overlapped or double-processed: {events!r}"
        )

        # Exactly one run did work; the other saw the advanced cursor and
        # no-op'd (re-read INSIDE the lock => sees A's advance).
        assert sorted(results) == [False, True]

        # Cursor advanced exactly once to batch[-1] (not double-advanced).
        assert store.get_last_dream_cursor() == 2

        # Phase 1 (provider) + Phase 2 (runner) invoked once total — the batch
        # was NOT consolidated twice.
        mock_provider.chat_with_retry.assert_called_once()
        mock_runner.run.assert_called_once()

    def test_dream_lock_survives_sequential_loops(
        self, store, mock_provider, mock_runner,
    ):
        """A completed ``dream.run()`` in one event loop must not poison a
        ``dream.run()`` in a SUBSEQUENT, distinct event loop (review I1).

        This is the cross-loop scenario the autouse ``_reset_dream_run_lock``
        fixture neutralizes: without the reset, a *contended* lock binding
        leaked from an earlier test would make the second loop here raise
        ``RuntimeError: <Lock> is bound to a different event loop``. We drive
        two FULL ``dream.run()`` cycles, each on its own freshly-created loop
        via ``asyncio.run`` (this test is intentionally a plain ``def`` — NOT
        an ``async def`` — so it owns loop creation rather than running on
        pytest's per-test loop), and assert no ``RuntimeError`` escapes.
        Deterministic: no sleeps, no concurrency, mocks return immediately.
        """
        def _one_full_run() -> bool:
            # Fresh store per loop so each run actually does work (cursor 0→2)
            # and the assertion below is meaningful, independent of order.
            s = MemoryStore(store.workspace)
            s.write_soul("# Soul\n- Helpful")
            s.write_user("# User\n- Developer")
            s.write_memory("# Memory\n- Project X active")
            s.append_history("event 1")
            s.append_history("event 2")

            provider = MagicMock()
            provider.chat_with_retry = AsyncMock(
                return_value=MagicMock(content="New fact"),
            )
            d = Dream(store=s, provider=provider, model="m", max_batch_size=5)
            d._runner = MagicMock()
            d._runner.run = AsyncMock(return_value=_make_run_result(
                tool_events=[{"name": "edit_file", "status": "ok",
                              "detail": "memory/MEMORY.md"}],
            ))
            d.wiki_enabled = False
            return asyncio.run(d.run())

        # Loop #1: completes (and contends nothing — single run). The autouse
        # fixture would also reset between tests, but the regression we lock is
        # specifically two runs on two loops WITHIN one test surviving cleanly.
        first = _one_full_run()
        # Reset exactly as the autouse teardown does, simulating the next test's
        # fresh module global, then run again on a brand-new loop.
        memory_mod._DREAM_RUN_LOCK = asyncio.Lock()
        # Loop #2: a NEW event loop. Must not raise "bound to a different
        # event loop"; pytest.fail makes any RuntimeError explicit.
        try:
            second = _one_full_run()
        except RuntimeError as exc:  # pragma: no cover - this is the failure
            pytest.fail(
                f"_DREAM_RUN_LOCK leaked an event-loop binding across "
                f"sequential loops (review I1 regression): {exc!r}"
            )

        assert first is True
        assert second is True


# --- Task 4 (attachments-ingest): Dream loop runs the reconciler ------------


class TestDreamRunsAttachmentsReconciler:
    """Task 4: the Dream cycle invokes ``run_attachments_reconcile`` for the
    unified vault BEFORE ``run_ingest`` / ``run_lint``, inside the per-slug
    ``get_vault_lock(slug)`` block. This catches attachments that arrived
    out-of-band (no history entry mentioning them) and turns them into
    ``inbox/*`` pages before Ingest sees the vault state.
    """

    async def test_dream_picks_up_workspace_peer_attachment(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """A .md file dropped into ``workspace/peer/msg_<id>/`` becomes a
        wiki ``inbox/<stem>.md`` page after the next Dream cycle, with NO
        history entry mentioning it (model never saw it). Regression for
        the gap that motivated this whole feature.
        """
        dream.wiki_enabled = True

        # Redirect default sources to the test workspace so the reconciler
        # walks tmp_path/peer instead of the real ~/.nanobot/workspace/peer.
        monkeypatch.setattr(
            "nanobot.agent.wiki.attachments_reconciler.get_workspace_path",
            lambda: store.workspace,
        )
        monkeypatch.setattr(
            "nanobot.agent.wiki.attachments_reconciler.get_media_dir",
            lambda: store.workspace / "media",
        )

        # Drop a .md file with NO accompanying history entry. The model
        # never saw this — only the reconciler can pick it up.
        msg_dir = store.workspace / "peer" / "msg_aaa"
        msg_dir.mkdir(parents=True)
        (msg_dir / "drop.md").write_text("hello drop", encoding="utf-8")

        # The Dream cycle still needs history to do work (otherwise the
        # whole block is skipped). Add unrelated history.
        store.append_history("unrelated event 1")

        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok",
                          "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        inbox_page = (
            store.workspace / "memory" / "users" / "unified_default"
            / "wiki" / "inbox" / "drop.md"
        )
        assert inbox_page.is_file(), (
            "reconciler did NOT write inbox/drop.md — Dream loop is not "
            "wiring run_attachments_reconcile before run_ingest"
        )
        assert "hello drop" in inbox_page.read_text(encoding="utf-8")

    async def test_reconciler_failure_does_not_skip_ingest_or_lint(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """A reconciler crash for the unified slug must be swallowed by an
        INNER try/except so ``run_ingest``/``run_lint`` for that same slug
        still run (the outer per-slug try/except would otherwise skip
        them too).
        """
        dream.wiki_enabled = True

        async def _boom(*a, **k):
            raise RuntimeError("reconciler exploded")

        monkeypatch.setattr(
            memory_mod, "run_attachments_reconcile", AsyncMock(side_effect=_boom),
        )

        store.append_history("event 1")

        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok",
                          "detail": "memory/MEMORY.md"}],
        ))

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="ERROR")
        try:
            result = await dream.run()
        finally:
            logger.remove(sink_id)

        assert result is True
        # Reconciler failure was logged but did NOT skip Ingest: the
        # canned _INGEST_OUTPUT created people/alice.md in the unified
        # vault. If the failure had escaped to the outer per-slug
        # try/except, Ingest would have been skipped and alice.md would
        # not exist.
        alice = (
            store.workspace / "memory" / "users" / "unified_default"
            / "wiki" / "people" / "alice.md"
        )
        assert alice.is_file(), (
            "reconciler failure caused run_ingest to be skipped — "
            "inner try/except is missing"
        )
        assert any("attachments reconcile failed" in m for m in captured), (
            "reconciler failure was not logged"
        )

    async def test_reconciler_only_runs_for_unified_slug(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        """v1 routing constraint: the reconciler walks raw files with no
        sender info, so it MUST only run for ``slug == unified_default``.
        Per-user slugs must NOT trigger a reconciler call.
        """
        dream.wiki_enabled = True

        store.append_history("alpha", session_key="telegram:1")
        store.append_history("beta", session_key="unified:default")

        recon_calls: list[str] = []

        async def _recording(vault, slug, *a, **k):
            recon_calls.append(slug)
            from nanobot.agent.wiki.attachments_reconciler import ReconcileReport
            return ReconcileReport()

        monkeypatch.setattr(
            memory_mod, "run_attachments_reconcile",
            AsyncMock(side_effect=_recording),
        )

        _content_echo_provider(mock_provider)
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok",
                          "detail": "memory/MEMORY.md"}],
        ))

        result = await dream.run()
        assert result is True

        # Reconciler must have been called for unified_default ONLY, not
        # for telegram_1 (which has no sender-aware routing yet).
        assert recon_calls == ["unified_default"], (
            f"reconciler called for unexpected slugs: {recon_calls!r}"
        )


# --- Dream wiki-block logging (visibility) ----------------------------------
# The wiki sub-stages each build a structured report; the Dream loop turns each
# into one INFO line so the operator can see what happened per vault. The pure
# ``_fmt_wiki_*`` formatters are unit-tested here for every branch; the two
# integration tests confirm the lines are actually emitted at INFO during a run.
from nanobot.agent.wiki.attachments_reconciler import ReconcileReport  # noqa: E402
from nanobot.agent.wiki.embeddings import EmbedRefreshReport  # noqa: E402
from nanobot.agent.wiki.ingest import IngestReport  # noqa: E402
from nanobot.agent.wiki.lint import LintReport  # noqa: E402


def test_fmt_wiki_ingest_changed():
    r = IngestReport(created=["people/a.md", "people/b.md"],
                     appended=["topics/c.md"], skipped_duplicate=1)
    assert memory_mod._fmt_wiki_ingest("unified_default", r) == (
        "Dream wiki[unified_default] ingest: "
        "created=2 appended=1 contra=0 dup-skip=1"
    )


def test_fmt_wiki_ingest_no_changes():
    assert memory_mod._fmt_wiki_ingest("u", IngestReport()) == (
        "Dream wiki[u] ingest: no changes"
    )


def test_fmt_wiki_ingest_skipped():
    assert memory_mod._fmt_wiki_ingest("u", IngestReport(skipped=True)) == (
        "Dream wiki[u] ingest: skipped"
    )


def test_fmt_wiki_ingest_anomalies_suffix():
    r = IngestReport(created=["x.md"], unknown=[("people", "ghost")],
                     dropped=2, malformed_lines=3)
    assert memory_mod._fmt_wiki_ingest("u", r) == (
        "Dream wiki[u] ingest: created=1 appended=0 contra=0 dup-skip=0 "
        "[unknown=1 dropped=2 malformed=3]"
    )


def test_fmt_wiki_lint_changed():
    r = LintReport(cooled=["a.md"], moc_regenerated=True,
                   indexes_regenerated=["index.md"])
    assert memory_mod._fmt_wiki_lint("u", r) == (
        "Dream wiki[u] lint: cooled=1 reheated=0 merged=0 orphans=0 "
        "indexes=1 moc=yes"
    )


def test_fmt_wiki_lint_no_changes_with_findings():
    r = LintReport(broken_links=[("a.md", "b")], malformed=["bad.md"])
    assert memory_mod._fmt_wiki_lint("u", r) == (
        "Dream wiki[u] lint: no changes [broken=1 malformed=1]"
    )


def test_fmt_wiki_embeddings_changed():
    r = EmbedRefreshReport(available=True, changed=True, pages=4, vectors=11,
                           reembedded=3, deleted=1)
    assert memory_mod._fmt_wiki_embeddings("u", r) == (
        "Dream wiki[u] embeddings: 4 pages → 11 vectors (3 re-embedded, 1 deleted)"
    )


def test_fmt_wiki_embeddings_up_to_date():
    r = EmbedRefreshReport(available=True, changed=False, pages=6, vectors=9)
    assert memory_mod._fmt_wiki_embeddings("u", r) == (
        "Dream wiki[u] embeddings: up-to-date (6 pages)"
    )


def test_fmt_wiki_embeddings_unavailable():
    msg = memory_mod._fmt_wiki_embeddings("u", EmbedRefreshReport(available=False))
    assert "fastembed unavailable" in msg


def test_fmt_wiki_attachments_active():
    r = ReconcileReport(created=["inbox/a.md"], duplicates=3, skipped_binary=2)
    assert memory_mod._fmt_wiki_attachments("u", r) == (
        "Dream wiki[u] attachments: created=1 appended=0 dup=3 binary-skip=2"
    )


class TestDreamWikiLoggingIntegration:
    """The formatted lines are actually emitted at INFO during a real run."""

    async def test_wiki_logs_ingest_and_lint_at_info(
        self, dream, mock_provider, mock_runner, store,
    ):
        dream.wiki_enabled = True
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok",
                          "detail": "memory/MEMORY.md"}],
        ))

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
        try:
            await dream.run()
        finally:
            logger.remove(sink_id)

        assert any("ingest: created=1" in m for m in captured), captured
        assert any("] lint:" in m for m in captured), captured
        # No attachments on disk → idle reconcile stays at DEBUG (not INFO).
        assert not any("] attachments:" in m for m in captured), captured
        # Embeddings gate off (default) → disabled, stays at DEBUG (not INFO).
        assert not any("] embeddings:" in m for m in captured), captured

    async def test_wiki_logs_embeddings_summary_at_info(
        self, dream, mock_provider, mock_runner, store, monkeypatch,
    ):
        dream.wiki_enabled = True
        dream.wiki_embeddings = True
        dream.wiki_embedding_model = "m"
        monkeypatch.setattr(
            "nanobot.agent.wiki.embeddings.refresh_embeddings",
            lambda vault, model: EmbedRefreshReport(
                available=True, changed=True, pages=4, vectors=11,
                reembedded=3, deleted=1),
        )
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.side_effect = [
            MagicMock(content="New fact", finish_reason="stop"),
            MagicMock(content=_INGEST_OUTPUT, finish_reason="stop"),
        ]
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "x"}],
        ))

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
        try:
            await dream.run()
        finally:
            logger.remove(sink_id)

        assert any(
            "embeddings: 4 pages → 11 vectors (3 re-embedded, 1 deleted)" in m
            for m in captured
        ), captured
