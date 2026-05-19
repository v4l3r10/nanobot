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
