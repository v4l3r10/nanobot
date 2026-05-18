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

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import Dream, MemoryStore
from nanobot.agent.runner import AgentRunResult

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
