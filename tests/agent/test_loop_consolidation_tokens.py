from unittest.mock import AsyncMock, MagicMock

import pytest

import nanobot.agent.memory as memory_module
from nanobot.agent.loop import AgentLoop
from nanobot.agent.wiki.paths import vault_dir
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(
    tmp_path,
    *,
    estimated_tokens: int,
    context_window_tokens: int,
    wiki_enabled: bool = False,
) -> AgentLoop:
    from nanobot.providers.base import GenerationSettings
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (estimated_tokens, "test-counter")
    _response = LLMResponse(content="ok", tool_calls=[])
    provider.chat_with_retry = AsyncMock(return_value=_response)
    provider.chat_stream_with_retry = AsyncMock(return_value=_response)

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=context_window_tokens,
        wiki_enabled=wiki_enabled,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator._SAFETY_BUFFER = 0
    return loop


@pytest.mark.asyncio
async def test_prompt_below_threshold_does_not_consolidate(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await loop.process_direct("hello", session_key="cli:test")

    loop.consolidator.archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_above_threshold_triggers_consolidation(tmp_path, monkeypatch) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=1000, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value=True)  # type: ignore[method-assign]
    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
    ]
    loop.sessions.save(session)
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _message: 500)

    await loop.process_direct("hello", session_key="cli:test")

    assert loop.consolidator.archive.await_count >= 1


@pytest.mark.asyncio
async def test_prompt_above_threshold_archives_until_next_user_boundary(tmp_path, monkeypatch) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=1000, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value=True)  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
        {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
    ]
    loop.sessions.save(session)

    token_map = {"u1": 120, "a1": 120, "u2": 120, "a2": 120, "u3": 120}
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda message: token_map[message["content"]])

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    archived_chunk = loop.consolidator.archive.await_args.args[0]
    assert [message["content"] for message in archived_chunk] == ["u1", "a1", "u2", "a2"]
    assert session.last_consolidated == 4


@pytest.mark.asyncio
async def test_consolidation_loops_until_target_met(tmp_path, monkeypatch) -> None:
    """Verify maybe_consolidate_by_tokens keeps looping until under threshold."""
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value=True)  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
        {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
        {"role": "assistant", "content": "a3", "timestamp": "2026-01-01T00:00:05"},
        {"role": "user", "content": "u4", "timestamp": "2026-01-01T00:00:06"},
    ]
    loop.sessions.save(session)

    call_count = [0]
    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500, "test")
        if call_count[0] == 2:
            return (300, "test")
        return (80, "test")

    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 100)

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    assert loop.consolidator.archive.await_count == 2
    assert session.last_consolidated == 6


@pytest.mark.asyncio
async def test_consolidation_continues_below_trigger_until_half_target(tmp_path, monkeypatch) -> None:
    """Once triggered, consolidation should continue until it drops below half threshold."""
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value=True)  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
        {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
        {"role": "assistant", "content": "a3", "timestamp": "2026-01-01T00:00:05"},
        {"role": "user", "content": "u4", "timestamp": "2026-01-01T00:00:06"},
    ]
    loop.sessions.save(session)

    call_count = [0]

    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500, "test")
        if call_count[0] == 2:
            return (150, "test")
        return (80, "test")

    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 100)

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    assert loop.consolidator.archive.await_count == 2
    assert session.last_consolidated == 6


@pytest.mark.asyncio
async def test_consolidation_persists_summary_for_next_prepare_session(tmp_path, monkeypatch) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)
    loop.consolidator.archive = AsyncMock(return_value="User discussed project status.")  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
    ]
    loop.sessions.save(session)

    call_count = [0]

    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (500, "test")
        return (80, "test")

    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 150)

    await loop.consolidator.maybe_consolidate_by_tokens(session)

    reloaded = loop.sessions.get_or_create("cli:test")
    meta = reloaded.metadata.get("_last_summary")
    assert meta is not None
    assert meta["text"] == "User discussed project status."

    reloaded, pending = loop.auto_compact.prepare_session(reloaded, "cli:test")
    assert pending is not None
    assert "User discussed project status." in pending
    # _last_summary persists for restart survival.
    assert "_last_summary" in reloaded.metadata


@pytest.mark.asyncio
async def test_preflight_consolidation_receives_pending_summary(tmp_path) -> None:
    loop = _make_loop(tmp_path, estimated_tokens=100, context_window_tokens=200)
    session = loop.sessions.get_or_create("cli:test")
    loop.auto_compact.prepare_session = MagicMock(
        return_value=(session, "Previous conversation summary: earlier context")
    )  # type: ignore[method-assign]
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)  # type: ignore[method-assign]
    loop._schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]

    await loop.process_direct("hello", session_key="cli:test")

    loop.consolidator.maybe_consolidate_by_tokens.assert_any_await(
        session,
        replay_max_messages=loop._max_messages,
    )


@pytest.mark.asyncio
async def test_preflight_consolidation_before_llm_call(tmp_path, monkeypatch) -> None:
    """Verify preflight consolidation runs before the LLM call in process_direct."""
    order: list[str] = []

    loop = _make_loop(tmp_path, estimated_tokens=0, context_window_tokens=200)

    async def track_consolidate(messages, **kwargs):
        order.append("consolidate")
        return True
    loop.consolidator.archive = track_consolidate  # type: ignore[method-assign]

    async def track_llm(*args, **kwargs):
        order.append("llm")
        return LLMResponse(content="ok", tool_calls=[])
    loop.provider.chat_with_retry = track_llm
    loop.provider.chat_stream_with_retry = track_llm
    loop._schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
        {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
    ]
    loop.sessions.save(session)
    monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 500)

    call_count = [0]
    def mock_estimate(_session, *, session_summary=None):
        call_count[0] += 1
        return (1000 if call_count[0] <= 1 else 80, "test")
    loop.consolidator.estimate_session_prompt_tokens = mock_estimate  # type: ignore[method-assign]

    await loop.process_direct("hello", session_key="cli:test")

    assert "consolidate" in order
    assert "llm" in order
    assert order.index("consolidate") < order.index("llm")


# ---------------------------------------------------------------------------
# I1 — the Consolidator token probe must build the SAME prompt the real turn
# uses. When wiki is enabled the real turn injects the per-user vault MOC +
# the smaller history tail; the probe must too, else auto-consolidation is
# estimated against the wrong (wiki-off) prompt.
# ---------------------------------------------------------------------------

VAULT_MOC = "# MOC\n\n## Recent\n- [[note-alpha]] VAULT-ONLY-PROBE-FACT\n"
GLOBAL_MEMORY = "GLOBAL-MEMORY-PROBE-FACT: the sky is teal."


def _probe_system_prompt(loop) -> str:
    """The system prompt the probe actually built (captured from the
    provider token-counter call args)."""
    args, _ = loop.provider.estimate_prompt_tokens.call_args
    probe_messages = args[0]
    assert probe_messages[0]["role"] == "system"
    return probe_messages[0]["content"]


def _seed_global_memory(tmp_path) -> None:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text(GLOBAL_MEMORY, encoding="utf-8")


def _write_vault_moc(tmp_path, session_key: str, content: str) -> None:
    vdir = vault_dir(tmp_path, session_key)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "MEMORY.md").write_text(content, encoding="utf-8")


class TestConsolidatorProbeMatchesRealPrompt:
    def test_probe_uses_vault_moc_when_wiki_enabled(self, tmp_path) -> None:
        """wiki_enabled=True ⇒ probe builds the prompt from the VAULT MOC,
        NOT the global MEMORY.md (matches the real turn path)."""
        loop = _make_loop(
            tmp_path,
            estimated_tokens=123,
            context_window_tokens=200,
            wiki_enabled=True,
        )
        session = loop.sessions.get_or_create("cli:test")
        _seed_global_memory(tmp_path)
        _write_vault_moc(tmp_path, session.key, VAULT_MOC)

        tokens, _src = loop.consolidator.estimate_session_prompt_tokens(session)

        prompt = _probe_system_prompt(loop)
        assert "VAULT-ONLY-PROBE-FACT" in prompt
        assert "GLOBAL-MEMORY-PROBE-FACT" not in prompt
        assert tokens == 123

    def test_probe_estimate_differs_from_wiki_off(self, tmp_path) -> None:
        """The vault MOC vs global MEMORY.md differ in size ⇒ the wiki-on
        probe and a wiki-off probe build different prompts (the probe is no
        longer estimating the wrong prompt)."""
        big_moc = "# MOC\n\n" + ("X" * 5000)

        on = _make_loop(
            tmp_path / "on",
            estimated_tokens=1,
            context_window_tokens=200,
            wiki_enabled=True,
        )
        son = on.sessions.get_or_create("cli:test")
        _seed_global_memory(tmp_path / "on")
        _write_vault_moc(tmp_path / "on", son.key, big_moc)
        on.consolidator.estimate_session_prompt_tokens(son)
        on_prompt = _probe_system_prompt(on)

        off = _make_loop(
            tmp_path / "off",
            estimated_tokens=1,
            context_window_tokens=200,
            wiki_enabled=False,
        )
        soff = off.sessions.get_or_create("cli:test")
        _seed_global_memory(tmp_path / "off")
        _write_vault_moc(tmp_path / "off", soff.key, big_moc)
        off.consolidator.estimate_session_prompt_tokens(soff)
        off_prompt = _probe_system_prompt(off)

        assert on_prompt != off_prompt
        assert "X" * 5000 in on_prompt          # wiki-on: vault MOC
        assert "X" * 5000 not in off_prompt     # wiki-off: global MEMORY.md
        assert GLOBAL_MEMORY in off_prompt

    def test_probe_unchanged_when_wiki_disabled(self, tmp_path) -> None:
        """Regression guard: wiki_enabled=False ⇒ probe builds the identical
        wiki-off prompt (global MEMORY.md, no vault), no behavior change."""
        loop = _make_loop(
            tmp_path,
            estimated_tokens=77,
            context_window_tokens=200,
            wiki_enabled=False,
        )
        session = loop.sessions.get_or_create("cli:test")
        _seed_global_memory(tmp_path)
        # A vault exists but wiki is OFF — it must be ignored entirely.
        _write_vault_moc(tmp_path, session.key, VAULT_MOC)

        tokens, _src = loop.consolidator.estimate_session_prompt_tokens(session)

        prompt = _probe_system_prompt(loop)
        assert "GLOBAL-MEMORY-PROBE-FACT" in prompt
        assert "VAULT-ONLY-PROBE-FACT" not in prompt
        assert tokens == 77
