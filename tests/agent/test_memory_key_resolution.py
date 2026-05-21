"""CV2 tests: memory_key resolution decouples vault from chat session.

Covers the new ``unified_memory`` flag added to :class:`AgentDefaults`:

* ``unified_memory=True`` collapses memory_key to ``UNIFIED_SESSION_KEY``
  while leaving session_key per-channel — i.e. per-user chat sessions
  sharing one wiki vault.
* Legacy ``unified_session=True`` keeps the pre-CV2 collapse of BOTH keys.
* Default (both flags off) keeps memory_key == session_key (byte-identical
  back-compat).
* ``memory_key_override`` on InboundMessage wins over flag-driven derivation.
* :meth:`AgentLoop._set_tool_context` populates ``RequestContext.memory_key``
  consistently with ``_effective_memory_key``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import UNIFIED_SESSION_KEY, AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


# ---------------------------------------------------------------------------
# Helpers (mirror tests/agent/test_unified_session.py)
# ---------------------------------------------------------------------------

def _make_loop(
    tmp_path: Path,
    *,
    unified_session: bool = False,
    unified_memory: bool = False,
) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr, \
         patch("nanobot.agent.loop.Dream"):
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            unified_session=unified_session,
            unified_memory=unified_memory,
        )
    return loop


def _make_msg(
    channel: str = "telegram",
    chat_id: str = "111",
    *,
    session_key_override: str | None = None,
    memory_key_override: str | None = None,
) -> InboundMessage:
    return InboundMessage(
        channel=channel,
        chat_id=chat_id,
        sender_id="user1",
        content="hello",
        session_key_override=session_key_override,
        memory_key_override=memory_key_override,
    )


# ---------------------------------------------------------------------------
# _effective_memory_key — 4-cell truth table
# ---------------------------------------------------------------------------

class TestEffectiveMemoryKey:
    def test_both_flags_off_returns_session_key(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        msg = _make_msg(channel="telegram", chat_id="111")
        session_key = "telegram:111"
        assert loop._effective_memory_key(msg, session_key) == session_key

    def test_unified_memory_only_collapses_memory(self, tmp_path: Path):
        loop = _make_loop(tmp_path, unified_memory=True)
        msg = _make_msg(channel="telegram", chat_id="111")
        # session_key stays per-channel
        assert loop._effective_session_key(msg) == "telegram:111"
        # memory_key collapses to UNIFIED_SESSION_KEY
        assert loop._effective_memory_key(msg, "telegram:111") == UNIFIED_SESSION_KEY

    def test_legacy_unified_session_also_collapses_memory(self, tmp_path: Path):
        loop = _make_loop(tmp_path, unified_session=True)
        msg = _make_msg(channel="telegram", chat_id="111")
        # Pre-CV2: both keys collapsed.
        assert loop._effective_session_key(msg) == UNIFIED_SESSION_KEY
        assert loop._effective_memory_key(msg, UNIFIED_SESSION_KEY) == UNIFIED_SESSION_KEY

    def test_both_flags_on_collapses_memory(self, tmp_path: Path):
        loop = _make_loop(tmp_path, unified_session=True, unified_memory=True)
        msg = _make_msg(channel="telegram", chat_id="111")
        assert loop._effective_memory_key(msg, UNIFIED_SESSION_KEY) == UNIFIED_SESSION_KEY


# ---------------------------------------------------------------------------
# memory_key_override on InboundMessage wins
# ---------------------------------------------------------------------------

class TestMemoryKeyOverride:
    def test_override_wins_over_default(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        msg = _make_msg(memory_key_override="custom:vault")
        assert loop._effective_memory_key(msg, "telegram:111") == "custom:vault"

    def test_override_wins_over_unified_memory(self, tmp_path: Path):
        loop = _make_loop(tmp_path, unified_memory=True)
        msg = _make_msg(memory_key_override="custom:vault")
        # Despite unified_memory=true, an explicit override takes precedence.
        assert loop._effective_memory_key(msg, "telegram:111") == "custom:vault"


# ---------------------------------------------------------------------------
# _set_tool_context populates RequestContext.memory_key
# ---------------------------------------------------------------------------

class TestSetToolContextPropagatesMemoryKey:
    def _capture_ctx(self, loop: AgentLoop):
        """Replace tools.set with a spy that captures the RequestContext."""
        captured: dict[str, object] = {}

        # Walk the ContextAware tools and intercept set_context.
        for name in list(loop.tools.tool_names):
            tool = loop.tools.get(name)
            if tool is None:
                continue
            if hasattr(tool, "set_context"):
                original = tool.set_context

                def spy(ctx, _orig=original, _name=name):
                    captured[_name] = ctx
                    return _orig(ctx)

                tool.set_context = spy  # type: ignore[assignment]
        return captured

    def test_default_memory_key_equals_session_key(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        captured = self._capture_ctx(loop)
        loop._set_tool_context("telegram", "111")
        for ctx in captured.values():
            assert ctx.session_key == "telegram:111"
            assert ctx.memory_key == "telegram:111"

    def test_unified_memory_diverges_memory_from_session(self, tmp_path: Path):
        loop = _make_loop(tmp_path, unified_memory=True)
        captured = self._capture_ctx(loop)
        loop._set_tool_context("telegram", "111")
        for ctx in captured.values():
            assert ctx.session_key == "telegram:111"
            assert ctx.memory_key == UNIFIED_SESSION_KEY

    def test_explicit_memory_key_arg_wins(self, tmp_path: Path):
        loop = _make_loop(tmp_path)  # no flags
        captured = self._capture_ctx(loop)
        loop._set_tool_context("telegram", "111", memory_key="explicit:vault")
        for ctx in captured.values():
            assert ctx.session_key == "telegram:111"
            assert ctx.memory_key == "explicit:vault"


# ---------------------------------------------------------------------------
# AgentDefaults schema: warning when both flags are set
# ---------------------------------------------------------------------------

class TestSchemaWarning:
    def test_both_flags_emits_warning(self, caplog):
        from nanobot.config.schema import AgentDefaults

        with caplog.at_level("WARNING"):
            AgentDefaults(unified_session=True, unified_memory=True)
        assert any(
            "unified_session=true implies unified_memory" in record.message
            for record in caplog.records
        )

    def test_only_unified_memory_silent(self, caplog):
        from nanobot.config.schema import AgentDefaults

        with caplog.at_level("WARNING"):
            AgentDefaults(unified_memory=True)
        # No warning expected.
        assert not any("unified_session" in record.message for record in caplog.records)
