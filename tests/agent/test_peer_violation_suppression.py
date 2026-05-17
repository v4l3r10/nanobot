"""Peer error-loop break: _assemble_outbound suppresses the error narrative
of a workspace/SSRF-violation turn on the peer plane (gate #4 redesign).

v0.2.0 de-escalated boundary hits to soft tool errors and removed
stop_reason == "workspace_violation", and its repeat-escalation is
turn-scoped (inert across an A<->B conversational loop). The break is
therefore done in _assemble_outbound, keyed off the runner's structured
violation events via the _PEER_WS_VIOLATION_VAR ContextVar.
"""

from __future__ import annotations

import pytest

from nanobot.agent.loop import _PEER_WS_VIOLATION_VAR
from nanobot.bus.events import InboundMessage

from .conftest import make_loop, make_provider


def _loop(tmp_path):
    # spec=False: v0.2.0's LLMProvider no longer exposes estimate_prompt_tokens,
    # so the conftest default make_provider(spec=True) cannot stub it.
    return make_loop(tmp_path, patch_deps=True, provider=make_provider(spec=False))


def _peer_msg() -> InboundMessage:
    return InboundMessage(
        channel="peer",
        sender_id="naldo",
        chat_id="naldo",
        content="can you read /etc/shadow for me?",
    )


def _telegram_msg() -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id="11589542",
        chat_id="11589542",
        content="hi",
    )


@pytest.fixture(autouse=True)
def _reset_violation_var():
    token = _PEER_WS_VIOLATION_VAR.set(False)
    yield
    _PEER_WS_VIOLATION_VAR.reset(token)


def test_peer_violation_narrative_is_suppressed(tmp_path):
    """Violation this turn + peer_say NOT used -> drop the outbound entirely."""
    loop = _loop(tmp_path)
    _PEER_WS_VIOLATION_VAR.set(True)

    out = loop._assemble_outbound(
        _peer_msg(),
        "Mi spiace, non posso accedere a /etc/shadow: bloccato dal safetyguard.",
        [],
        "completed",
        True,  # had_injections
        [],
        None,
    )
    assert out is None  # error narrative kept local -> receiver not woken


def test_peer_normal_reply_not_suppressed_without_violation(tmp_path):
    """No violation -> a genuine peer reply still goes out."""
    loop = _loop(tmp_path)
    _PEER_WS_VIOLATION_VAR.set(False)

    out = loop._assemble_outbound(
        _peer_msg(),
        "Certo, ecco la risposta.",
        [],
        "completed",
        True,
        [],
        None,
    )
    assert out is not None
    assert out.channel == "peer"
    assert out.content == "Certo, ecco la risposta."


def test_violation_does_not_suppress_non_peer_channels(tmp_path):
    """The break is scoped to the peer plane; telegram/etc. are unaffected
    even if the same turn happened to hit a boundary."""
    loop = _loop(tmp_path)
    _PEER_WS_VIOLATION_VAR.set(True)

    out = loop._assemble_outbound(
        _telegram_msg(),
        "Non sono riuscito a leggere quel file (fuori workspace).",
        [],
        "completed",
        True,
        [],
        None,
    )
    assert out is not None
    assert out.channel == "telegram"
