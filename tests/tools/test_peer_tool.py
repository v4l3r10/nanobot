"""Unit tests for the inter-agent peer tools (peer_say, peer_list)."""

from __future__ import annotations

import json

import pytest

from nanobot.agent.tools.peer import PeerListTool, PeerSayTool
from nanobot.bus.events import OutboundMessage


def _accept_all(_value: str) -> bool:
    return True


@pytest.mark.asyncio
async def test_peer_say_constructs_outbound_message() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = PeerSayTool(
        send_callback=_send,
        own_agent_id="bronzo",
        peer_validator=_accept_all,
    )
    result = await tool.execute(to="grocco", text="ciao")

    assert "Sent to peer grocco" in result
    assert len(sent) == 1
    msg = sent[0]
    assert msg.channel == "peer"
    assert msg.chat_id == "peer:grocco"
    assert msg.content == "ciao"


@pytest.mark.asyncio
async def test_peer_say_threads_metadata() -> None:
    """thread_id and in_reply_to ride through OutboundMessage.metadata so the
    PeerChannel can serialize them into the wire frame."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    await tool.execute(
        to="grocco", text="re: parlavi di X",
        thread_id="thr_X", in_reply_to="msg_PREV",
    )
    assert sent[0].metadata == {
        "peer_thread_id": "thr_X",
        "peer_in_reply_to": "msg_PREV",
    }


@pytest.mark.asyncio
async def test_peer_say_refuses_self() -> None:
    async def _send(_msg: OutboundMessage) -> None:
        raise AssertionError("send_callback should not be called for self-send")

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    result = await tool.execute(to="bronzo", text="hi me")
    assert "cannot peer_say to self" in result


@pytest.mark.asyncio
async def test_peer_say_refuses_invalid_id() -> None:
    """Validator rejects names that would not survive the router-side regex."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = PeerSayTool(
        send_callback=_send,
        own_agent_id="bronzo",
        peer_validator=lambda v: v.isalpha() and v.islower(),
    )
    result = await tool.execute(to="Bad-ID", text="hi")
    assert "invalid peer agent id" in result
    assert sent == []


@pytest.mark.asyncio
async def test_peer_say_propagates_send_failure() -> None:
    async def _send(_msg: OutboundMessage) -> None:
        raise RuntimeError("router unreachable")

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    result = await tool.execute(to="grocco", text="hi")
    assert "Error sending peer message" in result
    assert "router unreachable" in result


@pytest.mark.asyncio
async def test_peer_say_passes_media_through() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    await tool.execute(
        to="grocco", text="ti mando il pacchetto",
        media=["/tmp/file.bin"],
    )
    assert sent[0].media == ["/tmp/file.bin"]


@pytest.mark.asyncio
async def test_peer_list_filters_self_and_returns_json() -> None:
    tool = PeerListTool(
        list_peers_callback=lambda: ["bronzo", "grocco", "naldo"],
        own_agent_id="bronzo",
    )
    result = await tool.execute()
    payload = json.loads(result)
    ids = sorted(p["id"] for p in payload)
    assert ids == ["grocco", "naldo"]
    assert all(p["online"] is True for p in payload)


@pytest.mark.asyncio
async def test_peer_list_empty_callback() -> None:
    tool = PeerListTool(
        list_peers_callback=lambda: [],
        own_agent_id="bronzo",
    )
    result = await tool.execute()
    assert json.loads(result) == []
