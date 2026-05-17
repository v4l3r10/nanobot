"""Unit tests for the inter-agent peer tools (peer_say, peer_list, peer_thread_show)."""

from __future__ import annotations

import json

import pytest

from nanobot.agent.tools.peer import PeerListTool, PeerSayTool, PeerThreadShowTool
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
    # chat_id is the bare peer agent_id (no "peer:" prefix). Channel field
    # already disambiguates routing; the prefix was redundant self-tagging.
    assert msg.chat_id == "grocco"
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
async def test_peer_say_propagates_closing_in_metadata() -> None:
    """closing=true must arrive in OutboundMessage.metadata so PeerChannel
    can serialize it into the wire frame's 'closing' field."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    result = await tool.execute(
        to="grocco", text="task done", closing=True,
    )
    assert "[closing]" in result  # surfaces in tool output
    assert sent[0].metadata.get("peer_closing") is True


@pytest.mark.asyncio
async def test_peer_say_marks_sent_in_turn_after_success() -> None:
    """The agent loop relies on this flag to suppress its narrative
    final_content emission. Without it, after peer_say closes a thread,
    the final_content goes out as a second non-closing message and
    re-wakes the receiver."""
    async def _send(msg: OutboundMessage) -> None:
        return None

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    tool.start_turn()
    assert tool._sent_in_turn is False
    await tool.execute(to="grocco", text="task done", closing=True)
    assert tool._sent_in_turn is True


@pytest.mark.asyncio
async def test_peer_say_does_not_mark_sent_on_validation_error() -> None:
    """Validation errors must not count as 'sent in turn' — the loop's
    final_content should still be emitted in those cases."""
    async def _send(msg: OutboundMessage) -> None:
        raise AssertionError("send must not be called on validation error")

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    tool.start_turn()
    await tool.execute(to="bronzo", text="hi me")  # self-send, rejected
    assert tool._sent_in_turn is False


@pytest.mark.asyncio
async def test_peer_say_does_not_mark_sent_on_send_failure() -> None:
    """A send_callback exception means nothing actually went out — the
    flag must stay False so final_content can fill in if appropriate."""
    async def _send(msg: OutboundMessage) -> None:
        raise RuntimeError("router unreachable")

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    tool.start_turn()
    result = await tool.execute(to="grocco", text="hi")
    assert "Error sending" in result
    assert tool._sent_in_turn is False


@pytest.mark.asyncio
async def test_peer_say_default_closing_false_omits_metadata() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = PeerSayTool(send_callback=_send, own_agent_id="bronzo")
    await tool.execute(to="grocco", text="ciao")
    assert sent[0].metadata.get("peer_closing") is None


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


# --- peer_thread_show -------------------------------------------------------

def _msg(from_a: str, to_a: str, text: str, ts: str) -> dict:
    return {
        "id": f"msg_{ts.replace(':','').replace('-','')}",
        "from": from_a, "to": to_a, "text": text, "ts": ts,
        "thread_id": "thr_x", "in_reply_to": None, "delivered": True,
    }


@pytest.mark.asyncio
async def test_peer_thread_show_renders_chronological_transcript() -> None:
    captured_args = {}

    async def _fetch(peer, *, last_n):
        captured_args["peer"] = peer
        captured_args["last_n"] = last_n
        return [
            _msg("bronzo", "grocco", "Ciao Grocco", "2026-05-08T17:42:34.000Z"),
            _msg("grocco", "bronzo", "Ciao Bronzo!", "2026-05-08T17:42:47.000Z"),
            _msg("bronzo", "grocco", "Tutto a posto?", "2026-05-08T17:43:03.000Z"),
        ]

    tool = PeerThreadShowTool(
        fetch_thread_callback=_fetch,
        own_agent_id="bronzo",
        peer_validator=_accept_all,
    )
    result = await tool.execute(peer="grocco", last_n=5)
    assert "ultimi 3 messaggi" in result
    assert "tu → grocco: Ciao Grocco" in result
    assert "grocco → tu: Ciao Bronzo" in result
    assert "tu → grocco: Tutto a posto?" in result
    # Order preserved (oldest→newest)
    pos1 = result.index("Ciao Grocco")
    pos2 = result.index("Ciao Bronzo")
    pos3 = result.index("Tutto a posto")
    assert pos1 < pos2 < pos3
    # Args propagated
    assert captured_args == {"peer": "grocco", "last_n": 5}


@pytest.mark.asyncio
async def test_peer_thread_show_empty_thread_message() -> None:
    async def _fetch(peer, *, last_n):
        return []

    tool = PeerThreadShowTool(
        fetch_thread_callback=_fetch,
        own_agent_id="bronzo",
        peer_validator=_accept_all,
    )
    result = await tool.execute(peer="manuzio")
    assert "Nessun messaggio scambiato con manuzio" in result


@pytest.mark.asyncio
async def test_peer_thread_show_refuses_self() -> None:
    async def _fetch(peer, *, last_n):
        raise AssertionError("fetch must not be called for self")

    tool = PeerThreadShowTool(
        fetch_thread_callback=_fetch,
        own_agent_id="bronzo",
    )
    result = await tool.execute(peer="bronzo")
    assert "must differ from yourself" in result


@pytest.mark.asyncio
async def test_peer_thread_show_refuses_invalid_peer_id() -> None:
    async def _fetch(peer, *, last_n):
        raise AssertionError("fetch must not be called for invalid id")

    tool = PeerThreadShowTool(
        fetch_thread_callback=_fetch,
        own_agent_id="bronzo",
        peer_validator=lambda v: v.isalpha() and v.islower(),
    )
    result = await tool.execute(peer="Bad-ID")
    assert "invalid peer agent id" in result


@pytest.mark.asyncio
async def test_peer_thread_show_clamps_last_n_bounds() -> None:
    seen_last_n = []

    async def _fetch(peer, *, last_n):
        seen_last_n.append(last_n)
        return []

    tool = PeerThreadShowTool(
        fetch_thread_callback=_fetch,
        own_agent_id="bronzo",
    )
    await tool.execute(peer="grocco", last_n=999)
    await tool.execute(peer="grocco", last_n=0)
    await tool.execute(peer="grocco", last_n=-5)
    assert seen_last_n == [100, 1, 1]


@pytest.mark.asyncio
async def test_peer_thread_show_surfaces_router_errors() -> None:
    async def _fetch(peer, *, last_n):
        raise RuntimeError("HTTP 401: unauthorized")

    tool = PeerThreadShowTool(
        fetch_thread_callback=_fetch,
        own_agent_id="bronzo",
    )
    result = await tool.execute(peer="grocco")
    assert "Error fetching thread with grocco" in result
    assert "HTTP 401" in result
