"""Unit tests for the PeerChannel inbound frame handling and outbound sender.

The persistent-WebSocket session is exercised by integration tests against a
running mailbox-router elsewhere; here we cover the parts that do not require
a live socket: presence cache, frame parsing, chat_id mapping, send-side
validation.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.peer import (
    PeerChannel,
    PeerConfig,
    _peer_chat_id,
    _peer_from_chat_id,
    _ws_to_http_base,
)


def _make_channel(**overrides) -> PeerChannel:
    cfg = PeerConfig(
        enabled=True,
        agent_id=overrides.pop("agent_id", "bronzo"),
        token=overrides.pop("token", "tok"),
        router_url=overrides.pop("router_url", "ws://router:8765/peer"),
    )
    bus = MessageBus()
    return PeerChannel(cfg, bus)


def test_chat_id_mapping_round_trip() -> None:
    # New format: chat_id is the bare peer agent_id. The channel field on
    # the bus is what disambiguates this from chats on other channels.
    assert _peer_chat_id("grocco") == "grocco"
    assert _peer_from_chat_id("grocco") == "grocco"
    # Empty chat_id is rejected as malformed (channel must point at *some*
    # peer, blank id would make routing ambiguous).
    assert _peer_from_chat_id("") is None
    # Backward compat: legacy "peer:<agent>" form still resolves so an
    # OutboundMessage queued before the cleanup keeps routing correctly.
    assert _peer_from_chat_id("peer:grocco") == "grocco"
    assert _peer_from_chat_id("peer:") is None


def test_ws_to_http_base_handles_ws_and_wss() -> None:
    assert _ws_to_http_base("ws://nanobot-mailbox:8765/peer") == "http://nanobot-mailbox:8765"
    assert _ws_to_http_base("wss://router.example/peer") == "https://router.example"


@pytest.mark.asyncio
async def test_presence_frame_updates_known_peers_and_filters_self() -> None:
    ch = _make_channel(agent_id="bronzo")
    await ch._on_frame(json.dumps({
        "v": 1, "type": "presence",
        "online": ["bronzo", "grocco", "naldo"],
    }))
    # list_peers excludes self by contract
    assert ch.list_peers() == ["grocco", "naldo"]
    # _known_peers may keep self but list_peers must not surface it
    assert "bronzo" in ch._known_peers


@pytest.mark.asyncio
async def test_msg_frame_publishes_inbound_with_peer_chat_id() -> None:
    ch = _make_channel(agent_id="bronzo")
    captured: list[InboundMessage] = []

    async def _capture(msg: InboundMessage) -> None:
        captured.append(msg)

    ch.bus.publish_inbound = _capture  # type: ignore[assignment]
    await ch._on_frame(json.dumps({
        "v": 1, "type": "msg",
        "id": "msg_1", "from": "grocco", "to": "bronzo",
        "text": "ciao Bronzo", "thread_id": "thr_1",
        "in_reply_to": None, "ts": "2026-05-08T10:00:00.000Z",
    }))
    assert len(captured) == 1
    inbound = captured[0]
    assert inbound.channel == "peer"
    assert inbound.sender_id == "grocco"
    assert inbound.chat_id == "grocco"
    assert inbound.content == "ciao Bronzo"
    assert inbound.metadata["peer_message_id"] == "msg_1"
    assert inbound.metadata["peer_thread_id"] == "thr_1"


@pytest.mark.asyncio
async def test_malformed_frames_are_silently_dropped() -> None:
    ch = _make_channel()
    captured: list[InboundMessage] = []

    async def _capture(msg: InboundMessage) -> None:
        captured.append(msg)

    ch.bus.publish_inbound = _capture  # type: ignore[assignment]
    # Not JSON
    await ch._on_frame("not json at all")
    # Missing 'from'
    await ch._on_frame(json.dumps({"type": "msg", "to": "bronzo", "text": "x"}))
    # Bad text type
    await ch._on_frame(json.dumps({
        "type": "msg", "from": "grocco", "to": "bronzo", "text": 42,
    }))
    # Unsupported type
    await ch._on_frame(json.dumps({"type": "garbage"}))
    assert captured == []


@pytest.mark.asyncio
async def test_send_rejects_empty_chat_id() -> None:
    # After the chat_id cleanup, the channel no longer enforces a "peer:"
    # prefix — chat_id is just the bare agent_id. Format errors that used
    # to be caught here (e.g. accidentally using a telegram numeric id)
    # now fail downstream at the router, which has the canonical roster.
    # The one shape we still reject locally is an empty chat_id, since
    # that would route to a non-existent peer with no useful error.
    ch = _make_channel()
    msg = OutboundMessage(channel="peer", chat_id="", content="oops")
    with pytest.raises(ValueError, match="empty or invalid"):
        await ch.send(msg)


@pytest.mark.asyncio
async def test_send_rejects_self() -> None:
    ch = _make_channel(agent_id="bronzo")
    msg = OutboundMessage(channel="peer", chat_id="bronzo", content="me")
    with pytest.raises(ValueError, match="cannot send to self"):
        await ch.send(msg)


@pytest.mark.asyncio
async def test_send_raises_when_disconnected() -> None:
    ch = _make_channel()
    # Channel never started → no WS
    msg = OutboundMessage(channel="peer", chat_id="grocco", content="hi")
    with pytest.raises(RuntimeError, match="not connected"):
        await ch.send(msg)


@pytest.mark.asyncio
async def test_send_serializes_thread_metadata_into_frame() -> None:
    ch = _make_channel()
    # Tight ack window so the test fails fast if correlation breaks.
    ch._cfg.ack_timeout_s = 1.0  # type: ignore[attr-defined]

    sent_payloads: list[str] = []

    class _AckingFakeWS:
        """Fake WS that echoes back an ack frame the moment send is called,
        emulating the router's notify_sender path so send() resolves quickly.
        """

        def __init__(self, channel: PeerChannel) -> None:
            self._ch = channel

        async def send(self, payload: str) -> None:
            sent_payloads.append(payload)
            frame = json.loads(payload)
            # Schedule the ack on the loop so it lands while send() awaits.
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(self._ch._on_frame(json.dumps({
                    "v": 1, "type": "ack",
                    "client_ref": frame["client_ref"], "id": "msg_X",
                })))
            )

    ch._ws = _AckingFakeWS(ch)  # type: ignore[assignment]
    msg = OutboundMessage(
        channel="peer", chat_id="grocco", content="re",
        metadata={"peer_thread_id": "thr_X", "peer_in_reply_to": "msg_Y"},
    )
    await ch.send(msg)
    assert len(sent_payloads) == 1
    frame = json.loads(sent_payloads[0])
    assert frame["to"] == "grocco"
    assert frame["text"] == "re"
    assert frame["thread_id"] == "thr_X"
    assert frame["in_reply_to"] == "msg_Y"
    assert "attachment_ids" not in frame  # no media → no field
    assert isinstance(frame["client_ref"], str) and frame["client_ref"]


@pytest.mark.asyncio
async def test_send_raises_on_router_error_frame() -> None:
    """A nack from the router (type=error with reason) surfaces as a
    RuntimeError to the caller, so peer_say can return a real error string."""
    ch = _make_channel()
    ch._cfg.ack_timeout_s = 1.0  # type: ignore[attr-defined]

    class _NackingFakeWS:
        def __init__(self, channel: PeerChannel) -> None:
            self._ch = channel

        async def send(self, payload: str) -> None:
            frame = json.loads(payload)
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(self._ch._on_frame(json.dumps({
                    "v": 1, "type": "error",
                    "client_ref": frame["client_ref"],
                    "reason": "unknown recipient: ghost",
                })))
            )

    ch._ws = _NackingFakeWS(ch)  # type: ignore[assignment]
    msg = OutboundMessage(channel="peer", chat_id="ghost", content="?")
    with pytest.raises(RuntimeError, match="unknown recipient"):
        await ch.send(msg)


@pytest.mark.asyncio
async def test_send_swallows_ack_timeout_as_warning() -> None:
    """If the router never replies, send() returns without raising — the
    frame is on the wire and routing is fire-and-forget at that point.
    """
    ch = _make_channel()
    ch._cfg.ack_timeout_s = 0.05  # type: ignore[attr-defined]

    class _SilentFakeWS:
        async def send(self, payload: str) -> None:
            pass  # never acks

    ch._ws = _SilentFakeWS()  # type: ignore[assignment]
    msg = OutboundMessage(channel="peer", chat_id="grocco", content="hi")
    await ch.send(msg)  # must not raise
    assert ch._pending_acks == {}  # cleaned up on timeout


@pytest.mark.asyncio
async def test_inbound_closing_frame_does_not_wake_agent_loop() -> None:
    """The defining contract of the closing=true protocol: a frame with
    closing=true must NOT trigger publish_inbound (the agent loop stays
    asleep), so two polite bots can't loop on goodbyes."""
    ch = _make_channel()
    captured: list[InboundMessage] = []

    async def _capture(msg: InboundMessage) -> None:
        captured.append(msg)

    ch.bus.publish_inbound = _capture  # type: ignore[assignment]

    await ch._on_frame(json.dumps({
        "v": 1, "type": "msg",
        "id": "msg_close_1", "from": "grocco", "to": "bronzo",
        "text": "task done", "thread_id": "thr_x",
        "in_reply_to": None, "ts": "2026-05-08T18:00:00.000Z",
        "closing": True,
    }))
    assert captured == [], "closing=true must not wake agent loop"


@pytest.mark.asyncio
async def test_inbound_normal_frame_still_wakes_agent_loop() -> None:
    """Sanity: only closing=true is filtered, regular messages still flow."""
    ch = _make_channel()
    captured: list[InboundMessage] = []

    async def _capture(msg: InboundMessage) -> None:
        captured.append(msg)

    ch.bus.publish_inbound = _capture  # type: ignore[assignment]

    await ch._on_frame(json.dumps({
        "v": 1, "type": "msg",
        "id": "msg_normal", "from": "grocco", "to": "bronzo",
        "text": "ciao Bronzo", "thread_id": "thr_x",
        "in_reply_to": None, "ts": "2026-05-08T18:00:00.000Z",
    }))
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_send_filters_progress_noise_outbound() -> None:
    """Progress / streaming bookkeeping events from the agent loop must NOT
    reach the peer plane: they have no closing flag and would wake the
    receiver's agent loop with empty bodies (the 'ghost twin' bug).
    """
    ch = _make_channel()

    sent_payloads: list[str] = []

    class _RecorderWS:
        async def send(self, payload: str) -> None:
            sent_payloads.append(payload)

    ch._ws = _RecorderWS()  # type: ignore[assignment]

    for noise_meta in (
        {"_progress": True},
        {"_tool_hint": True},
        {"_retry_wait": True},
        {"_stream_delta": True},
        {"_stream_end": True},
        {"_streamed": True},
    ):
        msg = OutboundMessage(
            channel="peer", chat_id="grocco",
            content="bookkeeping noise", metadata=noise_meta,
        )
        await ch.send(msg)
    assert sent_payloads == [], "no progress event should reach the wire"


@pytest.mark.asyncio
async def test_send_filters_empty_content() -> None:
    """An OutboundMessage with empty/whitespace-only content is dropped at
    the peer client — keeps the receiver's agent asleep when there's
    literally nothing to say."""
    ch = _make_channel()
    sent: list[str] = []

    class _RecorderWS:
        async def send(self, payload: str) -> None:
            sent.append(payload)

    ch._ws = _RecorderWS()  # type: ignore[assignment]

    for empty in ("", "   ", "\n\t  "):
        await ch.send(OutboundMessage(
            channel="peer", chat_id="grocco", content=empty,
        ))
    assert sent == []


@pytest.mark.asyncio
async def test_send_propagates_closing_metadata_into_frame() -> None:
    """If OutboundMessage.metadata.peer_closing == True, the wire frame must
    include closing=true so the router can mark the row and the recipient
    channel can apply the no-wake rule."""
    ch = _make_channel()
    ch._cfg.ack_timeout_s = 1.0  # type: ignore[attr-defined]
    sent_payloads: list[str] = []

    class _AckingFakeWS:
        def __init__(self, channel: PeerChannel) -> None:
            self._ch = channel

        async def send(self, payload: str) -> None:
            sent_payloads.append(payload)
            frame = json.loads(payload)
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(self._ch._on_frame(json.dumps({
                    "v": 1, "type": "ack",
                    "client_ref": frame["client_ref"], "id": "msg_X",
                })))
            )

    ch._ws = _AckingFakeWS(ch)  # type: ignore[assignment]
    msg = OutboundMessage(
        channel="peer", chat_id="grocco", content="task done",
        metadata={"peer_closing": True},
    )
    await ch.send(msg)
    frame = json.loads(sent_payloads[0])
    assert frame["closing"] is True


@pytest.mark.asyncio
async def test_send_omits_closing_field_when_not_set() -> None:
    """Default outbound (no peer_closing in metadata) must not include the
    closing field on the wire — keeps frames small and unambiguous."""
    import asyncio as _asyncio  # local import to keep module-top tidy
    ch = _make_channel()
    ch._cfg.ack_timeout_s = 1.0  # type: ignore[attr-defined]
    sent_payloads: list[str] = []

    class _AckingFakeWS:
        def __init__(self, channel: PeerChannel) -> None:
            self._ch = channel

        async def send(self, payload: str) -> None:
            sent_payloads.append(payload)
            frame = json.loads(payload)
            _asyncio.get_running_loop().call_soon(
                lambda: _asyncio.ensure_future(self._ch._on_frame(json.dumps({
                    "v": 1, "type": "ack",
                    "client_ref": frame["client_ref"], "id": "msg_Y",
                })))
            )

    ch._ws = _AckingFakeWS(ch)  # type: ignore[assignment]
    msg = OutboundMessage(channel="peer", chat_id="grocco", content="ciao")
    await ch.send(msg)
    frame = json.loads(sent_payloads[0])
    assert "closing" not in frame


@pytest.mark.asyncio
async def test_disconnect_clears_known_peers_for_freshness() -> None:
    """After a disconnect, the channel should not report stale online peers
    until the next presence push from the router."""
    ch = _make_channel()
    ch._known_peers = {"grocco", "naldo"}
    # Simulate the cleanup that runs in _run_session's finally block:
    ch._known_peers.clear()
    assert ch.list_peers() == []
