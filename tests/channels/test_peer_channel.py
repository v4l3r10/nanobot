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
    _guess_mime,
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


# --- _guess_mime: encoding-aware MIME resolution -----------------------------

def test_guess_mime_tar_gz_returns_gzip() -> None:
    """foo.tar.gz must be reported as application/gzip — without this the
    router rejects with 415 because mimetypes.guess_type returns
    ('application/x-tar', 'gzip') and we'd ship the uncompressed type."""
    assert _guess_mime("foo.tar.gz") == "application/gzip"


def test_guess_mime_tar_bz2_returns_bzip2() -> None:
    assert _guess_mime("foo.tar.bz2") == "application/x-bzip2"


def test_guess_mime_tar_xz_returns_xz() -> None:
    assert _guess_mime("foo.tar.xz") == "application/x-xz"


def test_guess_mime_known_type_unchanged() -> None:
    assert _guess_mime("note.txt") == "text/plain"
    assert _guess_mime("photo.png") == "image/png"


def test_guess_mime_unknown_falls_back_to_octet_stream() -> None:
    assert _guess_mime("blob.weirdext") == "application/octet-stream"


# --- _upload_media: absolute-path enforcement --------------------------------

@pytest.mark.asyncio
async def test_upload_media_rejects_relative_path() -> None:
    """Relative paths are rejected with a clear, parseable error so the agent
    can correct on its next turn instead of triggering 3x retries on a
    deterministic failure. Regression: pre-fix, Bronzo passed workspace-
    relative paths like 'progetti/foo.zip' and got the misleading 'not a
    file' error after retries had already wasted seconds."""
    ch = _make_channel()
    # Stub HTTP client so we never reach the network — the path check fires
    # before any I/O.
    ch._http = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="must be absolute"):
        await ch._upload_media(["progetti/foo.zip"])


@pytest.mark.asyncio
async def test_upload_media_rejects_nonexistent_absolute_path() -> None:
    """Absolute path that doesn't exist still raises the existing 'not a
    file' error — the absoluteness check is purely additive."""
    ch = _make_channel()
    ch._http = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="not a file"):
        await ch._upload_media(["/nonexistent/abs/path.zip"])


# --- _download_attachments: no empty msg dirs --------------------------------

@pytest.mark.asyncio
async def test_download_attachments_skips_dir_creation_when_empty(tmp_path, monkeypatch) -> None:
    """If a frame carries no attachments (or only malformed entries), no
    peer/msg_*/ dir should be created. Empty dirs misled receivers into
    thinking a download had failed when in reality there was no payload."""
    from nanobot.channels import peer as peer_mod

    monkeypatch.setattr(peer_mod, "get_workspace_path", lambda: tmp_path)
    ch = _make_channel()
    ch._http = object()  # type: ignore[assignment]

    results = await ch._download_attachments([], msg_id="msg_empty")
    assert results == []
    assert not (tmp_path / "peer" / "msg_empty").exists()

    # Malformed entries (missing id, wrong type) are silently skipped and
    # likewise must not leave a dir behind.
    results = await ch._download_attachments(
        [{"name": "x"}, "not-a-dict"], msg_id="msg_malformed"  # type: ignore[list-item]
    )
    assert results == []
    assert not (tmp_path / "peer" / "msg_malformed").exists()


# --- _on_frame: closing + attachments must not silently drop payload ---------

@pytest.mark.asyncio
async def test_inbound_closing_with_attachments_still_processes(tmp_path, monkeypatch) -> None:
    """A closing frame that carries attachments is almost certainly a sender
    bug (closing is for pure conversational closure, not file delivery).
    The receiver MUST process it anyway: dropping silently would lose the
    payload. We log a warning and forward to the agent loop."""
    from nanobot.channels import peer as peer_mod

    monkeypatch.setattr(peer_mod, "get_workspace_path", lambda: tmp_path)
    ch = _make_channel()

    captured: list[InboundMessage] = []

    async def _capture(msg: InboundMessage) -> None:
        captured.append(msg)

    ch.bus.publish_inbound = _capture  # type: ignore[assignment]

    # Stub _download_attachments so the test doesn't need a live HTTP client.
    async def _fake_download(attachments, *, msg_id):
        return ["/fake/local/path.zip"]

    ch._download_attachments = _fake_download  # type: ignore[assignment]

    await ch._on_frame(json.dumps({
        "v": 1, "type": "msg",
        "id": "msg_close_with_files", "from": "grocco", "to": "bronzo",
        "text": "ecco il pacchetto", "thread_id": None,
        "in_reply_to": None, "ts": "2026-05-10T12:00:00.000Z",
        "closing": True,
        "attachments": [{"id": 42, "name": "foo.zip", "mime": "application/zip", "size_bytes": 1}],
    }))
    assert len(captured) == 1, "closing+attachments must reach the agent loop"
    assert captured[0].media == ["/fake/local/path.zip"]


@pytest.mark.asyncio
async def test_inbound_closing_without_attachments_still_silent() -> None:
    """Sanity: pure closing frames (no attachments) keep the original
    behavior — agent loop NOT awoken. The new exception is narrow."""
    ch = _make_channel()
    captured: list[InboundMessage] = []

    async def _capture(msg: InboundMessage) -> None:
        captured.append(msg)

    ch.bus.publish_inbound = _capture  # type: ignore[assignment]

    await ch._on_frame(json.dumps({
        "v": 1, "type": "msg",
        "id": "msg_pure_close", "from": "grocco", "to": "bronzo",
        "text": "alla prossima", "thread_id": None,
        "in_reply_to": None, "ts": "2026-05-10T12:00:00.000Z",
        "closing": True,
    }))
    assert captured == []
