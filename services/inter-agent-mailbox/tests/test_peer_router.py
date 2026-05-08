"""Unit tests for the inter-agent mailbox peer router.

These exercise the PeerHub against a real SQLite DB on a tmp path, with a
fake :class:`WebSocket` that records send_text calls and reports a controllable
client_state. The Starlette WS endpoint and the websockets handshake are out
of scope here — covered by integration tests against a running container.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

# Make the package importable when running ``pytest`` from the service root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanobot_mailbox.auth import TokenRegistry
from nanobot_mailbox.db import Database
from nanobot_mailbox.peer_router import FRAME_VERSION, PeerHub
from nanobot_mailbox.protocol import is_valid_agent_id


class _FakeWS:
    """Minimal stand-in for ``starlette.websockets.WebSocket``.

    Records ``send_text`` payloads and exposes a mutable ``client_state`` so
    tests can simulate disconnects without spinning up a real Starlette app.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []
        # Use the real enum to match the production ``WebSocketState.CONNECTED``
        # check inside PeerHub._deliver and broadcast_presence.
        from starlette.websockets import WebSocketState
        self._state_cls = WebSocketState
        self.client_state = WebSocketState.CONNECTED

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)

    def disconnect(self) -> None:
        self.client_state = self._state_cls.DISCONNECTED


@pytest.fixture
def registry() -> TokenRegistry:
    # Simulate three peers registered with the mailbox.
    return TokenRegistry({
        "tok_bronzo": "bronzo",
        "tok_grocco": "grocco",
        "tok_naldo": "naldo",
    })


@pytest.fixture
async def db(tmp_path) -> Database:
    db = Database(tmp_path / "mailbox.db")
    await db.open()
    yield db
    await db.close()


@pytest.fixture
async def hub(db, registry) -> PeerHub:
    return PeerHub(db, registry)


@pytest.mark.asyncio
async def test_forward_to_online_peer_persists_and_marks_delivered(hub, db) -> None:
    grocco_ws = _FakeWS()
    await hub.register("grocco", grocco_ws)

    envelope = await hub.forward(
        from_agent="bronzo", to_agent="grocco", text="ciao",
        thread_id=None, in_reply_to=None,
    )
    assert envelope["from"] == "bronzo"
    assert envelope["to"] == "grocco"
    assert envelope["text"] == "ciao"
    assert envelope["v"] == FRAME_VERSION
    assert envelope["id"].startswith("msg_")

    # The frame was delivered over the live WS
    assert len(grocco_ws.sent) == 1
    delivered = json.loads(grocco_ws.sent[0])
    assert delivered["id"] == envelope["id"]

    # And marked delivered in the DB so a reconnect would not re-deliver it
    pending = await db.fetch_pending_delivery("grocco")
    assert pending == []


@pytest.mark.asyncio
async def test_forward_to_offline_peer_persists_pending(hub, db) -> None:
    envelope = await hub.forward(
        from_agent="bronzo", to_agent="grocco", text="sei offline",
        thread_id=None, in_reply_to=None,
    )
    assert envelope["text"] == "sei offline"
    pending = await db.fetch_pending_delivery("grocco")
    assert len(pending) == 1
    assert pending[0]["body"] == "sei offline"


@pytest.mark.asyncio
async def test_drain_inbox_replays_pending_in_order(hub, db) -> None:
    # Three messages while grocco is offline
    for i in range(3):
        await hub.forward(
            from_agent="bronzo", to_agent="grocco", text=f"msg {i}",
            thread_id=None, in_reply_to=None,
        )

    # Reconnect grocco and drain
    grocco_ws = _FakeWS()
    await hub.register("grocco", grocco_ws)
    drained = await hub.drain_inbox("grocco")

    assert drained == 3
    bodies = [json.loads(s)["text"] for s in grocco_ws.sent]
    assert bodies == ["msg 0", "msg 1", "msg 2"]
    # Subsequent drain finds nothing
    assert await hub.drain_inbox("grocco") == 0


@pytest.mark.asyncio
async def test_forward_rejects_self_send(hub) -> None:
    with pytest.raises(ValueError, match="cannot send to self"):
        await hub.forward(
            from_agent="bronzo", to_agent="bronzo", text="me",
            thread_id=None, in_reply_to=None,
        )


@pytest.mark.asyncio
async def test_forward_rejects_unknown_recipient(hub) -> None:
    with pytest.raises(ValueError, match="unknown recipient"):
        await hub.forward(
            from_agent="bronzo", to_agent="ghost", text="?",
            thread_id=None, in_reply_to=None,
        )


@pytest.mark.asyncio
async def test_broadcast_presence_to_all_connections(hub) -> None:
    a, b, c = _FakeWS(), _FakeWS(), _FakeWS()
    await hub.register("bronzo", a)
    await hub.register("grocco", b)
    await hub.register("naldo", c)

    await hub.broadcast_presence()

    for ws in (a, b, c):
        assert len(ws.sent) == 1
        frame = json.loads(ws.sent[-1])
        assert frame["type"] == "presence"
        assert frame["online"] == ["bronzo", "grocco", "naldo"]


class _RecordingObserver:
    """Captures hub callbacks so we can assert the wiring fires."""

    def __init__(self) -> None:
        self.msgs: list[dict] = []
        self.presence: list[dict] = []

    def on_msg(self, envelope: dict) -> None:
        self.msgs.append(envelope)

    def on_presence(self, *, event: str, agent_id: str, online: list[str]) -> None:
        self.presence.append({"event": event, "agent_id": agent_id, "online": list(online)})


@pytest.mark.asyncio
async def test_observer_receives_forwarded_msg(hub) -> None:
    obs = _RecordingObserver()
    hub.set_observer(obs)
    grocco_ws = _FakeWS()
    await hub.register("grocco", grocco_ws)

    await hub.forward(
        from_agent="bronzo", to_agent="grocco", text="ciao",
        thread_id=None, in_reply_to=None,
    )

    assert len(obs.msgs) == 1
    assert obs.msgs[0]["from"] == "bronzo"
    assert obs.msgs[0]["to"] == "grocco"
    assert obs.msgs[0]["text"] == "ciao"


@pytest.mark.asyncio
async def test_observer_msg_for_offline_peer_still_fires(hub) -> None:
    # We want the operator to see the traffic even if the recipient is
    # offline (the message is queued, but the conversation is the point of
    # the dump).
    obs = _RecordingObserver()
    hub.set_observer(obs)

    await hub.forward(
        from_agent="bronzo", to_agent="grocco", text="offline ciao",
        thread_id=None, in_reply_to=None,
    )

    assert len(obs.msgs) == 1
    assert obs.msgs[0]["text"] == "offline ciao"


@pytest.mark.asyncio
async def test_observer_failure_does_not_break_forward(hub) -> None:
    # The observer is a watcher, not a router: a buggy on_msg must never
    # surface as a peer-side error.
    class _Boom:
        def on_msg(self, envelope):  # noqa: ARG002
            raise RuntimeError("observer broken")
        def on_presence(self, **_kwargs):
            pass

    hub.set_observer(_Boom())
    envelope = await hub.forward(
        from_agent="bronzo", to_agent="grocco", text="ok",
        thread_id=None, in_reply_to=None,
    )
    assert envelope["text"] == "ok"


@pytest.mark.asyncio
async def test_broadcast_presence_skips_disconnected_socket(hub) -> None:
    a, b = _FakeWS(), _FakeWS()
    await hub.register("bronzo", a)
    await hub.register("grocco", b)
    b.disconnect()

    await hub.broadcast_presence()

    assert len(a.sent) == 1  # still gets the presence
    assert len(b.sent) == 0  # disconnected socket was skipped


@pytest.mark.asyncio
async def test_register_returns_previous_connection_for_same_agent(hub) -> None:
    """A peer reconnecting must supersede its old socket; the hub returns the
    old WS so the caller can close it cleanly."""
    first = _FakeWS()
    second = _FakeWS()
    assert await hub.register("bronzo", first) is None
    assert await hub.register("bronzo", second) is first
    # And the new socket is now the live one
    assert hub.is_online("bronzo")


@pytest.mark.asyncio
async def test_drain_inbox_loops_past_batch_limit(hub, db, monkeypatch) -> None:
    """With INBOX_DRAIN_BATCH messages waiting, a single drain returns the
    batch; the endpoint loops to drain the rest. Test the loop logic by
    calling drain_inbox repeatedly until it reports zero."""
    from nanobot_mailbox import peer_router

    # Shrink the batch so we don't have to insert 100+ rows just to verify
    # the multi-batch path.
    monkeypatch.setattr(peer_router, "INBOX_DRAIN_BATCH", 3)

    for i in range(7):
        await hub.forward(
            from_agent="bronzo", to_agent="grocco", text=f"m{i}",
            thread_id=None, in_reply_to=None,
        )

    grocco_ws = _FakeWS()
    await hub.register("grocco", grocco_ws)

    total = 0
    while True:
        n = await hub.drain_inbox("grocco")
        total += n
        if n < peer_router.INBOX_DRAIN_BATCH:
            break
    assert total == 7
    bodies = [json.loads(s)["text"] for s in grocco_ws.sent]
    assert bodies == [f"m{i}" for i in range(7)]


@pytest.mark.asyncio
async def test_handle_frame_sends_ack_on_success(hub) -> None:
    """A msg frame carrying a client_ref must round-trip an ack to sender."""
    from nanobot_mailbox.peer_router import _handle_frame

    bronzo_ws = _FakeWS()
    grocco_ws = _FakeWS()
    await hub.register("bronzo", bronzo_ws)
    await hub.register("grocco", grocco_ws)
    bronzo_ws.sent.clear()  # discard the initial presence frame
    grocco_ws.sent.clear()

    await _handle_frame(hub, "bronzo", {
        "type": "msg", "to": "grocco", "text": "ehi", "client_ref": "ref_1",
    })

    # bronzo (sender) gets the ack with matching client_ref
    acks = [json.loads(s) for s in bronzo_ws.sent if json.loads(s).get("type") == "ack"]
    assert len(acks) == 1
    assert acks[0]["client_ref"] == "ref_1"
    assert acks[0]["id"].startswith("msg_")
    # grocco (recipient) got the actual msg envelope
    msgs = [json.loads(s) for s in grocco_ws.sent if json.loads(s).get("type") == "msg"]
    assert len(msgs) == 1
    assert msgs[0]["text"] == "ehi"


@pytest.mark.asyncio
async def test_handle_frame_sends_error_on_unknown_recipient(hub) -> None:
    from nanobot_mailbox.peer_router import _handle_frame

    bronzo_ws = _FakeWS()
    await hub.register("bronzo", bronzo_ws)
    bronzo_ws.sent.clear()

    await _handle_frame(hub, "bronzo", {
        "type": "msg", "to": "ghost", "text": "?", "client_ref": "ref_2",
    })

    errs = [json.loads(s) for s in bronzo_ws.sent if json.loads(s).get("type") == "error"]
    assert len(errs) == 1
    assert errs[0]["client_ref"] == "ref_2"
    assert "unknown recipient" in errs[0]["reason"]


@pytest.mark.asyncio
async def test_fetch_thread_between_returns_chronological_pair_messages(hub, db) -> None:
    """The DB helper backing peer_thread_show must return only A↔B messages,
    sorted oldest→newest, and ignore traffic involving third parties."""
    # bronzo↔grocco (3 messages)
    await hub.forward(from_agent="bronzo", to_agent="grocco", text="A1",
                      thread_id=None, in_reply_to=None)
    await hub.forward(from_agent="grocco", to_agent="bronzo", text="A2",
                      thread_id=None, in_reply_to=None)
    await hub.forward(from_agent="bronzo", to_agent="grocco", text="A3",
                      thread_id=None, in_reply_to=None)
    # bronzo↔naldo (1 message — must NOT show up in bronzo↔grocco fetch)
    await hub.forward(from_agent="bronzo", to_agent="naldo", text="B1",
                      thread_id=None, in_reply_to=None)
    # grocco↔naldo (1 message — must NOT show up either)
    await hub.forward(from_agent="grocco", to_agent="naldo", text="C1",
                      thread_id=None, in_reply_to=None)

    rows = await db.fetch_thread_between("bronzo", "grocco", last_n=10)
    bodies = [r["body"] for r in rows]
    assert bodies == ["A1", "A2", "A3"]


@pytest.mark.asyncio
async def test_fetch_thread_between_respects_last_n_keeps_newest(hub, db) -> None:
    """last_n caps the result to the most recent N pair-messages, but the
    returned slice is still in chronological order."""
    for i in range(5):
        await hub.forward(from_agent="bronzo", to_agent="grocco", text=f"m{i}",
                          thread_id=None, in_reply_to=None)
    rows = await db.fetch_thread_between("bronzo", "grocco", last_n=3)
    assert [r["body"] for r in rows] == ["m2", "m3", "m4"]


@pytest.mark.asyncio
async def test_forward_rejects_invalid_attachments_without_orphan(hub, db) -> None:
    """Bad attachment_ids must fail BEFORE the message row is inserted, so a
    rejected forward leaves no orphan row to be replayed at reconnect."""
    before = await db.count_messages()
    with pytest.raises(ValueError, match="attachment"):
        await hub.forward(
            from_agent="bronzo", to_agent="grocco", text="ko",
            thread_id=None, in_reply_to=None,
            attachment_ids=[99999],  # not in attachments table
        )
    after = await db.count_messages()
    assert after == before


@pytest.mark.asyncio
async def test_forward_persists_closing_flag(hub, db) -> None:
    """A frame with closing=true must persist closing=1 in the messages row
    and emit envelope with closing=true so the receiver channel can skip
    waking its agent."""
    grocco_ws = _FakeWS()
    await hub.register("grocco", grocco_ws)

    envelope = await hub.forward(
        from_agent="bronzo", to_agent="grocco",
        text="task done", thread_id=None, in_reply_to=None,
        closing=True,
    )
    assert envelope.get("closing") is True

    delivered = json.loads(grocco_ws.sent[-1])
    assert delivered["closing"] is True

    # DB row stores closing=1
    rows = await db.fetch_thread_between("bronzo", "grocco", last_n=5)
    assert any(r["closing"] == 1 and r["body"] == "task done" for r in rows)


@pytest.mark.asyncio
async def test_forward_omits_closing_when_false(hub, db) -> None:
    """closing=False (default) must not pollute the envelope and must store
    closing=0 in DB so peer_thread_show can render the flag accurately."""
    grocco_ws = _FakeWS()
    await hub.register("grocco", grocco_ws)
    await hub.forward(
        from_agent="bronzo", to_agent="grocco",
        text="hi", thread_id=None, in_reply_to=None,
    )
    delivered = json.loads(grocco_ws.sent[-1])
    assert "closing" not in delivered
    rows = await db.fetch_thread_between("bronzo", "grocco", last_n=5)
    assert all(r["closing"] == 0 for r in rows)


@pytest.mark.asyncio
async def test_handle_frame_propagates_closing_flag(hub) -> None:
    """A msg frame from a client with closing=true is honoured end-to-end."""
    from nanobot_mailbox.peer_router import _handle_frame

    bronzo_ws = _FakeWS()
    grocco_ws = _FakeWS()
    await hub.register("bronzo", bronzo_ws)
    await hub.register("grocco", grocco_ws)
    bronzo_ws.sent.clear()
    grocco_ws.sent.clear()

    await _handle_frame(hub, "bronzo", {
        "type": "msg", "to": "grocco", "text": "fine",
        "client_ref": "ref_close", "closing": True,
    })
    msgs = [json.loads(s) for s in grocco_ws.sent if json.loads(s).get("type") == "msg"]
    assert len(msgs) == 1
    assert msgs[0]["closing"] is True


def test_is_valid_agent_id_pattern() -> None:
    assert is_valid_agent_id("bronzo")
    assert is_valid_agent_id("g-r-occo")
    assert is_valid_agent_id("a_b_1")
    assert not is_valid_agent_id("Bronzo")  # uppercase
    assert not is_valid_agent_id("1bronzo")  # leading digit
    assert not is_valid_agent_id("a" * 40)   # too long
    assert not is_valid_agent_id("")          # empty
