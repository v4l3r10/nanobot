"""Unit tests for the embedded Telegram observer.

The Telegram bot itself (long-polling, Application lifecycle) is not started:
we only exercise the pure logic — env parsing, event formatting, subscriber
persistence, allow-list enforcement — which is what would actually break in
day-to-day editing. Real bot wiring is integration territory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanobot_mailbox.db import Database
from nanobot_mailbox.telegram_observer import (
    CB_HELP,
    CB_LAST_10,
    CB_LAST_30,
    CB_MUTE,
    CB_PEERS,
    CB_STATUS,
    CB_STOP,
    CB_UNMUTE,
    TelegramObserver,
    _main_menu_keyboard,
    parse_allowed_users,
)


@pytest.fixture
async def db(tmp_path) -> Database:
    db = Database(tmp_path / "mailbox.db")
    await db.open()
    yield db
    await db.close()


def test_parse_allowed_users_json() -> None:
    assert parse_allowed_users("[1,2,3]") == {1, 2, 3}


def test_parse_allowed_users_csv_fallback() -> None:
    # Operator typo: forgot brackets. We accept it rather than fail closed
    # silently — the bot would otherwise reject every command with no hint.
    assert parse_allowed_users("1, 2 ,3") == {1, 2, 3}


def test_parse_allowed_users_empty() -> None:
    assert parse_allowed_users(None) == set()
    assert parse_allowed_users("") == set()


@pytest.fixture
def observer(db) -> TelegramObserver:
    async def _online() -> list[str]:
        return ["bronzo", "grocco"]

    return TelegramObserver(
        token="fake-token",
        allowed_users={42},
        db=db,
        hub_status_provider=_online,
    )


def test_format_msg_basic(observer) -> None:
    text = observer._format({
        "kind": "msg",
        "envelope": {
            "v": 1, "type": "msg", "id": "msg_x",
            "from": "bronzo", "to": "grocco",
            "text": "ciao", "thread_id": "thr_x",
            "in_reply_to": None, "ts": "2026-05-08T10:00:00Z",
        },
    })
    assert "<b>bronzo</b>" in text
    assert "<b>grocco</b>" in text
    assert "ciao" in text
    assert "🔚" not in text


def test_format_msg_closing_flag(observer) -> None:
    text = observer._format({
        "kind": "msg",
        "envelope": {
            "from": "bronzo", "to": "grocco", "text": "addio",
            "closing": True, "ts": "2026-05-08T10:00:00Z",
        },
    })
    assert "🔚 closing" in text


def test_format_msg_html_escapes_body(observer) -> None:
    # We post with parse_mode=HTML, so a body containing < or & must not
    # leak into the markup. Critical: a peer typing "<script>" should not
    # crash sendMessage with parse error or worse, render as HTML.
    text = observer._format({
        "kind": "msg",
        "envelope": {
            "from": "bronzo", "to": "grocco",
            "text": "<script>alert(1)</script> & co",
            "ts": "2026-05-08T10:00:00Z",
        },
    })
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "&amp;" in text


def test_format_msg_with_attachments(observer) -> None:
    text = observer._format({
        "kind": "msg",
        "envelope": {
            "from": "bronzo", "to": "grocco", "text": "foto",
            "attachments": [{"name": "a.png", "mime": "image/png", "size_bytes": 1234}],
            "ts": "2026-05-08T10:00:00Z",
        },
    })
    assert "📎 a.png" in text
    assert "image/png" in text
    assert "1234" in text


def test_format_presence_up(observer) -> None:
    text = observer._format({
        "kind": "presence", "event": "up", "agent_id": "bronzo",
        "online": ["bronzo", "grocco"],
    })
    assert "🟢" in text
    assert "bronzo" in text


def test_format_presence_down(observer) -> None:
    text = observer._format({
        "kind": "presence", "event": "down", "agent_id": "bronzo",
        "online": [],
    })
    assert "🔴" in text


@pytest.mark.asyncio
async def test_subscriber_persistence_round_trip(db) -> None:
    await db.tg_subscriber_add(chat_id=111, user_id=42)
    await db.tg_subscriber_add(chat_id=222, user_id=42)
    active = await db.tg_subscribers_active()
    assert active == [111, 222]

    # Mute one — it should drop out of the "active" list but still be tracked.
    ok = await db.tg_subscriber_set_muted(chat_id=111, muted=True)
    assert ok is True
    assert await db.tg_subscribers_active() == [222]
    all_rows = await db.tg_subscribers_all()
    assert {int(r["chat_id"]) for r in all_rows} == {111, 222}

    await db.tg_subscriber_remove(222)
    assert await db.tg_subscribers_active() == []


@pytest.mark.asyncio
async def test_subscriber_add_is_idempotent(db) -> None:
    # /start called twice from the same chat must not create duplicates,
    # and must clear a prior muted state (re-subscribing means "I want to
    # see things again").
    await db.tg_subscriber_add(chat_id=111, user_id=42)
    await db.tg_subscriber_set_muted(chat_id=111, muted=True)
    await db.tg_subscriber_add(chat_id=111, user_id=42)

    rows = await db.tg_subscribers_all()
    assert len(rows) == 1
    assert rows[0]["muted"] == 0


def test_main_menu_keyboard_covers_every_callback() -> None:
    # Regression guard: if someone adds a callback constant but forgets to
    # surface it on the keyboard, the menu silently degrades. Every CB_*
    # the dispatcher knows about should appear as a tappable button.
    kb = _main_menu_keyboard()
    seen = {btn.callback_data for row in kb.inline_keyboard for btn in row}
    expected = {
        CB_PEERS, CB_STATUS, CB_LAST_10, CB_LAST_30,
        CB_MUTE, CB_UNMUTE, CB_HELP, CB_STOP,
    }
    assert seen == expected


@pytest.mark.asyncio
async def test_break_invokes_bilateral_forward(db) -> None:
    """`/break A B` must call forward twice: A→B then B→A, both closing=true."""

    async def _online() -> list[str]:
        return ["bronzo", "grocco"]

    calls: list[dict] = []

    async def _fake_forward(**kwargs):
        calls.append(kwargs)
        return {"id": f"msg_{len(calls)}", "to": kwargs["to_agent"]}

    observer = TelegramObserver(
        token="fake", allowed_users={42}, db=db,
        hub_status_provider=_online, forward_callable=_fake_forward,
    )

    # Build the minimal Update-like inputs the handler needs without standing
    # up the full PTB Application. We patch out _is_allowed and _reply
    # because they touch Telegram surfaces; the goal here is the forward wiring.
    seen_replies: list[str] = []
    observer._is_allowed = lambda _u: True  # type: ignore[method-assign]

    async def _capture_reply(_update, text, *, keyboard=None):
        seen_replies.append(text)
    observer._reply = _capture_reply  # type: ignore[method-assign]

    class _Ctx:
        args = ["bronzo", "grocco"]

    await observer._cmd_break(update=None, ctx=_Ctx())  # type: ignore[arg-type]

    assert len(calls) == 2
    assert (calls[0]["from_agent"], calls[0]["to_agent"]) == ("bronzo", "grocco")
    assert (calls[1]["from_agent"], calls[1]["to_agent"]) == ("grocco", "bronzo")
    assert all(c["closing"] is True for c in calls)
    assert "🛑" in seen_replies[0]


@pytest.mark.asyncio
async def test_break_rejects_wrong_arity(db) -> None:
    """`/break` without exactly two args should refuse and not call forward."""

    async def _online() -> list[str]:
        return []

    calls: list[dict] = []

    async def _fake_forward(**kwargs):
        calls.append(kwargs)

    observer = TelegramObserver(
        token="fake", allowed_users={42}, db=db,
        hub_status_provider=_online, forward_callable=_fake_forward,
    )
    observer._is_allowed = lambda _u: True  # type: ignore[method-assign]
    seen: list[str] = []

    async def _capture_reply(_update, text, *, keyboard=None):
        seen.append(text)
    observer._reply = _capture_reply  # type: ignore[method-assign]

    class _Ctx:
        args = ["only_one"]
    await observer._cmd_break(update=None, ctx=_Ctx())  # type: ignore[arg-type]

    assert calls == []  # forward never called
    assert "uso:" in seen[0]


@pytest.mark.asyncio
async def test_break_reports_partial_failure(db) -> None:
    """If only one direction fails (e.g. unknown peer), report partial."""

    async def _online() -> list[str]:
        return []

    async def _flaky_forward(**kwargs):
        if kwargs["to_agent"] == "ghost":
            raise ValueError("unknown recipient: ghost")
        return {"id": "msg_ok", "to": kwargs["to_agent"]}

    observer = TelegramObserver(
        token="fake", allowed_users={42}, db=db,
        hub_status_provider=_online, forward_callable=_flaky_forward,
    )
    observer._is_allowed = lambda _u: True  # type: ignore[method-assign]
    seen: list[str] = []

    async def _capture_reply(_update, text, *, keyboard=None):
        seen.append(text)
    observer._reply = _capture_reply  # type: ignore[method-assign]

    class _Ctx:
        args = ["bronzo", "ghost"]
    await observer._cmd_break(update=None, ctx=_Ctx())  # type: ignore[arg-type]

    # Operator must see both the success and the failure to know the
    # state of the break (one side closed, the other rejected).
    msg = seen[0]
    assert "⚠️" in msg
    assert "ghost" in msg
    assert "bronzo→ghost" in msg


def test_event_queue_drops_on_overflow(observer) -> None:
    # Fill the queue beyond capacity. on_msg/on_presence must NEVER raise
    # at the call site — they're invoked from the hub's hot path.
    for _ in range(observer._queue.maxsize + 5):
        observer.on_msg({"from": "a", "to": "b", "text": "x"})
    # Same for presence overflow.
    for _ in range(10):
        observer.on_presence(event="up", agent_id="bronzo", online=[])
    # Queue should be capped; no exception should have escaped.
    assert observer._queue.qsize() == observer._queue.maxsize
