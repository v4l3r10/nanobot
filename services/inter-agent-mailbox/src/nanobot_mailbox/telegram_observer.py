"""Embedded Telegram observer for the inter-agent mailbox.

Streams every forwarded peer frame and presence event to a set of
subscribed Telegram chats so an operator can watch live inter-agent
traffic from a phone. The bot is opt-in: if ``MAILBOX_TELEGRAM_BOT_TOKEN``
is unset the service runs exactly as before, with no extra dependencies
loaded at runtime.

Design notes
------------
* The hub calls :meth:`TelegramObserver.on_msg` and :meth:`on_presence`
  in the same task that just persisted/forwarded the frame. We MUST NOT
  block that task on Telegram I/O — a slow Telegram API call would back
  up the entire peer plane. Events are therefore pushed onto a bounded
  asyncio.Queue and a separate worker drains them and posts to subscribers.
* Subscribers persist in SQLite (``tg_subscribers``) so a restart does
  not lose the watch list.
* Only Telegram user_ids in ``MAILBOX_TELEGRAM_ALLOWED_USERS`` may run
  any command. The list is the security boundary — we never trust a
  bare chat_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from html import escape as _h
from typing import Any

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from .db import Database

log = logging.getLogger(__name__)

# Bound to keep memory predictable if Telegram is slow / down. Older events
# are dropped first; the observer is best-effort, the DB is the source of
# truth for replay if needed.
EVENT_QUEUE_MAX = 500
RECENT_PREVIEW_MAX = 50  # max for /last N
TELEGRAM_BODY_PREVIEW_MAX = 1500  # truncate per-frame body to keep messages readable

# callback_data values for the inline menu. Kept short (Telegram caps at 64
# bytes) and prefixed so we can spot stale buttons from older builds.
CB_PEERS = "mb:peers"
CB_STATUS = "mb:status"
CB_LAST_10 = "mb:last:10"
CB_LAST_30 = "mb:last:30"
CB_MUTE = "mb:mute"
CB_UNMUTE = "mb:unmute"
CB_STOP = "mb:stop"
CB_HELP = "mb:help"


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard surfaced on /start, /help, /menu.

    Two-column layout. Buttons map 1:1 onto existing slash commands so the
    menu is purely a discoverability aid: anything tappable here is also
    typeable.
    """
    rows = [
        [
            InlineKeyboardButton("🟢 peers", callback_data=CB_PEERS),
            InlineKeyboardButton("📊 status", callback_data=CB_STATUS),
        ],
        [
            InlineKeyboardButton("📜 last 10", callback_data=CB_LAST_10),
            InlineKeyboardButton("📜 last 30", callback_data=CB_LAST_30),
        ],
        [
            InlineKeyboardButton("🔇 mute", callback_data=CB_MUTE),
            InlineKeyboardButton("🔔 unmute", callback_data=CB_UNMUTE),
        ],
        [
            InlineKeyboardButton("ℹ️ help", callback_data=CB_HELP),
            InlineKeyboardButton("👋 stop", callback_data=CB_STOP),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def parse_allowed_users(raw: str | None) -> set[int]:
    """Parse the JSON list in ``MAILBOX_TELEGRAM_ALLOWED_USERS`` into ints.

    Accepts ``"[123,456]"`` or a CSV fallback ``"123,456"`` for convenience.
    Empty / missing → empty set, which means the observer is enabled but
    nobody is allowed to /start it (safe default).
    """
    if not raw:
        return set()
    raw = raw.strip()
    try:
        data = json.loads(raw)
        return {int(x) for x in data}
    except (ValueError, TypeError):
        # CSV fallback: tolerate "123,456" without JSON brackets
        return {int(x.strip()) for x in raw.split(",") if x.strip()}


class TelegramObserver:
    """Bridge between the peer hub and a Telegram bot.

    Owns one ``telegram.ext.Application`` (long-polling) plus a worker that
    formats and dispatches hub events to every subscriber chat.
    """

    def __init__(
        self,
        *,
        token: str,
        allowed_users: set[int],
        db: Database,
        hub_status_provider,
        forward_callable=None,
        base_url: str | None = None,
        base_file_url: str | None = None,
    ):
        self._token = token
        self._allowed = set(allowed_users)
        self._db = db
        # hub_status_provider is a callable returning the current online
        # roster (List[str]). Injected to avoid a circular import with PeerHub.
        self._hub_status = hub_status_provider
        # forward_callable mirrors PeerHub.forward — used by /break to inject
        # a synthetic closing=true frame between two peers and break a runaway
        # chat loop. Optional: if None, /break is a no-op that explains the
        # feature is disabled.
        self._forward = forward_callable
        # Optional override to point at a self-hosted Bot API (e.g. the
        # ``telegram-bot-api`` sidecar shared with the other nanobots).
        self._base_url = base_url
        self._base_file_url = base_file_url
        self._app: Application | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=EVENT_QUEUE_MAX)
        self._worker: asyncio.Task | None = None
        self._started = False

    # ------------------------------------------------------------------ hub API
    def on_msg(self, envelope: dict[str, Any]) -> None:
        """Hub callback: a peer message has been forwarded.

        Non-blocking: we drop on overflow rather than backpressuring the
        peer plane. Synchronous on purpose so the hub can call it without
        ``await`` from any path.
        """
        try:
            self._queue.put_nowait({"kind": "msg", "envelope": envelope})
        except asyncio.QueueFull:
            log.warning("telegram observer queue full; dropping msg event")

    def on_presence(self, *, event: str, agent_id: str, online: list[str]) -> None:
        """Hub callback: a peer connected (event='up') or disconnected ('down')."""
        try:
            self._queue.put_nowait({
                "kind": "presence",
                "event": event,
                "agent_id": agent_id,
                "online": list(online),
            })
        except asyncio.QueueFull:
            log.warning("telegram observer queue full; dropping presence event")

    # ----------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self._started:
            return
        builder = Application.builder().token(self._token)
        if self._base_url:
            builder = builder.base_url(self._base_url)
        if self._base_file_url:
            builder = builder.base_file_url(self._base_file_url)
        self._app = builder.build()
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("peers", self._cmd_peers))
        self._app.add_handler(CommandHandler("mute", self._cmd_mute))
        self._app.add_handler(CommandHandler("unmute", self._cmd_unmute))
        self._app.add_handler(CommandHandler("last", self._cmd_last))
        self._app.add_handler(CommandHandler("menu", self._cmd_menu))
        self._app.add_handler(CommandHandler("break", self._cmd_break))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CallbackQueryHandler(self._on_callback, pattern=r"^mb:"))

        await self._app.initialize()
        # Configure the bot's UI surfaces *after* initialize() so the bot
        # client is fully wired (token validated, API endpoint resolved).
        # Done explicitly here rather than via post_init because the latter
        # was not invoked reliably across PTB versions, leaving an empty
        # commands list and a default menu button.
        await self._configure_bot_ui()
        await self._app.start()
        # drop_pending_updates avoids replaying days of /start commands the
        # first time the operator deploys this build.
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._worker = asyncio.create_task(self._dispatch_loop(), name="tg-observer-dispatch")
        self._started = True
        log.info("telegram observer started")

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        if self._app is not None:
            try:
                if self._app.updater is not None and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:  # noqa: BLE001
                log.exception("telegram observer shutdown failed")
            self._app = None
        log.info("telegram observer stopped")

    async def _configure_bot_ui(self) -> None:
        """Push the slash-commands list and the chat menu button to Telegram.

        Idempotent: re-runs on every startup without harm. Logs success
        loudly so operators can grep ``configured bot UI`` in container
        logs to confirm the menu is in sync after a deploy.
        """
        assert self._app is not None
        await self._app.bot.set_my_commands([
            BotCommand("start", "iscrivi questa chat al dump live"),
            BotCommand("menu", "menu interattivo (inline keyboard)"),
            BotCommand("stop", "annulla iscrizione"),
            BotCommand("status", "stato servizio + sottoscrittori"),
            BotCommand("peers", "peer online ora"),
            BotCommand("mute", "silenzia il dump in questa chat"),
            BotCommand("unmute", "riattiva il dump"),
            BotCommand("last", "ultimi N messaggi (es. /last 20)"),
            BotCommand("break", "/break A B — interrompi loop A↔B"),
            BotCommand("help", "elenco comandi"),
        ])
        # Force the chat menu button to render as the native Telegram
        # commands list (the "MENU" pill next to the text input). Without
        # this some clients fall back to MenuButtonDefault which can show
        # a plain "/" icon.
        await self._app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        log.info("configured bot UI: commands + menu button")

    # ------------------------------------------------------------------ dispatch
    async def _dispatch_loop(self) -> None:
        assert self._app is not None
        while True:
            event = await self._queue.get()
            try:
                text = self._format(event)
                if text is None:
                    continue
                subscribers = await self._db.tg_subscribers_active()
                for chat_id in subscribers:
                    try:
                        await self._app.bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                            disable_notification=True,
                        )
                    except TelegramError as exc:
                        # 403 = user blocked / kicked the bot; auto-evict so we
                        # don't keep retrying forever and spamming logs.
                        if "Forbidden" in str(exc) or "blocked" in str(exc).lower():
                            log.info("evicting blocked subscriber", extra={"chat_id": chat_id})
                            await self._db.tg_subscriber_remove(chat_id)
                        else:
                            log.warning(
                                "telegram send failed",
                                extra={"chat_id": chat_id, "error": str(exc)},
                            )
            except Exception:  # noqa: BLE001
                log.exception("telegram observer dispatch crashed (event dropped)")

    def _format(self, event: dict[str, Any]) -> str | None:
        kind = event["kind"]
        if kind == "msg":
            env = event["envelope"]
            head = f"<b>{_h(env['from'])}</b> → <b>{_h(env['to'])}</b>"
            tags: list[str] = []
            if env.get("closing"):
                tags.append("🔚 closing")
            if env.get("in_reply_to"):
                tags.append(f"reply→<code>{_h(env['in_reply_to'])}</code>")
            tag_line = (" · " + " · ".join(tags)) if tags else ""
            body = env.get("text") or ""
            if len(body) > TELEGRAM_BODY_PREVIEW_MAX:
                body = body[:TELEGRAM_BODY_PREVIEW_MAX] + "…"
            body_block = f"<blockquote>{_h(body)}</blockquote>" if body else ""
            atts = env.get("attachments") or []
            att_lines = ""
            if atts:
                rendered = []
                for a in atts:
                    rendered.append(
                        f"📎 {_h(a.get('name', '?'))} "
                        f"({_h(a.get('mime', '?'))}, {a.get('size_bytes', 0)} B)"
                    )
                att_lines = "\n" + "\n".join(rendered)
            ts = env.get("ts", "")
            footer = f"\n<i>{_h(ts)}</i>" if ts else ""
            return f"{head}{tag_line}\n{body_block}{att_lines}{footer}"
        if kind == "presence":
            verb = "🟢 connected" if event["event"] == "up" else "🔴 disconnected"
            agent = _h(event["agent_id"])
            online = ", ".join(_h(a) for a in event["online"]) or "<i>(none)</i>"
            return f"{verb}: <b>{agent}</b>\nonline: {online}"
        return None

    # ------------------------------------------------------------------ commands
    def _is_allowed(self, update: Update) -> bool:
        user = update.effective_user
        if user is None:
            return False
        return user.id in self._allowed

    async def _reply(
        self,
        update: Update,
        text: str,
        *,
        keyboard: InlineKeyboardMarkup | None = None,
    ) -> None:
        # callback_query updates carry no effective_message; reply on the
        # originating chat through the bot directly so /menu buttons can
        # also call into _reply transparently.
        if update.callback_query is not None and update.effective_chat is not None:
            await update.callback_query.message.reply_text(
                text, parse_mode=ParseMode.HTML, reply_markup=keyboard,
            )
            return
        msg = update.effective_message
        if msg is None:
            return
        await msg.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    async def _cmd_start(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            await self._reply(update, "⛔ non sei autorizzato.")
            log.warning(
                "telegram observer unauthorized /start",
                extra={"user_id": getattr(update.effective_user, "id", None)},
            )
            return
        chat = update.effective_chat
        user = update.effective_user
        if chat is None or user is None:
            return
        await self._db.tg_subscriber_add(chat_id=chat.id, user_id=user.id)
        await self._reply(
            update,
            "✅ subscribed. Riceverai il dump real-time delle conversazioni peer.\n"
            "Usa il menu qui sotto o digita /help per la lista completa.",
            keyboard=_main_menu_keyboard(),
        )

    async def _cmd_stop(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        chat = update.effective_chat
        if chat is None:
            return
        await self._db.tg_subscriber_remove(chat.id)
        await self._reply(update, "👋 unsubscribed. /start per riattivare.")

    async def _cmd_mute(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        chat = update.effective_chat
        if chat is None:
            return
        ok = await self._db.tg_subscriber_set_muted(chat_id=chat.id, muted=True)
        await self._reply(update, "🔇 muted." if ok else "non sottoscritto. /start prima.")

    async def _cmd_unmute(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        chat = update.effective_chat
        if chat is None:
            return
        ok = await self._db.tg_subscriber_set_muted(chat_id=chat.id, muted=False)
        await self._reply(update, "🔔 unmuted." if ok else "non sottoscritto. /start prima.")

    async def _cmd_status(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        online = await self._hub_status()
        subs = await self._db.tg_subscribers_all()
        active = sum(1 for r in subs if not r["muted"])
        muted = len(subs) - active
        msg_count = await self._db.count_messages()
        await self._reply(
            update,
            f"<b>mailbox status</b>\n"
            f"peers online: {len(online)} ({_h(', '.join(online)) or '∅'})\n"
            f"subscribers: {active} active, {muted} muted\n"
            f"messages persisted: {msg_count}",
        )

    async def _cmd_peers(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        online = await self._hub_status()
        await self._reply(
            update,
            ("🟢 online: " + ", ".join(_h(a) for a in online)) if online else "nessun peer online.",
        )

    async def _cmd_last(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        n = 10
        if ctx.args:
            try:
                n = max(1, min(int(ctx.args[0]), RECENT_PREVIEW_MAX))
            except ValueError:
                await self._reply(update, "uso: /last N (intero, max 50)")
                return
        rows = await self._db.fetch_recent_messages(limit=n)
        if not rows:
            await self._reply(update, "nessun messaggio.")
            return
        lines: list[str] = []
        for r in rows:
            tag = " 🔚" if r["closing"] else ""
            body = (r["body"] or "").splitlines()[0] if r["body"] else ""
            if len(body) > 120:
                body = body[:120] + "…"
            lines.append(
                f"<i>{_h(r['created_at'])}</i> "
                f"<b>{_h(r['from_agent'])}</b>→<b>{_h(r['to_agent'])}</b>{tag}: "
                f"{_h(body)}"
            )
        await self._reply(update, "\n".join(lines))

    async def _cmd_break(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Inject a bilateral ``closing=true`` frame between two peers.

        Manual circuit breaker for chat loops: when peer A and peer B keep
        replying to each other, the operator runs ``/break A B`` and the
        observer asks the hub to forward a synthetic closing message in
        each direction. The receiving channel honours ``closing=true`` by
        skipping ``publish_inbound`` (see ``nanobot/channels/peer.py``),
        so neither agent wakes up on the synthetic frame and the loop dies.
        Works while peers are online or offline (queued for delivery).
        """
        if not self._is_allowed(update):
            return
        if self._forward is None:
            await self._reply(update, "⚠️ /break non disponibile (forward callable non agganciato).")
            return
        if not ctx.args or len(ctx.args) != 2:
            await self._reply(
                update,
                "uso: <code>/break peerA peerB</code> — invia closing=true bilaterale.",
            )
            return
        a, b = ctx.args[0].strip().lower(), ctx.args[1].strip().lower()
        text = "[break operatore] conversazione chiusa via mailbox observer"
        sent: list[str] = []
        errors: list[str] = []
        for src, dst in ((a, b), (b, a)):
            try:
                envelope = await self._forward(
                    from_agent=src,
                    to_agent=dst,
                    text=text,
                    thread_id=None,
                    in_reply_to=None,
                    closing=True,
                )
                sent.append(f"{src}→{dst} <code>{_h(envelope['id'])}</code>")
            except ValueError as exc:
                errors.append(f"{src}→{dst}: {_h(str(exc))}")
        head = "🛑 <b>break</b>" if not errors else "⚠️ <b>break (parziale)</b>"
        body = "\n".join(["✅ " + s for s in sent] + ["❌ " + e for e in errors])
        await self._reply(update, f"{head}\n{body}")

    async def _cmd_help(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        await self._reply(
            update,
            "<b>comandi</b>\n"
            "/start — subscribe live dump\n"
            "/menu — menu interattivo\n"
            "/stop — unsubscribe\n"
            "/mute — silenzia (resta subscriber)\n"
            "/unmute — riattiva\n"
            "/status — stato servizio\n"
            "/peers — peer online ora\n"
            "/last N — ultimi N messaggi (max 50)\n"
            "/break A B — interrompi loop A↔B (closing=true bilaterale)\n"
            "/help — questo messaggio",
            keyboard=_main_menu_keyboard(),
        )

    async def _cmd_menu(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        await self._reply(
            update,
            "<b>📡 mailbox observer</b> — scegli un'azione:",
            keyboard=_main_menu_keyboard(),
        )

    async def _on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Dispatch inline-keyboard taps onto the same handlers as commands.

        Telegram requires answering every callback_query (any answer, even
        empty) within ~10s, otherwise the spinner on the user's side never
        clears. We answer first, then run the handler.
        """
        query = update.callback_query
        if query is None:
            return
        # answer() before any DB work so the user gets immediate feedback.
        try:
            await query.answer()
        except TelegramError:
            pass

        if not self._is_allowed(update):
            await self._reply(update, "⛔ non sei autorizzato.")
            return

        data = query.data or ""
        if data == CB_PEERS:
            await self._cmd_peers(update, ctx)
        elif data == CB_STATUS:
            await self._cmd_status(update, ctx)
        elif data == CB_LAST_10:
            ctx.args = ["10"]
            await self._cmd_last(update, ctx)
        elif data == CB_LAST_30:
            ctx.args = ["30"]
            await self._cmd_last(update, ctx)
        elif data == CB_MUTE:
            await self._cmd_mute(update, ctx)
        elif data == CB_UNMUTE:
            await self._cmd_unmute(update, ctx)
        elif data == CB_STOP:
            await self._cmd_stop(update, ctx)
        elif data == CB_HELP:
            await self._cmd_help(update, ctx)
        else:
            log.warning("unknown callback_data", extra={"data": data})


def build_observer_from_env(
    *, db: Database, hub_status_provider, forward_callable=None,
) -> TelegramObserver | None:
    """Construct a TelegramObserver from environment, or return None if the
    bot token is not configured (observer disabled).
    """
    token = os.environ.get("MAILBOX_TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None
    allowed = parse_allowed_users(os.environ.get("MAILBOX_TELEGRAM_ALLOWED_USERS"))
    if not allowed:
        log.warning(
            "MAILBOX_TELEGRAM_BOT_TOKEN set but MAILBOX_TELEGRAM_ALLOWED_USERS empty:"
            " bot will reject every command until the allow-list is populated"
        )
    base_url = os.environ.get("MAILBOX_TELEGRAM_BASE_URL", "").strip() or None
    base_file_url = os.environ.get("MAILBOX_TELEGRAM_BASE_FILE_URL", "").strip() or None
    return TelegramObserver(
        token=token,
        allowed_users=allowed,
        db=db,
        hub_status_provider=hub_status_provider,
        forward_callable=forward_callable,
        base_url=base_url,
        base_file_url=base_file_url,
    )
