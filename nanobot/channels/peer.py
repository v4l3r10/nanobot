"""Peer channel: long-lived WebSocket client to the inter-agent mailbox router.

Each nanobot gateway opens **one** persistent connection to the mailbox
``/peer`` endpoint and stays connected for the lifetime of the process. The
hub on the other end forwards frames between connected peers.

Each remote peer the bot talks to is exposed locally as a distinct chat:
``chat_id = "<from_agent>"``. So a bot named *bronzo* talking with both
*grocco* and *naldo* sees two independent chat threads with their own session,
history, and persona-aware turns. There is no separate "RPC" mode — the bot
chats with peers exactly as it chats with humans, only the channel is
different. (Legacy: pre-2026-05-08 deployments used ``"peer:<from_agent>"``;
the helpers below still accept that form for backward compatibility.)

File transfer: outbound files attached via :class:`OutboundMessage.media` are
uploaded to the router's HTTP ``/files/`` endpoint (with the same bearer used
for the WebSocket), the resulting attachment ids ride in the frame, and on the
recipient side this channel downloads the blobs into a local cache and
populates ``InboundMessage.media`` with the local paths so the agent treats
peer files exactly like Telegram media.

Wire-level frame: see ``nanobot_mailbox.peer_router`` in the mailbox service.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import Field
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_workspace_path
from nanobot.config.schema import Base

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


# Pre-2026-05-08 chat_ids carried this self-tag (the channel re-prefixed every
# id with "peer:" before handing it to the bus, which then layered another
# "peer:" on top to form the session_key — producing "peer:peer:<agent>"
# session keys and disk filenames). We dropped the self-tag because the
# OutboundMessage.channel field already carries that information; chat_id
# now stores the bare peer agent_id. The constant survives only so legacy
# stored state (cron jobs, queued OutboundMessages) keeps routing.
LEGACY_CHAT_ID_PREFIX = "peer:"
FRAME_VERSION = 1
# 1 MB cap for an inbound WS frame payload. Body itself is ≤16 KB at the
# router; the headroom covers attachment metadata and JSON envelope overhead.
MAX_FRAME_BYTES = 1024 * 1024


def _peer_chat_id(agent: str) -> str:
    return agent


def _peer_from_chat_id(chat_id: str) -> str | None:
    if not chat_id:
        return None
    # Backward compat: accept the legacy "peer:<agent>" form transparently
    # so a stored OutboundMessage from before the cleanup still routes.
    if chat_id.startswith(LEGACY_CHAT_ID_PREFIX):
        chat_id = chat_id[len(LEGACY_CHAT_ID_PREFIX):]
    return chat_id or None


def _ws_to_http_base(ws_url: str) -> str:
    """Derive the HTTP base URL from the WS URL.

    ``ws://host:port/peer`` -> ``http://host:port``
    ``wss://host:port/peer`` -> ``https://host:port``
    """
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    netloc = parsed.netloc or parsed.path  # tolerate URL-without-scheme inputs
    return f"{scheme}://{netloc}"


_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_FILENAME.sub("_", name).strip("._") or "file"
    return cleaned[:120]


class PeerConfig(Base):
    """Configuration for the peer-to-peer mailbox channel."""

    enabled: bool = False
    router_url: str = "ws://nanobot-mailbox:8765/peer"
    agent_id: str = ""
    token: str = ""
    # Reconnection backoff. The router treats a peer as "online" only while
    # its WebSocket is registered; while offline, frames addressed to this
    # bot accumulate in the router DB and are drained at reconnect.
    reconnect_initial_delay_s: float = Field(default=1.0, ge=0.1, le=60.0)
    reconnect_max_delay_s: float = Field(default=30.0, ge=1.0, le=600.0)
    # Transport-layer keepalive. The websockets library handles ping/pong
    # automatically; if pong does not arrive within ping_timeout_s the
    # connection is considered dead and reconnect kicks in.
    ping_interval_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ping_timeout_s: float = Field(default=20.0, ge=5.0, le=300.0)
    # Authorize incoming peer messages. ``"*"`` accepts any peer the router
    # decides to forward to us (the router has already authenticated the
    # sender). Restrict to a list of agent_ids if you want a tighter ACL.
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    streaming: bool = False  # peer plane is request/response, not streamed
    # Subdir of the workspace media root where peer-received files land.
    media_subdir: str = "peer"
    # HTTP timeouts for the file-transfer endpoints (upload/download share).
    file_op_timeout_s: float = Field(default=60.0, ge=5.0, le=600.0)
    # Garbage-collect downloaded peer attachments older than this. The cache
    # is keyed by message id, so each conversation's files live in their own
    # subdir under <workspace>/<media_subdir>/. Set to 0 to disable cleanup.
    media_max_age_days: int = Field(default=7, ge=0, le=365)
    media_cleanup_interval_h: float = Field(default=24.0, ge=0.5, le=168.0)
    # How long send() waits for the router's ack/error frame before declaring
    # the send "submitted but not confirmed". On timeout we don't raise: the
    # frame is on the wire, the router will either persist + deliver it or log
    # a warning. Lowering this just shortens the agent's confirmation wait.
    ack_timeout_s: float = Field(default=10.0, ge=1.0, le=120.0)


class PeerChannel(BaseChannel):
    """Persistent WebSocket client to the mailbox peer hub."""

    name = "peer"
    display_name = "Peer Mailbox"

    def __init__(self, config: PeerConfig | dict[str, Any], bus: MessageBus):
        if isinstance(config, dict):
            config = PeerConfig.model_validate(config)
        super().__init__(config, bus)
        self._cfg: PeerConfig = config
        self._ws: ClientConnection | None = None
        self._ws_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._known_peers: set[str] = set()
        self._http: httpx.AsyncClient | None = None
        self._cleanup_task: asyncio.Task | None = None
        # Pending sends keyed by client_ref. Resolved by ack frames or rejected
        # by error frames coming back from the router. Cancelled on disconnect
        # so a pending send() doesn't hang past a session boundary.
        self._pending_acks: dict[str, asyncio.Future] = {}

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return PeerConfig().model_dump()

    # --- public API used by the peer tool registry --------------------------

    def list_peers(self) -> list[str]:
        """Return the most recent online peer roster pushed by the router.

        Excludes self. The list is updated whenever the router sends a
        ``presence`` frame (on any peer connect/disconnect). Empty until the
        first such frame is received after start.
        """
        return sorted(p for p in self._known_peers if p != self._cfg.agent_id)

    @property
    def own_agent_id(self) -> str:
        return self._cfg.agent_id

    async def fetch_thread(
        self, peer: str, *, last_n: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch up to *last_n* recent messages exchanged with *peer* from
        the router's authoritative DB. Used by the ``peer_thread_show`` tool.

        Returns messages in chronological order (oldest→newest) as a list of
        dicts with keys: id, from, to, text, ts, in_reply_to, thread_id,
        delivered. Empty list if no messages exist.

        Raises RuntimeError on auth/network failure so the tool can surface
        the issue to the agent rather than pretend the thread is empty.
        """
        if self._http is None:
            raise RuntimeError("peer: HTTP client not initialized")
        resp = await self._http.get(
            "/messages", params={"peer": peer, "last_n": last_n},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"peer router /messages returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        data = resp.json()
        return list(data.get("messages", []))

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        if not self._cfg.agent_id:
            logger.error("peer: missing agent_id; channel disabled")
            return
        if not self._cfg.token:
            logger.error("peer: missing token; channel disabled")
            return
        self._running = True
        self._stop_event.clear()
        self._http = httpx.AsyncClient(
            base_url=_ws_to_http_base(self._cfg.router_url),
            timeout=self._cfg.file_op_timeout_s,
            headers={"Authorization": f"Bearer {self._cfg.token}"},
        )
        self._task = asyncio.create_task(self._supervisor(), name="peer-supervisor")
        if self._cfg.media_max_age_days > 0:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(), name="peer-media-cleanup",
            )
        logger.info(
            "peer channel started (agent_id={}, router={})",
            self._cfg.agent_id, self._cfg.router_url,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close(code=1001, reason="shutdown")
            except Exception:
                pass
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self._task = None
        self._ws = None
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except (asyncio.CancelledError, Exception):
                pass
            self._cleanup_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # --- supervisor: connect with exponential backoff, drain frames ---------

    async def _supervisor(self) -> None:
        delay = self._cfg.reconnect_initial_delay_s
        while not self._stop_event.is_set():
            try:
                await self._run_session()
                # Clean disconnect by remote — reset backoff so the next
                # reconnect happens fast.
                delay = self._cfg.reconnect_initial_delay_s
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError, InvalidStatus) as exc:
                logger.warning(
                    "peer disconnected ({}: {}); reconnect in {:.1f}s",
                    type(exc).__name__, exc, delay,
                )
            except Exception as exc:  # noqa: BLE001 — never let supervisor die
                logger.exception("peer session crashed: {}", exc)
            if self._stop_event.is_set():
                break
            jitter = random.uniform(0, delay * 0.25)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=delay + jitter
                )
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, self._cfg.reconnect_max_delay_s)

    async def _run_session(self) -> None:
        headers = [("Authorization", f"Bearer {self._cfg.token}")]
        async with connect(
            self._cfg.router_url,
            additional_headers=headers,
            ping_interval=self._cfg.ping_interval_s,
            ping_timeout=self._cfg.ping_timeout_s,
            max_size=MAX_FRAME_BYTES,
        ) as ws:
            async with self._ws_lock:
                self._ws = ws
            logger.info("peer connected (agent_id={})", self._cfg.agent_id)
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    await self._on_frame(raw)
            finally:
                async with self._ws_lock:
                    if self._ws is ws:
                        self._ws = None
                # On disconnect we conservatively clear known peers; the next
                # presence frame from the router after reconnect will refresh
                # the roster. This avoids stale "online" reports during the
                # gap.
                self._known_peers.clear()
                # Fail any sends that were waiting for an ack on this session;
                # the next reconnect starts a fresh request/response window.
                self._fail_pending_acks(ConnectionError("peer session ended"))

    def _fail_pending_acks(self, exc: BaseException) -> None:
        for fut in list(self._pending_acks.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending_acks.clear()

    # --- inbound ------------------------------------------------------------

    async def _on_frame(self, raw: str) -> None:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("peer bad inbound frame: {}", exc)
            return
        ftype = frame.get("type", "msg")
        if ftype == "presence":
            online = frame.get("online", [])
            if isinstance(online, list):
                self._known_peers = {x for x in online if isinstance(x, str)}
                logger.info(
                    "peer presence (online={})", sorted(self._known_peers),
                )
            return
        if ftype in ("ack", "error"):
            self._resolve_ack_frame(ftype, frame)
            return
        if ftype != "msg":
            logger.debug("peer ignoring non-msg frame type={}", ftype)
            return
        from_agent = frame.get("from")
        text = frame.get("text", "")
        if not isinstance(from_agent, str) or not from_agent:
            logger.warning("peer frame missing from")
            return
        if not isinstance(text, str):
            logger.warning("peer frame bad text type")
            return

        # closing=True signals "this is the sender's last word — don't expect
        # a reply". The router has already persisted the row, so the message
        # is visible via peer_thread_show. We deliberately skip publish_inbound
        # here so the agent loop is NOT awakened: this is the system-level
        # break that prevents pleasantry/echo loops without any LLM call.
        if frame.get("closing") is True:
            logger.info(
                "peer closing received from {} (msg_id={}); agent not awoken",
                from_agent, frame.get("id"),
            )
            return

        media_paths: list[str] = []
        attachments = frame.get("attachments") or []
        if isinstance(attachments, list):
            media_paths = await self._download_attachments(
                attachments, msg_id=str(frame.get("id", "unknown"))
            )

        await self._handle_message(
            sender_id=from_agent,
            chat_id=_peer_chat_id(from_agent),
            content=text,
            media=media_paths or None,
            metadata={
                "peer_message_id": frame.get("id"),
                "peer_thread_id": frame.get("thread_id"),
                "peer_in_reply_to": frame.get("in_reply_to"),
                "peer_ts": frame.get("ts"),
                "peer_attachments": attachments,
            },
        )

    def _resolve_ack_frame(self, ftype: str, frame: dict[str, Any]) -> None:
        """Resolve a pending send() Future on receipt of an ack/error frame.

        ``client_ref`` is matched against the table populated in send(). If
        unknown (rare: late arrival after a timeout / reconnect), the frame is
        dropped with a debug log so we never crash on routing replays.
        """
        client_ref = frame.get("client_ref")
        if not isinstance(client_ref, str):
            return
        fut = self._pending_acks.pop(client_ref, None)
        if fut is None or fut.done():
            logger.debug(
                "peer ack/error for unknown client_ref ({}, type={})",
                client_ref, ftype,
            )
            return
        if ftype == "ack":
            fut.set_result(frame)
        else:
            reason = frame.get("reason", "router rejected the send")
            fut.set_exception(RuntimeError(f"peer router rejected: {reason}"))

    async def _download_attachments(
        self, attachments: list[dict[str, Any]], *, msg_id: str
    ) -> list[str]:
        if self._http is None:
            return []
        target_dir = (
            get_workspace_path() / self._cfg.media_subdir / _safe_filename(msg_id)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        results: list[str] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            att_id = att.get("id")
            if not isinstance(att_id, int):
                continue
            name = _safe_filename(str(att.get("name", f"att_{att_id}")))
            dest = target_dir / name
            try:
                async with self._http.stream(
                    "GET", f"/files/{att_id}",
                ) as resp:
                    if resp.status_code != 200:
                        logger.warning(
                            "peer attachment download failed (id={}, status={})",
                            att_id, resp.status_code,
                        )
                        continue
                    with dest.open("wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                            f.write(chunk)
                results.append(str(dest))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "peer attachment download error (id={}): {}", att_id, exc
                )
        return results

    # --- outbound -----------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        meta = msg.metadata or {}
        # Filter out progress/streaming events meant for human-facing channels.
        # Without this guard, the agent loop's progress callbacks (tool hints,
        # streaming deltas, retry-wait notices, stream-end markers) would be
        # forwarded as peer messages — they have no closing=true so they wake
        # the receiver's agent loop, generating a ghost-twin loop alongside
        # legitimate peer_say outputs. Peer plane is request/response only.
        for noise_flag in (
            "_progress", "_tool_hint", "_retry_wait",
            "_stream_delta", "_stream_end", "_streamed",
        ):
            if meta.get(noise_flag):
                return
        # Empty / whitespace-only content also has no place on the peer plane:
        # it's typically a bookkeeping placeholder from the agent loop and
        # would arrive at the receiver as a wake-up-with-no-payload.
        if not (msg.content or "").strip():
            logger.debug("peer skipping empty outbound (no content to deliver)")
            return

        peer = _peer_from_chat_id(msg.chat_id)
        if peer is None:
            raise ValueError(
                f"peer: chat_id {msg.chat_id!r} is empty or invalid (expected '<agent>')"
            )
        if peer == self._cfg.agent_id:
            raise ValueError("peer: cannot send to self")
        ws = self._ws
        if ws is None:
            raise RuntimeError("peer: not connected to router")

        attachment_ids: list[int] = []
        if msg.media:
            # All-or-nothing: if a single upload fails, abort the whole send.
            # Partial deliveries would silently drop files the agent expected
            # the recipient to see.
            attachment_ids = await self._upload_media(msg.media)

        client_ref = uuid.uuid4().hex
        frame: dict[str, Any] = {
            "v": FRAME_VERSION,
            "type": "msg",
            "to": peer,
            "text": msg.content,
            "thread_id": meta.get("peer_thread_id"),
            "in_reply_to": meta.get("peer_in_reply_to"),
            "client_ref": client_ref,
        }
        if attachment_ids:
            frame["attachment_ids"] = attachment_ids
        # closing=True is the explicit "thread terminator" flag set by the
        # peer_say tool when the agent decides this is its last word in the
        # exchange. Routed unmodified; the recipient channel uses it to skip
        # waking its agent loop, breaking pleasantry loops by design.
        if meta.get("peer_closing") is True:
            frame["closing"] = True

        loop = asyncio.get_running_loop()
        ack_future: asyncio.Future = loop.create_future()
        self._pending_acks[client_ref] = ack_future
        try:
            await ws.send(json.dumps(frame, ensure_ascii=False))
            try:
                await asyncio.wait_for(ack_future, timeout=self._cfg.ack_timeout_s)
            except asyncio.TimeoutError:
                # The frame was written to the wire; the router will still
                # process it. We treat this as "submitted, unconfirmed" and
                # surface a warning rather than an error so the agent's UX
                # isn't disrupted by transient router latency.
                self._pending_acks.pop(client_ref, None)
                logger.warning(
                    "peer ack timeout (to={}, ref={}); send is fire-and-forget",
                    peer, client_ref,
                )
        finally:
            self._pending_acks.pop(client_ref, None)

        logger.debug(
            "peer sent (to={}, bytes={}, attachments={})",
            peer, len(msg.content.encode("utf-8")), len(attachment_ids),
        )

    # --- media cache cleanup -----------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Periodically prune downloaded peer attachments older than the TTL.

        Each inbound message creates a subdir under <workspace>/<media_subdir>/
        keyed by message id; once the agent has processed the turn, the local
        path is no longer referenced. We rmtree subdirs whose mtime is older
        than ``media_max_age_days``.
        """
        interval_s = self._cfg.media_cleanup_interval_h * 3600.0
        # Run once shortly after start, then on the regular interval.
        first_delay = min(60.0, interval_s)
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=first_delay)
            return
        except asyncio.TimeoutError:
            pass
        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self._prune_media_cache_sync)
            except Exception as exc:  # noqa: BLE001
                logger.warning("peer media cleanup failed: {}", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_s)
                return
            except asyncio.TimeoutError:
                continue

    def _prune_media_cache_sync(self) -> None:
        root = get_workspace_path() / self._cfg.media_subdir
        if not root.is_dir():
            return
        cutoff = time.time() - (self._cfg.media_max_age_days * 86400.0)
        removed = 0
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "peer media cleanup could not remove {}: {}", entry, exc
                    )
        if removed:
            logger.info("peer media cleanup pruned {} cached message dirs", removed)

    async def _upload_media(self, media: list[str]) -> list[int]:
        """Upload every path in *media* to the router. Raises RuntimeError if
        any single upload fails: the message is then never sent, surfacing the
        problem to the caller instead of silently shipping a partial set.
        """
        if self._http is None:
            raise RuntimeError("peer: HTTP client not initialized")
        ids: list[int] = []
        for path_str in media:
            path = Path(path_str)
            if not path.is_file():
                raise RuntimeError(f"peer upload: not a file: {path_str}")
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            try:
                with path.open("rb") as f:
                    files = {"file": (path.name, f, mime)}
                    resp = await self._http.post("/files/", files=files)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"peer upload error for {path_str}: {exc}"
                ) from exc
            if resp.status_code != 200:
                raise RuntimeError(
                    f"peer upload failed ({path_str}): "
                    f"HTTP {resp.status_code} — {resp.text[:200]}"
                )
            try:
                data = resp.json()
                ids.append(int(data["file_id"]))
            except (KeyError, ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"peer upload bad response for {path_str}: {exc}"
                ) from exc
        return ids
