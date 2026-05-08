"""Peer-to-peer WebSocket hub.

Each nanobot gateway opens one long-lived WebSocket connection to ``/peer``.
The hub maintains a live ``agent_id -> connection`` registry and forwards
frames between connected peers. Frames are persisted via :class:`Database` so
that messages addressed to an offline peer survive until reconnection, at
which point the queue is drained in chronological order.

Wire-level frame (JSON, one frame per WS message):

    {
        "v": 1,
        "type": "msg" | "presence" | "ping",
        "from": "<agent_id>",   # ignored on send (server overrides with auth)
        "to":   "<agent_id>",
        "text": "<plain text>",
        "thread_id": "thr_<ULID> | null",
        "in_reply_to": "msg_<ULID> | null",
        "attachments": [           # optional, for file transfer
            {"id": <int>, "name": "<str>", "mime": "<str>", "size_bytes": <int>}
        ],
        "id": "msg_<ULID>",     # set by server on outbound forward
        "ts": "<ISO 8601 UTC>"  # set by server on outbound forward
    }

Presence frames are broadcast by the hub when a peer connects or disconnects:

    {"v": 1, "type": "presence", "online": ["bronzo", "grocco"]}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from .auth import TokenRegistry, extract_bearer
from .db import Database
from .protocol import BODY_MAX_BYTES, SUBJECT_MAX, is_valid_agent_id, utcnow_iso

log = logging.getLogger(__name__)

FRAME_VERSION = 1
INBOX_DRAIN_BATCH = 100


class PeerHub:
    """Live registry of agent_id -> active WebSocket connection.

    All mutating operations are guarded by an asyncio lock so that connect /
    disconnect / forward are linearizable. Per-agent forward concurrency is
    additionally serialized via per-target asyncio queues to preserve frame
    order on a single recipient.
    """

    def __init__(self, db: Database, registry: TokenRegistry):
        self._db = db
        self._registry = registry
        self._conns: dict[str, WebSocket] = {}
        self._send_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def register(self, agent_id: str, ws: WebSocket) -> WebSocket | None:
        """Register *ws* as the live connection for *agent_id*.

        If a previous connection exists for the same agent it is returned so
        the caller can close it: only one live connection per agent is kept.
        """
        async with self._lock:
            previous = self._conns.get(agent_id)
            self._conns[agent_id] = ws
            self._send_locks.setdefault(agent_id, asyncio.Lock())
            return previous

    async def unregister(self, agent_id: str, ws: WebSocket) -> None:
        async with self._lock:
            if self._conns.get(agent_id) is ws:
                self._conns.pop(agent_id, None)

    def is_online(self, agent_id: str) -> bool:
        return agent_id in self._conns

    async def online_peers(self) -> list[str]:
        async with self._lock:
            return sorted(self._conns.keys())

    async def broadcast_presence(self) -> None:
        """Send the current online roster to every connected peer.

        Called after register/unregister so each peer keeps an up-to-date view
        of who is reachable in real time. Failures on individual sockets are
        logged but never block the broadcast.
        """
        async with self._lock:
            online = sorted(self._conns.keys())
            targets = list(self._conns.items())
        frame = {"v": FRAME_VERSION, "type": "presence", "online": online}
        payload = json.dumps(frame, ensure_ascii=False)
        for agent_id, ws in targets:
            if ws.client_state != WebSocketState.CONNECTED:
                continue
            send_lock = self._send_locks.setdefault(agent_id, asyncio.Lock())
            async with send_lock:
                try:
                    await ws.send_text(payload)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "presence broadcast failed",
                        extra={"to": agent_id, "error": f"{type(exc).__name__}: {exc}"},
                    )

    async def forward(
        self,
        *,
        from_agent: str,
        to_agent: str,
        text: str,
        thread_id: str | None,
        in_reply_to: str | None,
        attachment_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Persist a frame and forward to recipient if connected.

        Returns the persisted envelope dict (as it would be wired to the
        recipient). Delivery failures (recipient offline, send raised) leave
        the row as ``delivered_at IS NULL`` so it can be drained later.

        ``attachment_ids`` reference rows previously created by the sender via
        ``POST /files/`` (those rows have ``message_id IS NULL``); on insert
        they are linked to the new message so the recipient can later GET
        them with the same bearer token. Validity is checked *before* the
        message row is inserted to avoid orphan rows on bad input.
        """
        if from_agent == to_agent:
            raise ValueError("cannot send to self")
        if to_agent not in self._registry.known_agents:
            raise ValueError(f"unknown recipient: {to_agent}")
        if len(text.encode("utf-8")) > BODY_MAX_BYTES:
            raise ValueError(f"text exceeds {BODY_MAX_BYTES} bytes")
        # Pre-check attachment ownership/availability so we don't insert a
        # message row that would later have to be rolled back.
        if attachment_ids and not await self._db.verify_attachments_available(
            attachment_ids=attachment_ids, owner_agent=from_agent,
        ):
            raise ValueError(
                "one or more attachment_ids are not owned by sender or already linked"
            )
        # Subject column is NOT NULL in the schema; we synthesize a short
        # prefix from the body so the row is well-formed. Nothing on the peer
        # plane reads subject — it's a residual constraint.
        subject = (text[:SUBJECT_MAX] or "(empty)").splitlines()[0][:SUBJECT_MAX]

        msg_id, thread = await self._db.insert_message(
            from_agent=from_agent,
            to_agent=to_agent,
            type="notification",
            subject=subject,
            body=text,
            in_reply_to=in_reply_to,
            thread_id=thread_id,
        )

        attachments_meta: list[dict[str, Any]] = []
        if attachment_ids:
            linked = await self._db.attach_to_message(
                attachment_ids=attachment_ids,
                message_id=msg_id,
                owner_agent=from_agent,
            )
            # Defensive: pre-check passed, so this should always match. If a
            # concurrent forward stole one of the ids in the gap, that's the
            # only realistic way to get a mismatch here.
            if linked != len(attachment_ids):
                raise ValueError(
                    "attachment race: another sender consumed an id between"
                    " precheck and link"
                )
            for att_id in attachment_ids:
                row = await self._db.fetch_attachment(att_id)
                if row is None:
                    continue
                attachments_meta.append({
                    "id": row["id"],
                    "name": row["filename"],
                    "mime": row["mime"],
                    "size_bytes": row["size_bytes"],
                })

        envelope = self._wire_envelope(
            id=msg_id, from_agent=from_agent, to_agent=to_agent,
            text=text, thread_id=thread, in_reply_to=in_reply_to,
            attachments=attachments_meta,
        )
        delivered = await self._deliver(to_agent, envelope)
        if delivered:
            await self._db.mark_delivered([msg_id])
        return envelope

    async def drain_inbox(self, agent_id: str) -> int:
        """Forward every pending (delivered_at IS NULL) message for *agent_id*.

        Called after registration so a peer that comes back online sees its
        accumulated traffic in chronological order. Returns the count drained.
        """
        rows = await self._db.fetch_pending_delivery(
            agent_id, limit=INBOX_DRAIN_BATCH,
        )
        drained: list[str] = []
        for row in rows:
            attachments_meta: list[dict[str, Any]] = []
            for att_row in await self._db.fetch_message_attachments_meta(row["id"]):
                attachments_meta.append({
                    "id": att_row["id"],
                    "name": att_row["filename"],
                    "mime": att_row["mime"],
                    "size_bytes": att_row["size_bytes"],
                })
            envelope = self._wire_envelope(
                id=row["id"],
                from_agent=row["from_agent"],
                to_agent=row["to_agent"],
                text=row["body"],
                thread_id=row["thread_id"],
                in_reply_to=row["in_reply_to"],
                attachments=attachments_meta,
                ts=row["created_at"],
            )
            if not await self._deliver(agent_id, envelope):
                break
            drained.append(row["id"])
        if drained:
            await self._db.mark_delivered(drained)
        return len(drained)

    async def notify_sender(self, agent_id: str, frame: dict[str, Any]) -> None:
        """Send an out-of-band control frame (ack / error) to *agent_id*.

        Used to confirm or reject a peer's submitted msg frame so the sender
        knows whether its ``peer_say`` actually got persisted. Best-effort:
        if the connection is gone the caller has nothing to retry against,
        so we just log.
        """
        ws = self._conns.get(agent_id)
        if ws is None or ws.client_state != WebSocketState.CONNECTED:
            log.debug(
                "notify_sender skipped (sender offline)",
                extra={"agent_id": agent_id, "frame_type": frame.get("type")},
            )
            return
        send_lock = self._send_locks.setdefault(agent_id, asyncio.Lock())
        async with send_lock:
            try:
                await ws.send_text(json.dumps(frame, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "notify_sender failed",
                    extra={"to": agent_id, "error": f"{type(exc).__name__}: {exc}"},
                )

    async def _deliver(self, agent_id: str, envelope: dict[str, Any]) -> bool:
        ws = self._conns.get(agent_id)
        if ws is None or ws.client_state != WebSocketState.CONNECTED:
            return False
        send_lock = self._send_locks.setdefault(agent_id, asyncio.Lock())
        async with send_lock:
            try:
                await ws.send_text(json.dumps(envelope, ensure_ascii=False))
                return True
            except Exception as exc:  # noqa: BLE001 — log everything, drop connection
                log.warning(
                    "peer deliver failed",
                    extra={"to": agent_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                # The websocket is in an inconsistent state; let the read loop
                # observe the close and unregister. Caller treats False as
                # "still pending, drain on reconnect".
                return False

    @staticmethod
    def _wire_envelope(
        *,
        id: str,
        from_agent: str,
        to_agent: str,
        text: str,
        thread_id: str | None,
        in_reply_to: str | None,
        attachments: list[dict[str, Any]] | None = None,
        ts: str | None = None,
    ) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "v": FRAME_VERSION,
            "type": "msg",
            "id": id,
            "from": from_agent,
            "to": to_agent,
            "text": text,
            "thread_id": thread_id,
            "in_reply_to": in_reply_to,
            "ts": ts or utcnow_iso(),
        }
        if attachments:
            envelope["attachments"] = attachments
        return envelope


def make_peer_endpoint(hub: PeerHub, registry: TokenRegistry):
    """Build a Starlette WebSocket endpoint bound to *hub*."""

    async def endpoint(ws: WebSocket) -> None:
        bearer = extract_bearer(ws.headers.get("authorization"))
        # Fallback for clients that cannot set custom headers (legacy proxies,
        # browser-side JS without a proxy). Mirrors the existing notifier
        # convention so we do not regress reachability.
        if bearer is None:
            bearer = ws.query_params.get("token")
        ctx = registry.resolve(bearer)
        if ctx is None:
            await ws.close(code=4401)  # 4401 = app-level "unauthorized"
            return

        await ws.accept()
        agent_id = ctx.agent_id
        previous = await hub.register(agent_id, ws)
        if previous is not None:
            try:
                await previous.close(code=4409)  # superseded
            except Exception:
                pass

        log.info("peer connected", extra={"agent_id": agent_id})
        # Presence change: notify all currently-connected peers (including
        # this one) of the new roster. Done outside the registry lock so we
        # do not hold it across a fan-out.
        await hub.broadcast_presence()
        try:
            # Drain pending offline traffic in successive batches; a single
            # call is capped at INBOX_DRAIN_BATCH (100). Loop until either
            # the queue is empty or a delivery fails (in which case the
            # connection is likely already dead and the read loop will
            # observe the close).
            total_drained = 0
            while True:
                drained = await hub.drain_inbox(agent_id)
                total_drained += drained
                if drained < INBOX_DRAIN_BATCH:
                    break
            if total_drained:
                log.info(
                    "peer inbox drained",
                    extra={"agent_id": agent_id, "drained": total_drained},
                )
            await _read_loop(hub, agent_id, ws)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.exception("peer endpoint crashed", extra={"agent_id": agent_id})
            try:
                await ws.close(code=1011, reason=str(exc)[:100])
            except Exception:
                pass
        finally:
            await hub.unregister(agent_id, ws)
            log.info("peer disconnected", extra={"agent_id": agent_id})
            await hub.broadcast_presence()

    return endpoint


async def _read_loop(hub: PeerHub, agent_id: str, ws: WebSocket) -> None:
    while True:
        raw = await ws.receive_text()
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning(
                "peer bad frame",
                extra={"agent_id": agent_id, "error": str(exc)},
            )
            continue
        await _handle_frame(hub, agent_id, frame)


async def _handle_frame(hub: PeerHub, from_agent: str, frame: dict) -> None:
    ftype = frame.get("type", "msg")
    if ftype == "ping":
        return  # transport-layer ping/pong is handled by websockets itself
    if ftype != "msg":
        log.debug("peer unsupported frame type", extra={"type": ftype})
        return
    # Optional client-generated correlation id. When present we echo it back
    # in an ack/error frame so the sender can resolve a pending Future and
    # surface success or failure to its agent.
    client_ref_raw = frame.get("client_ref")
    client_ref = client_ref_raw if isinstance(client_ref_raw, str) else None

    async def _nack(reason: str) -> None:
        if client_ref is None:
            return
        await hub.notify_sender(from_agent, {
            "v": FRAME_VERSION,
            "type": "error",
            "client_ref": client_ref,
            "reason": reason,
        })

    to_agent = frame.get("to")
    text = frame.get("text", "")
    if not isinstance(to_agent, str) or not is_valid_agent_id(to_agent):
        log.warning("peer frame bad recipient", extra={"to": to_agent})
        await _nack("bad recipient")
        return
    if not isinstance(text, str):
        log.warning("peer frame bad text type")
        await _nack("text must be a string")
        return
    raw_attachments = frame.get("attachment_ids") or []
    if not isinstance(raw_attachments, list) or not all(
        isinstance(x, int) for x in raw_attachments
    ):
        log.warning("peer frame bad attachment_ids type")
        await _nack("attachment_ids must be a list of ints")
        return
    try:
        envelope = await hub.forward(
            from_agent=from_agent,
            to_agent=to_agent,
            text=text,
            thread_id=frame.get("thread_id"),
            in_reply_to=frame.get("in_reply_to"),
            attachment_ids=raw_attachments or None,
        )
    except ValueError as exc:
        log.warning(
            "peer forward rejected",
            extra={"from_agent": from_agent, "to": to_agent, "error": str(exc)},
        )
        await _nack(str(exc))
        return
    if client_ref is not None:
        await hub.notify_sender(from_agent, {
            "v": FRAME_VERSION,
            "type": "ack",
            "client_ref": client_ref,
            "id": envelope["id"],
            "thread_id": envelope.get("thread_id"),
        })
