"""HTTP + WebSocket entry point for the inter-agent mailbox router.

The service exposes four surfaces:

* ``GET /healthz``       — liveness + basic stats (no auth).
* ``POST /files/``       — multipart upload of an attachment blob (bearer auth).
* ``GET /files/<id>``    — streaming download of an attachment blob (bearer auth).
* ``GET /messages``      — recent thread between caller and a named peer (bearer auth).
* ``WS  /peer``          — long-lived peer-to-peer messaging hub (bearer auth).

The peer hub is the only messaging plane: each gateway opens one WebSocket
and stays connected. Frames are forwarded between connected peers in real
time and persisted in SQLite (WAL) for offline-buffered delivery.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute

from . import __version__
from .auth import TokenRegistry, extract_bearer
from .db import Database
from .files import make_handlers
from .logging import configure_logging
from .peer_router import PeerHub, make_peer_endpoint
from .protocol import is_valid_agent_id
from .telegram_observer import build_observer_from_env

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("MAILBOX_DATA_DIR", "/data"))
START_TIME = time.monotonic()


def build_app() -> Starlette:
    configure_logging()

    db = Database(DATA_DIR / "mailbox.db")
    registry = TokenRegistry.from_env()
    peer_hub = PeerHub(db, registry)

    upload_handler, download_handler = make_handlers(db, registry, DATA_DIR)

    MAX_MESSAGES_PER_REQUEST = 100

    async def messages_handler(request: Request) -> JSONResponse:
        """Return recent messages between the bearer's owner and a named peer.

        Powers the ``peer_thread_show`` tool: an agent that wants to recap
        what it recently said to / received from a peer authenticates with
        its bearer (server identifies agent_id from the token) and asks for
        the thread with another agent.
        """
        bearer = extract_bearer(request.headers.get("authorization"))
        ctx = registry.resolve(bearer)
        if ctx is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        peer = (request.query_params.get("peer") or "").strip().lower()
        if not is_valid_agent_id(peer):
            return JSONResponse(
                {"error": f"invalid 'peer' param: {peer!r}"}, status_code=400
            )
        if peer == ctx.agent_id:
            return JSONResponse(
                {"error": "peer must differ from caller"}, status_code=400
            )
        try:
            last_n = int(request.query_params.get("last_n", 20))
        except ValueError:
            return JSONResponse(
                {"error": "'last_n' must be an integer"}, status_code=400
            )
        last_n = max(1, min(last_n, MAX_MESSAGES_PER_REQUEST))
        rows = await db.fetch_thread_between(
            ctx.agent_id, peer, last_n=last_n,
        )
        return JSONResponse({
            "agent": ctx.agent_id,
            "peer": peer,
            "count": len(rows),
            "messages": [
                {
                    "id": r["id"],
                    "thread_id": r["thread_id"],
                    "in_reply_to": r["in_reply_to"],
                    "from": r["from_agent"],
                    "to": r["to_agent"],
                    "text": r["body"],
                    "ts": r["created_at"],
                    "delivered": r["delivered_at"] is not None,
                    "closing": bool(r["closing"]),
                }
                for r in rows
            ],
        })

    async def healthz(_request: Request) -> JSONResponse:
        online = await peer_hub.online_peers()
        return JSONResponse({
            "status": "ok",
            "version": __version__,
            "uptime_s": int(time.monotonic() - START_TIME),
            "msg_count": await db.count_messages(),
            "peers_online": online,
        })

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        await db.open()
        # Optional Telegram observer. Built from env so the service runs
        # unchanged when the bot token is absent (the default in tests).
        observer = build_observer_from_env(
            db=db,
            hub_status_provider=peer_hub.online_peers,
            forward_callable=peer_hub.forward,
        )
        if observer is not None:
            peer_hub.set_observer(observer)
            try:
                await observer.start()
            except Exception:  # noqa: BLE001
                # Telegram failures must not prevent the mailbox from serving
                # peer traffic. Detach and continue without the observer.
                log.exception("telegram observer failed to start; continuing without it")
                peer_hub.set_observer(None)
                observer = None
        log.info("mailbox started", extra={"version": __version__})
        try:
            yield
        finally:
            if observer is not None:
                await observer.stop()
            await db.close()
            log.info("mailbox stopped")

    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/files/", upload_handler, methods=["POST"]),
        Route("/files/{attachment_id:int}", download_handler, methods=["GET"]),
        Route("/messages", messages_handler, methods=["GET"]),
        WebSocketRoute("/peer", make_peer_endpoint(peer_hub, registry)),
    ]

    return Starlette(routes=routes, lifespan=lifespan)


app = build_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "nanobot_mailbox.server:app",
        host="0.0.0.0",
        port=8765,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
