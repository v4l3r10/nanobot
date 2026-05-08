"""HTTP + WebSocket entry point for the inter-agent mailbox router.

The service exposes three surfaces:

* ``GET /healthz``       — liveness + basic stats (no auth).
* ``POST /files/``       — multipart upload of an attachment blob (bearer auth).
* ``GET /files/<id>``    — streaming download of an attachment blob (bearer auth).
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
from .auth import TokenRegistry
from .db import Database
from .files import make_handlers
from .logging import configure_logging
from .peer_router import PeerHub, make_peer_endpoint

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("MAILBOX_DATA_DIR", "/data"))
START_TIME = time.monotonic()


def build_app() -> Starlette:
    configure_logging()

    db = Database(DATA_DIR / "mailbox.db")
    registry = TokenRegistry.from_env()
    peer_hub = PeerHub(db, registry)

    upload_handler, download_handler = make_handlers(db, registry, DATA_DIR)

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
        log.info("mailbox started", extra={"version": __version__})
        try:
            yield
        finally:
            await db.close()
            log.info("mailbox stopped")

    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/files/", upload_handler, methods=["POST"]),
        Route("/files/{attachment_id:int}", download_handler, methods=["GET"]),
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
