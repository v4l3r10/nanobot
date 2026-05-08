"""Streaming blob upload/download routes for attachments larger than 256 KB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiofiles
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .auth import AuthContext, TokenRegistry, extract_bearer
from .db import Database

log = logging.getLogger(__name__)

INLINE_THRESHOLD = 256 * 1024
MAX_FILE_BYTES = 50 * 1024 * 1024
ALLOWED_MIMES = frozenset({
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/json",
    "image/png", "image/jpeg", "image/webp", "image/gif",
    "text/plain", "text/markdown", "text/x-python", "text/csv",
    "audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg",
    "video/mp4", "video/webm", "video/quicktime",
    "application/octet-stream",
})


def _blob_path_for(data_dir: Path, attachment_id: int) -> Path:
    today = datetime.now(timezone.utc)
    sub = data_dir / "blobs" / f"{today.year:04d}" / f"{today.month:02d}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{attachment_id:010d}.bin"


def make_handlers(db: Database, registry: TokenRegistry, data_dir: Path):
    """Build Starlette route handlers closed over the shared dependencies."""

    def _auth(request: Request) -> AuthContext | None:
        bearer = extract_bearer(request.headers.get("authorization"))
        return registry.resolve(bearer)

    async def upload_file(request: Request) -> Response:
        ctx = _auth(request)
        if ctx is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        form = await request.form(max_files=1, max_fields=2)
        upload = form.get("file")
        if upload is None or not getattr(upload, "filename", None):
            return JSONResponse({"error": "missing file part"}, status_code=400)
        mime = (getattr(upload, "content_type", None) or "application/octet-stream").lower()
        if mime not in ALLOWED_MIMES:
            return JSONResponse(
                {"error": f"mime not allowed: {mime}"}, status_code=415
            )

        attachment_id = await db.insert_attachment(
            owner_agent=ctx.agent_id,
            filename=upload.filename,
            mime=mime,
            size_bytes=0,
            storage="blob",
            blob_path="",
        )
        path = _blob_path_for(data_dir, attachment_id)
        size = 0
        try:
            async with aiofiles.open(path, "wb") as f:
                while True:
                    chunk = await upload.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        await f.flush()
                        break
                    await f.write(chunk)
            if size > MAX_FILE_BYTES:
                path.unlink(missing_ok=True)
                return JSONResponse(
                    {"error": f"file too large (>{MAX_FILE_BYTES} bytes)"},
                    status_code=413,
                )
        except Exception as exc:
            log.exception("upload write failed")
            path.unlink(missing_ok=True)
            return JSONResponse(
                {"error": "write failed", "detail": str(exc)}, status_code=500
            )

        await db.update_attachment_blob(
            attachment_id=attachment_id, size_bytes=size, blob_path=str(path)
        )
        log.info(
            "upload ok",
            extra={
                "agent": ctx.agent_id, "attachment_id": attachment_id,
                "size": size, "mime": mime,
            },
        )
        return JSONResponse(
            {"file_id": attachment_id, "size_bytes": size, "mime": mime}
        )

    async def download_file(request: Request) -> Response:
        ctx = _auth(request)
        if ctx is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        attachment_id = int(request.path_params["attachment_id"])
        row = await db.fetch_attachment(attachment_id)
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if row["owner_agent"] != ctx.agent_id:
            recipient = (
                await db.fetch_message_recipient(row["message_id"])
                if row["message_id"] else None
            )
            if recipient != ctx.agent_id:
                return JSONResponse({"error": "forbidden"}, status_code=403)
        if row["storage"] != "blob" or not row["blob_path"]:
            return JSONResponse({"error": "not a blob"}, status_code=400)
        path = Path(row["blob_path"])
        if not path.is_file():
            return JSONResponse({"error": "blob missing"}, status_code=410)

        async def stream() -> AsyncIterator[bytes]:
            async with aiofiles.open(path, "rb") as f:
                while True:
                    chunk = await f.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            stream(),
            media_type=row["mime"],
            headers={
                "Content-Disposition": f'attachment; filename="{row["filename"]}"',
                "Content-Length": str(row["size_bytes"]),
            },
        )

    return upload_file, download_file
