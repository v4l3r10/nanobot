"""Tests for the /files upload+download routes (attachments).

Focus: the MIME normalization policy. Unknown MIMEs must be accepted and
recorded as application/octet-stream rather than rejected with HTTP 415 —
the previous strict allowlist caused legitimate inter-agent transfers
(e.g. .tar.gz archives) to fail.
"""

from __future__ import annotations

import io

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from nanobot_mailbox.auth import TokenRegistry
from nanobot_mailbox.db import Database
from nanobot_mailbox.files import OCTET_STREAM, make_handlers


@pytest.fixture
async def client(tmp_path):
    """Build an isolated Starlette app with the /files routes only."""
    db = Database(tmp_path / "mailbox.db")
    await db.open()
    registry = TokenRegistry({"tok-bronzo": "bronzo"})
    upload, download = make_handlers(db, registry, tmp_path)
    app = Starlette(routes=[
        Route("/files/", upload, methods=["POST"]),
        Route("/files/{attachment_id:int}", download, methods=["GET"]),
    ])
    yield TestClient(app), db
    await db.close()


@pytest.mark.asyncio
async def test_upload_accepts_unknown_mime_as_octet_stream(client):
    """A MIME outside the allowlist must NOT be rejected; the server
    normalizes it to application/octet-stream and stores the blob."""
    tc, db = client
    resp = tc.post(
        "/files/",
        headers={"authorization": "Bearer tok-bronzo"},
        files={"file": ("archive.tar.gz", io.BytesIO(b"hello"), "application/x-some-weird-mime")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mime"] == OCTET_STREAM
    row = await db.fetch_attachment(body["file_id"])
    assert row["mime"] == OCTET_STREAM
    assert row["filename"] == "archive.tar.gz"


@pytest.mark.asyncio
async def test_upload_accepts_gzip_archive(client):
    """application/gzip is now in the allowlist (was missing, blocking
    .tar.gz transfers between bots)."""
    tc, _ = client
    resp = tc.post(
        "/files/",
        headers={"authorization": "Bearer tok-bronzo"},
        files={"file": ("dump.tar.gz", io.BytesIO(b"\x1f\x8b\x08gzipheader"), "application/gzip")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["mime"] == "application/gzip"


@pytest.mark.asyncio
async def test_upload_accepts_zip(client):
    tc, _ = client
    resp = tc.post(
        "/files/",
        headers={"authorization": "Bearer tok-bronzo"},
        files={"file": ("bundle.zip", io.BytesIO(b"PK\x03\x04"), "application/zip")},
    )
    assert resp.status_code == 200
    assert resp.json()["mime"] == "application/zip"


@pytest.mark.asyncio
async def test_upload_preserves_known_mime(client):
    """Sanity: a MIME inside the allowlist is recorded verbatim, not rewritten."""
    tc, _ = client
    resp = tc.post(
        "/files/",
        headers={"authorization": "Bearer tok-bronzo"},
        files={"file": ("note.txt", io.BytesIO(b"hi"), "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["mime"] == "text/plain"


@pytest.mark.asyncio
async def test_upload_requires_auth(client):
    tc, _ = client
    resp = tc.post(
        "/files/",
        files={"file": ("a.txt", io.BytesIO(b"hi"), "text/plain")},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_missing_file_part(client):
    tc, _ = client
    resp = tc.post(
        "/files/",
        headers={"authorization": "Bearer tok-bronzo"},
        data={"notafile": "x"},
    )
    assert resp.status_code == 400
