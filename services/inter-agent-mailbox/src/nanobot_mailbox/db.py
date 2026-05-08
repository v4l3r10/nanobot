"""SQLite (WAL) data access layer."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from .protocol import (
    AgentId,
    AttachmentMeta,
    Envelope,
    MessageType,
    Priority,
    new_message_id,
    new_thread_id,
    utcnow_iso,
)

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class Database:
    """Single async connection. WAL mode serializes writes; reads are concurrent."""

    def __init__(self, path: Path):
        self.path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA busy_timeout = 5000")
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._apply_migrations()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _apply_migrations(self) -> None:
        assert self._conn is not None
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        await self._conn.commit()
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = sql_file.stem.split("_", 1)[0]
            cursor = await self._conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is not None:
                continue
            log.info("applying migration", extra={"migration": sql_file.name})
            await self._conn.executescript(sql_file.read_text(encoding="utf-8"))
            await self._conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, utcnow_iso()),
            )
            await self._conn.commit()

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "database not opened"
        return self._conn

    # --- messages -----------------------------------------------------------

    async def insert_message(
        self,
        *,
        from_agent: AgentId,
        to_agent: AgentId,
        type: MessageType,
        subject: str,
        body: str,
        priority: Priority = "normal",
        in_reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> tuple[str, str]:
        msg_id = new_message_id()
        thread = thread_id
        if thread is None and in_reply_to is not None:
            thread = await self._thread_of(in_reply_to)
        if thread is None:
            thread = new_thread_id()
        async with self._lock:
            await self.conn.execute(
                "INSERT INTO messages (id, thread_id, in_reply_to, from_agent, to_agent,"
                " type, subject, body, priority) VALUES (?,?,?,?,?,?,?,?,?)",
                (msg_id, thread, in_reply_to, from_agent, to_agent, type, subject, body, priority),
            )
            await self.conn.commit()
        return msg_id, thread

    async def _thread_of(self, msg_id: str) -> str | None:
        cursor = await self.conn.execute(
            "SELECT thread_id FROM messages WHERE id = ?", (msg_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row["thread_id"] if row else None

    async def fetch_inbox(
        self,
        agent: AgentId,
        *,
        unread_only: bool = True,
        limit: int = 20,
        type_filter: MessageType | None = None,
        since: str | None = None,
    ) -> list[Envelope]:
        async with self._lock:
            now = utcnow_iso()
            params: list[Any] = [agent]
            sql = (
                "SELECT id, thread_id, in_reply_to, from_agent, to_agent,"
                " type, subject, body, priority, created_at, read_at"
                " FROM messages WHERE to_agent = ?"
            )
            if unread_only:
                sql += " AND read_at IS NULL"
            if type_filter:
                sql += " AND type = ?"
                params.append(type_filter)
            if since:
                sql += " AND created_at >= ?"
                params.append(since)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            cursor = await self.conn.execute(sql, params)
            rows = await cursor.fetchall()
            await cursor.close()

            ids = [r["id"] for r in rows]
            if ids and unread_only:
                placeholders = ",".join("?" * len(ids))
                await self.conn.execute(
                    f"UPDATE messages SET read_at = ? WHERE id IN ({placeholders})"
                    f" AND read_at IS NULL",
                    [now, *ids],
                )
                await self.conn.commit()

            envelopes: list[Envelope] = []
            for row in rows:
                attachments = await self._fetch_attachments_meta(row["id"])
                envelopes.append(Envelope(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    in_reply_to=row["in_reply_to"],
                    **{"from": row["from_agent"], "to": row["to_agent"]},
                    type=row["type"],
                    subject=row["subject"],
                    body=row["body"],
                    priority=row["priority"],
                    attachments=attachments,
                    created_at=row["created_at"],
                    read_at=now if unread_only else row["read_at"],
                ))
            return envelopes

    async def _fetch_attachments_meta(self, message_id: str) -> list[AttachmentMeta]:
        cursor = await self.conn.execute(
            "SELECT id, filename, mime, size_bytes FROM attachments WHERE message_id = ?",
            (message_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            AttachmentMeta(
                id=r["id"], name=r["filename"], mime=r["mime"], size_bytes=r["size_bytes"]
            )
            for r in rows
        ]

    async def count_messages(self) -> int:
        cursor = await self.conn.execute("SELECT COUNT(*) AS n FROM messages")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["n"])

    # --- attachments --------------------------------------------------------

    async def insert_attachment(
        self,
        *,
        owner_agent: str,
        filename: str,
        mime: str,
        size_bytes: int,
        storage: str,
        inline_b64: str | None = None,
        blob_path: str | None = None,
        message_id: str | None = None,
    ) -> int:
        async with self._lock:
            cursor = await self.conn.execute(
                "INSERT INTO attachments (message_id, owner_agent, filename, mime,"
                " size_bytes, storage, inline_b64, blob_path) VALUES (?,?,?,?,?,?,?,?)",
                (message_id, owner_agent, filename, mime, size_bytes, storage,
                 inline_b64, blob_path),
            )
            attachment_id = cursor.lastrowid
            await cursor.close()
            await self.conn.commit()
            assert attachment_id is not None
            return attachment_id

    async def update_attachment_blob(
        self, *, attachment_id: int, size_bytes: int, blob_path: str
    ) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE attachments SET size_bytes = ?, blob_path = ? WHERE id = ?",
                (size_bytes, blob_path, attachment_id),
            )
            await self.conn.commit()

    async def attach_to_message(
        self, *, attachment_ids: Iterable[int], message_id: str, owner_agent: str
    ) -> int:
        ids = list(attachment_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        async with self._lock:
            cursor = await self.conn.execute(
                f"UPDATE attachments SET message_id = ?"
                f" WHERE id IN ({placeholders}) AND message_id IS NULL"
                f" AND owner_agent = ?",
                [message_id, *ids, owner_agent],
            )
            updated = cursor.rowcount
            await cursor.close()
            await self.conn.commit()
            return updated

    async def verify_attachments_available(
        self, *, attachment_ids: Iterable[int], owner_agent: str
    ) -> bool:
        """True iff every id in *attachment_ids* exists, is owned by
        *owner_agent*, and has not yet been linked to a message.

        Used to fail-fast in :meth:`forward` before persisting the message,
        avoiding orphan rows when one or more attachment_ids are invalid.
        """
        ids = list(attachment_ids)
        if not ids:
            return True
        placeholders = ",".join("?" * len(ids))
        cursor = await self.conn.execute(
            f"SELECT COUNT(*) AS n FROM attachments"
            f" WHERE id IN ({placeholders})"
            f" AND owner_agent = ?"
            f" AND message_id IS NULL",
            [*ids, owner_agent],
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["n"]) == len(ids)

    async def fetch_attachment(self, attachment_id: int) -> aiosqlite.Row | None:
        cursor = await self.conn.execute(
            "SELECT * FROM attachments WHERE id = ?", (attachment_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row

    async def fetch_message_attachments_meta(
        self, message_id: str
    ) -> list[aiosqlite.Row]:
        """Return attachment metadata rows linked to *message_id* (no body)."""
        cursor = await self.conn.execute(
            "SELECT id, filename, mime, size_bytes, storage"
            " FROM attachments WHERE message_id = ?",
            (message_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def fetch_message_recipient(self, message_id: str) -> str | None:
        cursor = await self.conn.execute(
            "SELECT to_agent FROM messages WHERE id = ?", (message_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row["to_agent"] if row else None

    # --- peer delivery (real-time WS plane) --------------------------------

    async def fetch_pending_delivery(
        self, agent: AgentId, *, limit: int = 100
    ) -> list[aiosqlite.Row]:
        """Return messages addressed to *agent* not yet delivered over WS.

        Ordered chronologically so a peer reconnecting sees its accumulated
        traffic in the order it was sent.
        """
        async with self._lock:
            cursor = await self.conn.execute(
                "SELECT id, thread_id, in_reply_to, from_agent, to_agent,"
                " body, created_at"
                " FROM messages"
                " WHERE to_agent = ? AND delivered_at IS NULL"
                " ORDER BY created_at ASC LIMIT ?",
                (agent, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return list(rows)

    async def mark_delivered(self, message_ids: Iterable[str]) -> int:
        ids = list(message_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        now = utcnow_iso()
        async with self._lock:
            cursor = await self.conn.execute(
                f"UPDATE messages SET delivered_at = ?"
                f" WHERE id IN ({placeholders}) AND delivered_at IS NULL",
                [now, *ids],
            )
            updated = cursor.rowcount
            await cursor.close()
            await self.conn.commit()
            return updated
