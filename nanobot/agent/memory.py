"""Memory system: pure file I/O store, lightweight Consolidator, and Dream processor."""

from __future__ import annotations

import asyncio
import json
import os
import re
import weakref
from contextlib import suppress
from datetime import date as _date
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import tiktoken
from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.wiki.attachments_reconciler import run_attachments_reconcile
from nanobot.agent.wiki.ingest import run_ingest
from nanobot.agent.wiki.lint import run_lint
from nanobot.agent.wiki.paths import vault_slug
from nanobot.agent.wiki.vault import Vault
from nanobot.session.manager import Session
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.gitstore import GitStore
from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    strip_think,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.vault_lock import get_vault_lock

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


# Process-wide guard serializing every ``Dream.run()`` invocation (Task 4.6).
# The cron tick (`await agent.dream.run()`) and `/dream`'s unguarded
# `asyncio.create_task(_run_dream())` share one event loop and can otherwise
# run two `Dream.run()`s concurrently: both pass the cursor guard, process the
# SAME batch, and double-edit MEMORY.md (and, with the wiki on, double-Ingest).
# `run()` acquires this as its OUTERMOST lock so a waiting run re-reads the
# cursor the prior run advanced and correctly no-ops.
#
# Deliberately a plain MODULE-GLOBAL `asyncio.Lock` held by a STRONG module
# reference for the process lifetime — NOT `utils.vault_lock.get_vault_lock`,
# whose `WeakValueDictionary` can GC + recreate the lock between two
# non-overlapping `create_task`s, defeating mutual exclusion. On Python 3.11+
# a module-scope `asyncio.Lock()` has no loop bound at construction (it binds
# to the running loop lazily), so this is safe to define at import time.
#
# Lock ordering: dream-run-lock (this, outermost) -> per-vault lock
# (`get_vault_lock(slug)`, acquired inside the wiki block of `run()`). The
# `wiki_note` tool takes only the per-vault lock and NEVER this lock, so there
# is no lock-ordering inversion and no deadlock cycle.
#
# Cross-loop footgun (test isolation only, NOT a prod concern): the FIRST time
# this module-global lock is *contended* it permanently binds to that event
# loop; contending it again from a DIFFERENT loop in the same process raises
# `RuntimeError: <Lock> is bound to a different event loop`. This is only
# reachable under pytest's per-test event loops (`asyncio_mode=auto`); the
# production gateway is a single `asyncio.run` per process so import-time
# construction is always safe. The test suite handles this with an autouse
# fixture that resets the module global between tests (see
# tests/agent/test_dream_wiki.py).
#
# M1: the lock is intentionally held across the provider LLM calls (Phase 1/2
# plus the wiki Ingest) so Dream cycles can never overlap; this is bounded by
# the provider SDK default request timeout, and an explicit outer Dream timeout
# / provider client-timeout is a documented production prerequisite before
# enabling `wiki_enabled=true` (tracked in the plan).
_DREAM_RUN_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# MemoryStore — pure file I/O layer
# ---------------------------------------------------------------------------

class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history.jsonl, SOUL.md, USER.md."""

    _DEFAULT_MAX_HISTORY = 1000
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )
    _JOURNAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.journal_dir = ensure_dir(self.memory_dir / "journal")
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._corruption_logged = False  # rate-limit non-int cursor warning
        self._oversize_logged = False  # rate-limit oversized-entry warning
        # Static tracked base. GitStore dynamically also versions every file
        # under ``memory/users/**`` (the per-user wiki vaults; Task 5.1) on
        # top of this base at commit/revert time, so /dream-restore can roll
        # the wiki back. When no vault exists this expands to exactly these
        # four files — a stock workspace commits byte-identically to before.
        self._git = GitStore(
            workspace,
            tracked_files=[
                "SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor",
            ],
            tracked_dirs=["memory/journal"],
        )
        self._maybe_migrate_legacy_history()

    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers -----------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl.

        The migration is best-effort and prioritizes preserving as much content
        as possible over perfect parsing.
        """
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                # Default to "already processed" so upgrades do not replay the
                # user's entire historical archive into Dream on first start.
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- MEMORY.md (long-term facts) -----------------------------------------

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        atomic_write_text(self.memory_file, content)

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        atomic_write_text(self.soul_file, content)

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        atomic_write_text(self.user_file, content)

    # -- journal (per-day episodic notes) ------------------------------------

    def journal_path(self, date: str) -> Path:
        return self.journal_dir / f"{date}.md"

    def read_journal(self, date: str) -> str:
        return self.read_file(self.journal_path(date))

    def write_journal(self, date: str, content: str) -> None:
        self.journal_path(date).write_text(content, encoding="utf-8")

    def journal_exists(self, date: str) -> bool:
        return self.journal_path(date).exists()

    def list_recent_journal_notes(self, n: int) -> list[tuple[str, str]]:
        """Return up to *n* most recent daily notes as ``(date, content)``.

        Newest first, sorted by ISO date in the filename. Files whose stem
        does not match ``YYYY-MM-DD`` are skipped so a stray markdown file
        in the journal directory cannot break Dream's prompt assembly.
        """
        if n <= 0:
            return []
        candidates = [
            entry.stem
            for entry in self.journal_dir.iterdir()
            if entry.is_file()
            and entry.suffix == ".md"
            and self._JOURNAL_DATE_RE.match(entry.stem)
        ]
        candidates.sort(reverse=True)
        return [(date, self.read_journal(date)) for date in candidates[:n]]

    # -- context injection (used by context.py) ------------------------------

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- history.jsonl — append-only, JSONL format ---------------------------

    def append_history(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> int:
        """Append *entry* to history.jsonl and return its auto-incrementing cursor.

        Entries are passed through `strip_think` to drop template-level leaks
        (e.g. unclosed `<think` prefixes, `<channel|>` markers) before being
        persisted. If the cleaned content is empty but the raw entry wasn't,
        the record is persisted with an empty string rather than falling back
        to the raw leak — otherwise `strip_think`'s guarantees would be
        undone by history replay / consolidation downstream.

        A defensive cap (*max_chars*, default ``_HISTORY_ENTRY_HARD_CAP``) is
        applied as a final safety net: individual callers should cap their own
        content more tightly; this default only exists to catch unintentional
        large writes (e.g. an LLM echoing its input back as a "summary").

        Task 7.2: *session_key* tags the record with the consolidated
        session's EFFECTIVE key so Dream can route each user's entries into
        THAT user's per-user wiki vault (``vault_slug(session_key)``).
        ``None`` (the default, used when the caller genuinely has no session
        in scope) is persisted as the back-compat unified key
        ``"unified:default"``, which ``vault_slug`` maps to the single
        ``unified_default`` vault — identical to pre-7.2 behavior. LEGACY
        records on disk that physically LACK this field are still valid
        everywhere: ALL readers MUST use
        ``entry.get("session_key", "unified:default")`` (never indexing), so
        an untagged legacy record also routes to the unified vault. The field
        is INERT when the wiki is off (the Consolidator/context path keys off
        content/timestamp/cursor only) — wiki-off behavior is unchanged.
        """
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw = entry.rstrip()
        if len(raw) > limit:
            if not self._oversize_logged:
                self._oversize_logged = True
                logger.warning(
                    "history entry exceeds {} chars ({}); truncating. "
                    "Usually means a caller forgot its own cap; "
                    "further occurrences suppressed.",
                    limit, len(raw),
                )
            raw = truncate_text(raw, limit)
        content = strip_think(raw)
        if raw and not content:
            logger.debug(
                "history entry {} stripped to empty (likely template leak); "
                "persisting empty content to avoid re-polluting context",
                cursor,
            )
        record = {
            "cursor": cursor,
            "timestamp": ts,
            "content": content,
            "session_key": session_key or "unified:default",
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        """Int cursors only — reject bool (``isinstance(True, int)`` is True)."""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        """Yield ``(entry, cursor)`` for entries with int cursors; warn once on corruption."""
        poisoned: Any = None
        for entry in self._read_entries():
            raw = entry.get("cursor")
            if raw is None:
                continue
            cursor = self._valid_cursor(raw)
            if cursor is None:
                poisoned = raw
                continue
            yield entry, cursor
        if poisoned is not None and not self._corruption_logged:
            self._corruption_logged = True
            logger.warning(
                "history.jsonl contains a non-int cursor ({!r}); dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                poisoned,
            )

    def _next_cursor(self) -> int:
        """Read the current cursor counter and return the next value."""
        if self._cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
        # Fast path: trust the tail when intact.  Otherwise scan the whole
        # file and take ``max`` — that stays correct even if the monotonic
        # invariant was broken by external writes.
        last = self._read_last_entry() or {}
        cursor = self._valid_cursor(last.get("cursor"))
        if cursor is not None:
            return cursor + 1
        return max((c for _, c in self._iter_valid_entries()), default=0) + 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with a valid cursor > *since_cursor*."""
        return [e for e, c in self._iter_valid_entries() if c > since_cursor]

    def compact_history(self) -> None:
        """Drop oldest entries if the file exceeds *max_history_entries*."""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        kept = entries[-self.max_history_entries:]
        self._write_entries(kept)

    # -- JSONL helpers -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history.jsonl."""
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """Read the last entry from the JSONL file efficiently."""
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [line for line in data.split("\n") if line.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history.jsonl with the given entries (atomic write)."""
        tmp_path = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.history_file)

            # fsync the directory so the rename is durable.
            # On Windows, opening a directory with O_RDONLY raises
            # PermissionError — skip the dir sync there (NTFS
            # journals metadata synchronously).
            with suppress(PermissionError):
                fd = os.open(str(self.history_file.parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    # -- dream cursor --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(
        self,
        messages: list[dict],
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> None:
        """Fallback: dump raw messages to history.jsonl without LLM summarization.

        Task 7.2: *session_key* is threaded into the appended record so the
        raw breadcrumb routes to the SAME per-user vault the consolidated
        session would. ``None`` → unified (safe back-compat).
        """
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        formatted = truncate_text(self._format_messages(messages), limit)
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{formatted}",
            session_key=session_key,
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )



# ---------------------------------------------------------------------------
# Consolidator — lightweight token-budget triggered consolidation
# ---------------------------------------------------------------------------


# Individual history.jsonl writers cap their own payloads tightly; the
# _HISTORY_ENTRY_HARD_CAP at append_history() is a belt-and-suspenders default
# that catches any new caller that forgot to set its own cap.
_RAW_ARCHIVE_MAX_CHARS = 16_000       # fallback dump (LLM failed)
_ARCHIVE_SUMMARY_MAX_CHARS = 8_000    # LLM-produced consolidation summary
_HISTORY_ENTRY_HARD_CAP = 64_000      # emergency cap in append_history


class Consolidator:
    """Lightweight consolidation: summarizes evicted messages into history.jsonl."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        consolidation_ratio: float = 0.5,
        memory_key_resolver: Callable[["Session"], str] | None = None,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.consolidation_ratio = consolidation_ratio
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        # CV2: optional resolver mapping Session → vault/memory key.
        # When None (pre-CV2), every callsite that consolidates uses
        # ``session.key`` verbatim — back-compat. When provided (typically
        # by AgentLoop), the resolver collapses to UNIFIED_SESSION_KEY
        # under ``unified_memory=true`` so per-user chat sessions can share
        # a single memory vault.
        self._memory_key_resolver = memory_key_resolver
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def _resolve_memory_key(self, session: "Session") -> str:
        """CV2: return the memory/vault key for *session*, falling back to
        ``session.key`` when no resolver is configured (pre-CV2 behaviour)."""
        if self._memory_key_resolver is not None:
            return self._memory_key_resolver(session)
        return session.key

    def set_provider(
        self,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = provider.generation.max_tokens

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    @staticmethod
    def _full_unconsolidated_history(
        session: Session,
        *,
        include_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the whole unconsolidated tail for consolidation decisions."""
        unconsolidated_count = len(session.messages) - session.last_consolidated
        if unconsolidated_count <= 0:
            return []
        return session.get_history(
            max_messages=unconsolidated_count,
            include_timestamps=include_timestamps,
        )

    @staticmethod
    def _replay_overflow_boundary(
        session: Session,
        replay_max_messages: int | None,
    ) -> int | None:
        if not replay_max_messages or replay_max_messages <= 0:
            return None
        tail = list(enumerate(session.messages[session.last_consolidated:], session.last_consolidated))
        if len(tail) <= replay_max_messages:
            return None

        sliced = tail[-replay_max_messages:]
        for i, (_idx, message) in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1][1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        legal_start = find_legal_message_start([message for _idx, message in sliced])
        if legal_start:
            sliced = sliced[legal_start:]
        if not sliced:
            return len(session.messages)

        first_visible_idx = sliced[0][0]
        if first_visible_idx <= session.last_consolidated:
            return None
        return first_visible_idx

    async def _consolidate_replay_overflow(
        self,
        session: Session,
        replay_max_messages: int | None,
    ) -> str | None:
        """Archive messages that would be hidden by the replay message window."""
        end_idx = self._replay_overflow_boundary(session, replay_max_messages)
        if end_idx is None:
            return None
        chunk = session.messages[session.last_consolidated:end_idx]
        if not chunk:
            return None
        logger.info(
            "Replay-window consolidation for {}: chunk={} msgs, replay_max={}",
            session.key,
            len(chunk),
            replay_max_messages,
        )
        # Task 7.2: thread the EFFECTIVE session key (session.key — unified
        # or channel:chat_id) so Dream routes this user's consolidated
        # memory into THAT user's per-user vault.
        # CV2: under unified_memory=true the resolver collapses to
        # UNIFIED_SESSION_KEY so the shared vault receives consolidation.
        summary = await self.archive(chunk, session_key=self._resolve_memory_key(session))
        session.last_consolidated = end_idx
        self.sessions.save(session)
        return summary

    def _persist_last_summary(self, session: Session, summary: str | None) -> None:
        if summary and summary != "(nothing)":
            session.metadata["_last_summary"] = {
                "text": summary,
                "last_active": session.updated_at.isoformat(),
            }
            self.sessions.save(session)

    def estimate_session_prompt_tokens(
        self,
        session: Session,
    ) -> tuple[int, str]:
        """Estimate prompt size from the full unconsolidated session tail."""
        history = self._full_unconsolidated_history(session, include_timestamps=True)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        # Include archived summary in estimation so the budget accounts for it.
        meta = session.metadata.get("_last_summary")
        summary = meta.get("text") if isinstance(meta, dict) else (meta if isinstance(meta, str) else None)
        # I1: pass the effective session key so the probe builds the SAME
        # prompt the real turn will. The real turn path (AgentLoop.
        # _build_initial_messages) resolves `effective_key = session.key or
        # _effective_session_key(msg)`; since SessionManager.get_or_create
        # always stores a non-empty key, the `or` fallback is never taken in
        # practice and `session.key` IS that effective key. When wiki is off
        # ContextBuilder ignores session_key entirely, so the probe still
        # builds the byte-identical wiki-off prompt (regression-safe).
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            sender_id=None,
            session_summary=summary,
            session_metadata=session.metadata,
            session_key=session.key,
            memory_key=self._resolve_memory_key(session),  # CV2
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    @property
    def _input_token_budget(self) -> int:
        """Available input token budget for consolidation LLM."""
        return self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER

    def _truncate_to_token_budget(self, text: str) -> str:
        """Truncate text so it fits within the consolidation LLM's token budget."""
        budget = self._input_token_budget
        if budget <= 0:
            return truncate_text(text, _RAW_ARCHIVE_MAX_CHARS)
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= budget:
                return text
            return enc.decode(tokens[:budget]) + "\n... (truncated)"
        except Exception:
            return truncate_text(text, budget * 4)

    async def archive(
        self, messages: list[dict], *, session_key: str | None = None
    ) -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        Returns the summary text on success, None if nothing to archive.

        Task 7.2: *session_key* is the EFFECTIVE key of the session being
        consolidated (``session.key``: ``"unified:default"`` under
        ``unified_session``, else ``channel:chat_id``). It is threaded into
        the appended history record (both the LLM-summary path and the
        ``raw_archive`` degraded fallback) so Dream routes this user's
        consolidated memory into THAT user's per-user wiki vault. ``None``
        (a caller with no session in scope) → unified (safe back-compat);
        consolidation logic itself is UNCHANGED.
        """
        if not messages:
            return None
        try:
            formatted = MemoryStore._format_messages(messages)
            formatted = self._truncate_to_token_budget(formatted)
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/consolidator_archive.md",
                            strip=True,
                        ),
                    },
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"LLM returned error: {response.content}")
            summary = response.content or "[no summary]"
            self.store.append_history(
                summary,
                max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
                session_key=session_key,
            )
            return summary
        except Exception:
            logger.warning("Consolidation LLM call failed, raw-dumping to history")
            self.store.raw_archive(messages, session_key=session_key)
            return None

    async def maybe_consolidate_by_tokens(
        self,
        session: Session,
        *,
        replay_max_messages: int | None = None,
    ) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            # Refresh session reference: AutoCompact may have replaced it.
            fresh = self.sessions.get_or_create(session.key)
            if fresh is not session:
                session = fresh
            if not session.messages:
                return

            budget = self._input_token_budget
            target = int(budget * self.consolidation_ratio)
            last_summary = await self._consolidate_replay_overflow(
                session,
                replay_max_messages,
            )
            try:
                estimated, source = self.estimate_session_prompt_tokens(
                    session,
                )
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                self._persist_last_summary(session, last_summary)
                return
            if estimated < budget:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                self._persist_last_summary(session, last_summary)
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    break

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                end_idx = boundary[0]

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    break

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                # Task 7.2: thread the EFFECTIVE session key for per-user
                # vault routing (see _consolidate_replay_overflow).
                # CV2: resolver collapses to UNIFIED_SESSION_KEY under
                # unified_memory=true (shared vault).
                summary = await self.archive(chunk, session_key=self._resolve_memory_key(session))
                # Advance the cursor either way: on success the chunk was
                # summarized; on failure archive() already raw-archived it as
                # a breadcrumb. Re-archiving the same chunk on the next call
                # would just emit duplicate [RAW] entries.
                if summary:
                    last_summary = summary
                session.last_consolidated = end_idx
                self.sessions.save(session)
                if not summary:
                    # LLM is degraded — stop hammering it this call;
                    # the next invocation can retry a fresh chunk.
                    break

                try:
                    estimated, source = self.estimate_session_prompt_tokens(
                        session,
                    )
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                if estimated <= 0:
                    break

            # Persist the last summary to session metadata so it can be injected
            # into the runtime context on the next prepare_session() call, aligning
            # the summary injection strategy with AutoCompact._archive().
            self._persist_last_summary(session, last_summary)

    async def compact_idle_session(
        self,
        session_key: str,
        max_suffix: int = 8,
    ) -> str | None:
        """Hard-truncate an idle session under the consolidation lock.

        Used by AutoCompact so all session mutation goes through a single
        lock-protected path.  Returns the summary text on success, ``None``
        if the LLM failed (raw_archive fallback), or ``""`` if there was
        nothing to archive.
        """
        lock = self.get_lock(session_key)
        async with lock:
            self.sessions.invalidate(session_key)
            session = self.sessions.get_or_create(session_key)

            tail = list(session.messages[session.last_consolidated:])
            if not tail:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return ""

            probe = Session(
                key=session.key,
                messages=tail.copy(),
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata={},
                last_consolidated=0,
            )
            dropped, already_consolidated = probe.retain_recent_legal_suffix(max_suffix)
            kept = probe.messages
            archive_msgs = dropped[already_consolidated:]

            if not archive_msgs and not kept:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return ""

            last_active = session.updated_at
            summary: str | None = ""
            if archive_msgs:
                summary = await self.archive(archive_msgs)

            if summary and summary != "(nothing)":
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                }

            session.messages = kept
            session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)

            if archive_msgs:
                logger.info(
                    "Idle-session compact for {}: archived={}, kept={}, summary={}",
                    session_key,
                    len(archive_msgs),
                    len(kept),
                    bool(summary),
                )

            return summary


# ---------------------------------------------------------------------------
# Dream — heavyweight cron-scheduled memory consolidation
# ---------------------------------------------------------------------------


# Single source of truth for the staleness threshold used in _annotate_with_ages
# *and* in the Phase 1 prompt template (passed as `stale_threshold_days`).
# Keep code and prompt aligned — if you bump this, the LLM's instruction string
# updates automatically.
_STALE_THRESHOLD_DAYS = 14


class Dream:
    """Two-phase memory processor: analyze history.jsonl, then edit files via AgentRunner.

    Phase 1 produces an analysis summary (plain LLM call).
    Phase 2 delegates to AgentRunner with read_file / edit_file tools so the
    LLM can make targeted, incremental edits instead of replacing entire files.
    """

    # Caps on prompt-bound inputs so Dream's LLM calls never exceed the model's
    # context window just because a file (or a legacy large history entry) grew
    # unexpectedly. Each file still appears in full via read_file when the agent
    # needs it in Phase 2 — these caps only bound the Phase 1/2 prompt preview.
    # The class constants below are the historical defaults; runtime values are
    # held on the instance (`self.memory_file_max_chars` etc.) so they can be
    # overridden via DreamConfig in config.json. A value of 0 disables the cap
    # (truncate_text returns the full content unchanged).
    _MEMORY_FILE_MAX_CHARS = 32_000
    _SOUL_FILE_MAX_CHARS = 16_000
    _USER_FILE_MAX_CHARS = 16_000
    _HISTORY_ENTRY_PREVIEW_MAX_CHARS = 4_000

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        memory_file_max_chars: int | None = None,
        soul_file_max_chars: int | None = None,
        user_file_max_chars: int | None = None,
        history_entry_preview_max_chars: int | None = None,
        timezone: str | None = None,
        daily_notes_enabled: bool = True,
        daily_notes_context_days: int = 2,
        daily_notes_max_chars: int = 8_000,
        wiki_enabled: bool = False,
        wiki_embeddings: bool = False,
        wiki_embedding_model: str = "",
        lint_cadence_h: int | None = None,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        # IANA timezone (e.g. "Europe/Rome") used to derive Phase 1's
        # current_date and the YYYY-MM-DD path of the daily journal note.
        # None falls back to the system's local timezone so behavior matches
        # the historical datetime.now() default before this kwarg existed.
        self.timezone = timezone
        # Wiki-tree memory gate (Task 4.5). Real instance attributes (NOT
        # @property / __slots__) so cli/commands.py's pre-wiring
        # ``agent.dream.wiki_enabled = ...`` (Task 3.1) keeps working and the
        # golden test can set ``dream.wiki_enabled = False``. Default False:
        # with the gate off Dream is byte-identical to v0.2.0.
        self.wiki_enabled: bool = wiki_enabled
        self.wiki_embeddings: bool = wiki_embeddings
        self.wiki_embedding_model: str = wiki_embedding_model
        # NOTE: decoupled lint cadence (lint_cadence_h) is a future
        # refinement; Lint is idempotent so running it each Dream cycle is
        # safe. Stored here for the cli pre-wiring; not consulted in 4.5.
        self.lint_cadence_h: int | None = lint_cadence_h
        # Kill switch for the git-blame-based per-line age annotation in Phase 1.
        # Default True keeps the #3212 behavior; set False to feed MEMORY.md raw
        # (e.g. if a specific LLM reacts poorly to the `← Nd` suffix).
        self.annotate_line_ages = annotate_line_ages
        # Prompt-preview caps. None means "use the class default" — keeps
        # backward compatibility with callers that don't pass these kwargs.
        self.memory_file_max_chars = (
            memory_file_max_chars
            if memory_file_max_chars is not None
            else self._MEMORY_FILE_MAX_CHARS
        )
        self.soul_file_max_chars = (
            soul_file_max_chars
            if soul_file_max_chars is not None
            else self._SOUL_FILE_MAX_CHARS
        )
        self.user_file_max_chars = (
            user_file_max_chars
            if user_file_max_chars is not None
            else self._USER_FILE_MAX_CHARS
        )
        self.history_entry_preview_max_chars = (
            history_entry_preview_max_chars
            if history_entry_preview_max_chars is not None
            else self._HISTORY_ENTRY_PREVIEW_MAX_CHARS
        )
        # Daily notes — episodic per-day layer Dream produces under
        # memory/journal/. Phase 1 loads the most recent N notes (yesterday
        # + today by default) so it has temporal context without re-reading
        # the full unprocessed history every cycle.
        self.daily_notes_enabled = daily_notes_enabled
        self.daily_notes_context_days = daily_notes_context_days
        self.daily_notes_max_chars = daily_notes_max_chars
        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model
        self._runner.provider = provider

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build a minimal tool registry for the Dream agent."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.file_state import FileStates
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool

        tools = ToolRegistry()
        workspace = self.store.workspace
        # Allow reading builtin skills for reference during skill creation
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        # Dream gets its own FileStates so its caches stay isolated from the
        # main loop's sessions (issue #3571).
        file_states = FileStates()
        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_allowed_dirs=extra_read,
            file_states=file_states,
        ))
        tools.register(EditFileTool(workspace=workspace, allowed_dir=workspace, file_states=file_states))
        # write_file resolves relative paths from workspace root and is restricted
        # to skills/ (skill creation) and memory/journal/ (daily notes bootstrap —
        # the prompt steers the LLM to edit_file with old_text="", but some models
        # use write_file instead, and that's a legitimate use case).
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        journal_dir = workspace / "memory" / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        tools.register(WriteFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_allowed_dirs=[journal_dir],
            file_states=file_states,
        ))
        return tools

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
        import re as _re

        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        desc_re = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)
        entries: dict[str, str] = {}
        for base in (self.store.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                # Prefer workspace skills over builtin (same name)
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = desc_re.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # -- main entry ----------------------------------------------------------

    def _today(self) -> str:
        """Return today's date as ``YYYY-MM-DD`` in the configured timezone.

        Mirrors the pattern of utils.helpers.current_time_str: an IANA tz
        name resolves via ZoneInfo; None falls back to the system local
        timezone so behavior matches the historical datetime.now() default.
        Bad/unknown tz names degrade silently to local — Dream should never
        crash on a typo in agents.defaults.timezone.
        """
        from zoneinfo import ZoneInfo

        try:
            tz = ZoneInfo(self.timezone) if self.timezone else None
        except Exception:
            tz = None
        now = datetime.now(tz=tz) if tz else datetime.now().astimezone()
        return now.strftime("%Y-%m-%d")

    def _build_journal_section(self) -> str:
        """Render the recent journal notes as a Phase 1 prompt section.

        Returns an empty string when the feature is disabled, no notes
        exist, or the configured cap is 0 (no slot to show them in).
        The combined text is truncated as a whole to ``daily_notes_max_chars``
        so a single very long note cannot crowd out the rest of the prompt.
        """
        if not self.daily_notes_enabled:
            return ""
        notes = self.store.list_recent_journal_notes(self.daily_notes_context_days)
        if not notes:
            return ""
        rendered = "\n\n".join(
            f"### {date}.md\n{content}" for date, content in notes
        )
        capped = truncate_text(rendered, self.daily_notes_max_chars)
        return f"## Recent Journal Notes ({len(capped)} chars)\n{capped}"

    def _annotate_with_ages(self, content: str) -> str:
        """Append per-line age suffixes to MEMORY.md content.

        Each non-blank line whose age exceeds ``_STALE_THRESHOLD_DAYS`` gets a
        suffix like ``← 30d`` indicating days since last modification.
        Returns the original content unchanged if git is unavailable,
        annotate fails, or the line count doesn't match the age count
        (which can happen with an uncommitted working-tree edit — better to
        skip annotation than to tag the wrong line).
        SOUL.md and USER.md are never annotated.
        """
        file_path = "memory/MEMORY.md"
        try:
            ages = self.store.git.line_ages(file_path)
        except Exception:
            logger.debug("line_ages failed for {}", file_path)
            return content
        if not ages:
            return content

        had_trailing = content.endswith("\n")
        lines = content.splitlines()
        # If HEAD-blob line count disagrees with the working-tree content we
        # received, ages would be assigned to the wrong lines — skip entirely
        # and feed the LLM un-annotated content rather than misleading data.
        if len(lines) != len(ages):
            logger.debug(
                "line_ages length mismatch for {} (lines={}, ages={}); skipping annotation",
                file_path, len(lines), len(ages),
            )
            return content

        annotated: list[str] = []
        for line, age in zip(lines, ages):
            if not line.strip():
                annotated.append(line)
                continue
            if age.age_days > _STALE_THRESHOLD_DAYS:
                annotated.append(f"{line}  \u2190 {age.age_days}d")
            else:
                annotated.append(line)
        result = "\n".join(annotated)
        if had_trailing:
            result += "\n"
        return result

    @staticmethod
    def _entry_slug(entry: dict[str, Any]) -> str:
        """Vault slug for ONE history record — TOTAL, write-aligned, never raises.

        ``isinstance(sk, str) and sk`` routes ONLY a non-empty ``str``
        ``session_key`` per-user; EVERYTHING else collapses to the unified
        key, then applies ``vault_slug``. This is TOTAL over every value
        ``json.loads`` can produce from an untrusted ``history.jsonl``:

        * ABSENT ``session_key`` (legacy record physically lacking the field),
        * JSON ``null`` → Python ``None``,
        * empty string ``""``,
        * falsy non-str (``0``, ``[]``, ``{}``),
        * **truthy non-str** (``123``, ``1.5``, ``True``, ``["x"]``,
          ``{"a": 1}``) — reachable from the SAME external / legacy /
          hand-edited / malformed writers ``null`` is; ``... or
          "unified:default"`` does NOT collapse these (they are truthy),
          so the OLD code reached ``vault_slug(123)`` →
          ``int.replace`` → ``AttributeError``.

        All of the above → the unified slug. This MIRRORS
        ``append_history``'s write side (``session_key or "unified:default"``)
        and makes the read side defensively TOTAL: it can NEVER raise
        ``AttributeError`` on ``vault_slug(non-str)`` (which would skip the
        whole wiki pass for every user that cycle — the cursor has already
        advanced — see ``_vaults_for_batch``). Used by BOTH
        ``_vaults_for_batch`` (grouping) AND the ``Dream.run()`` per-slug
        slice (filtering) so grouping and slicing can never diverge.
        """
        sk = entry.get("session_key")
        return vault_slug(sk if isinstance(sk, str) and sk else "unified:default")

    def _vaults_for_batch(self, batch: list[dict[str, Any]]) -> list[str]:
        """Vault slugs the wiki Ingest+Lint pass should run for this batch.

        IMPLEMENTED in Task 7.2 (per-user routing): group ``batch`` entries
        by ``_entry_slug`` (``vault_slug`` of
        ``entry.get("session_key") or "unified:default"``) and return the
        SORTED distinct slug list (deterministic — no dict-order leakage; the
        Dream call site iterates this list and feeds each slug ONLY its own
        ``session_key`` slice of ``batch`` via the SAME ``_entry_slug``).
        ``vault_slug`` is applied (not hardcoded) so a slug stays in lockstep
        with the slug the ``wiki_note`` tool / per-vault lock use.

        Collapse behavior (back-compat, BY CONSTRUCTION) — TOTAL, NEVER
        raises (``_entry_slug`` is total over every value ``json.loads``
        can produce from an untrusted ``history.jsonl``):

        * ``unified_session=True`` → every entry's ``session_key`` is
          ``"unified:default"`` → one slug ``vault_slug("unified:default")``
          == ``"unified_default"`` → exactly the pre-7.2 single-vault path.
        * ANY non-(non-empty-``str``) ``session_key`` — ABSENT (legacy
          untagged record), JSON ``None`` (``null``), empty ``""``, falsy
          non-str (``0``/``[]``/``{}``), OR **truthy non-str**
          (``123``/``1.5``/``True``/``list``/``dict`` from an external /
          legacy / hand-edited / malformed writer) → ALL collapse to the
          unified slug (write-side aligned). ``vault_slug`` is NEVER called
          with a non-str → no ``AttributeError`` that would skip the entire
          wiki pass for every user this cycle (the grouping call is OUTSIDE
          the per-iteration ``try`` and the cursor has already advanced —
          residual follow-up to 7.2, same data-loss class as C1 via a
          different bad type).
        * Mixed real per-user keys (non-empty ``str``) → one slug per
          distinct user, sorted.

        NOTE(Task 7.2 — SEPARATE from the Ingest batch-slicing): the
        ``slug == unified`` gate on ``legacy_workspace`` at the
        ``Dream.run()`` wiki-block call site (added in the 7.1 review-fix)
        MUST be kept. ``migrate_legacy`` reads the SINGLE GLOBAL
        ``memory/MEMORY.md`` + root ``USER.md`` — running it for a per-user
        slug fans that global blob (incl. another user's ``USER.md``
        profile) into every vault, a PERMANENT cross-user contamination the
        per-vault ``.migrated`` marker makes stick. This is a DISTINCT issue
        from the Ingest cross-bleed: the batch-slicing fix does NOT cover
        migration fan-out. Do not remove the gate (now LIVE, not simulated).
        """
        return sorted({self._entry_slug(entry) for entry in batch})

    async def run(self) -> bool:
        """Process unprocessed history entries. Returns True if work was done."""
        # Task 4.6: serialize EVERY Dream.run() (cron tick vs /dream's
        # create_task) on the one shared loop. Outermost lock; the cursor
        # read below is INSIDE it, so a waiting run sees the prior run's
        # advance and no-ops instead of double-processing the same batch.
        async with _DREAM_RUN_LOCK:
            from nanobot.agent.skills import BUILTIN_SKILLS_DIR

            last_cursor = self.store.get_last_dream_cursor()
            entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
            if not entries:
                return False

            batch = entries[: self.max_batch_size]
            logger.info(
                "Dream: processing {} entries (cursor {}→{}), batch={}",
                len(entries), last_cursor, batch[-1]["cursor"], len(batch),
            )

            # Build history text for LLM — cap each entry so a legacy oversized
            # record (e.g. pre-#3412 raw_archive dump) can't blow up the prompt.
            history_text = "\n".join(
                f"[{e['timestamp']}] "
                f"{truncate_text(e['content'], self.history_entry_preview_max_chars)}"
                for e in batch
            )

            # Current file contents + per-line age annotations (MEMORY.md only).
            # Each file is capped in the *prompt preview* only; Phase 2 still sees
            # the full file via the read_file tool.
            current_date = self._today()
            raw_memory = self.store.read_memory() or "(empty)"
            annotated_memory = (
                self._annotate_with_ages(raw_memory)
                if self.annotate_line_ages
                else raw_memory
            )
            current_memory = truncate_text(annotated_memory, self.memory_file_max_chars)
            current_soul = truncate_text(
                self.store.read_soul() or "(empty)", self.soul_file_max_chars,
            )
            current_user = truncate_text(
                self.store.read_user() or "(empty)", self.user_file_max_chars,
            )
            journal_section = self._build_journal_section()

            file_context = (
                f"## Current Date\n{current_date}\n\n"
                + (f"{journal_section}\n\n" if journal_section else "")
                + f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
                f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
                f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
            )

            # Phase 1: Analyze (no skills list — dedup is Phase 2's job)
            phase1_prompt = (
                f"## Conversation History\n{history_text}\n\n{file_context}"
            )

            try:
                phase1_response = await self.provider.chat_with_retry(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": render_template(
                                "agent/dream_phase1.md",
                                strip=True,
                                stale_threshold_days=_STALE_THRESHOLD_DAYS,
                            ),
                        },
                        {"role": "user", "content": phase1_prompt},
                    ],
                    tools=None,
                    tool_choice=None,
                )
                analysis = phase1_response.content or ""
                logger.debug("Dream Phase 1 analysis ({} chars): {}", len(analysis), analysis[:500])
            except Exception:
                logger.exception("Dream Phase 1 failed")
                return False

            # Phase 2: Delegate to AgentRunner with read_file / edit_file
            existing_skills = self._list_existing_skills()
            skills_section = ""
            if existing_skills:
                skills_section = (
                    "\n\n## Existing Skills\n"
                    + "\n".join(f"- {s}" for s in existing_skills)
                )
            phase2_prompt = f"## Analysis Result\n{analysis}\n\n{file_context}{skills_section}"

            tools = self._tools
            skill_creator_path = BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"
            journal_path = f"memory/journal/{current_date}.md"
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": render_template(
                        "agent/dream_phase2.md",
                        strip=True,
                        skill_creator_path=str(skill_creator_path),
                        journal_path=journal_path,
                        daily_notes_enabled=self.daily_notes_enabled,
                    ),
                },
                {"role": "user", "content": phase2_prompt},
            ]

            try:
                result = await self._runner.run(AgentRunSpec(
                    initial_messages=messages,
                    tools=tools,
                    model=self.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    fail_on_tool_error=False,
                ))
                logger.debug(
                    "Dream Phase 2 complete: stop_reason={}, tool_events={}",
                    result.stop_reason, len(result.tool_events),
                )
                for ev in (result.tool_events or []):
                    logger.info("Dream tool_event: name={}, status={}, detail={}", ev.get("name"), ev.get("status"), ev.get("detail", "")[:200])
            except Exception:
                logger.exception("Dream Phase 2 failed")
                result = None

            # Build changelog from tool events
            changelog: list[str] = []
            if result and result.tool_events:
                for event in result.tool_events:
                    if event["status"] == "ok":
                        changelog.append(f"{event['name']}: {event['detail']}")

            # Only advance cursor on successful completion to prevent silent loss
            if result and result.stop_reason == "completed":
                new_cursor = batch[-1]["cursor"]
                self.store.set_last_dream_cursor(new_cursor)
                logger.info(
                    "Dream done: {} change(s), cursor advanced to {}",
                    len(changelog), new_cursor,
                )
            else:
                reason = result.stop_reason if result else "exception"
                logger.warning(
                    "Dream incomplete ({}): cursor NOT advanced, will retry next cron cycle",
                    reason,
                )

            self.store.compact_history()

            # Git auto-commit (only when there are actual changes)
            if changelog and self.store.git.is_initialized():
                ts = batch[-1]["timestamp"]
                summary = f"dream: {ts}, {len(changelog)} change(s)"
                commit_msg = f"{summary}\n\n{analysis.strip()}"
                sha = self.store.git.auto_commit(commit_msg)
                if sha:
                    logger.info("Dream commit: {}", sha)

            # --- Wiki-tree memory (Task 4.5) -----------------------------------
            # STRICTLY ADDITIVE, best-effort, gated. Reached only after the entire
            # legacy MEMORY.md path above (Phase 1/2, cursor advance,
            # compact_history, git commit) has run EXACTLY as in v0.2.0, and only
            # on the success path (we are past the `if not entries: return False`
            # guard, so there ARE entries / `batch` is non-empty). The whole block
            # is wrapped so ANY Ingest/Lint/Vault exception is logged and
            # SWALLOWED: it cannot alter the cursor, the changelog/git commit, the
            # compacted history, Phase 1/2, or the `return True` below. With
            # `wiki_enabled` False (default) this is a true no-op -> Dream is
            # byte-identical to v0.2.0 (TestDreamWikiDisabledGolden enforces this).
            if self.wiki_enabled:
                # C1 (review follow-up): legacy migration is gated to the
                # UNIFIED vault ONLY. migrate_legacy reads the SINGLE GLOBAL
                # workspace/memory/MEMORY.md + workspace/USER.md (there is
                # exactly ONE such pair for the whole workspace, NOT one per
                # user). Passing legacy_workspace for a per-user slug would
                # import that global blob — including whatever USER.md profile
                # is on disk, possibly another user's — into THAT user's
                # vault: silent, PERMANENT cross-user contamination (the
                # per-vault .migrated marker makes it stick). Task 7.2 made
                # per-user routing LIVE (_vaults_for_batch returns one slug
                # per distinct user); the gate keeps migration UNIFIED-ONLY so
                # the global blob is NEVER fanned into a per-user vault. DO NOT
                # remove this `slug == unified` gate — see migrate_legacy's
                # docstring warning and the _vaults_for_batch NOTE.
                #
                # I1 (review follow-up): per-slug isolation. The try/except is
                # now INSIDE the `for slug` loop so a transient/corrupt-vault
                # failure for ONE user (run_ingest/run_lint/Vault raising) is
                # logged WITH its slug and SWALLOWED for THAT slug ONLY — the
                # remaining users still ingest this cycle. The old batch-global
                # try wrapped the WHOLE loop: one bad vault aborted every user
                # sorted after it while the cursor had ALREADY advanced (above)
                # → unrecoverable for them, defeating 7.2's per-user
                # independence. _vaults_for_batch itself is now TOTAL
                # (_entry_slug coerces any non-(non-empty-str) session_key —
                # absent/None/""/0/non-str/list/dict — to the unified slug)
                # so the grouping call (which runs OUTSIDE this per-iteration
                # try, in the `for slug` header) can NEVER raise. The legacy
                # path / cursor advance / compact / git / Phase 1-2 all ran
                # BEFORE this block and are UNCHANGED; per-iteration isolation
                # (not un-advancing the cursor) is the correct mitigation —
                # the raw history.jsonl still retains the failed slug's batch.
                unified = vault_slug("unified:default")
                for slug in self._vaults_for_batch(batch):
                    try:
                        vault = Vault(
                            self.store.workspace / "memory" / "users" / slug
                        )
                        # Task 7.1: pass the workspace so the FIRST
                        # wiki-enabled cycle one-shot-migrates the LEGACY
                        # global memory/MEMORY.md + root USER.md into the
                        # UNIFIED vault (gated by migrate_legacy's own
                        # .migrated marker), then run_lint below builds the
                        # MOC — closing the 6.1 enable-ordering window.
                        # Per-user vaults pass None: they MUST NOT import the
                        # global blob (C1). This C1 7.1 gate stays
                        # per-iteration verbatim. Wiki-off never reaches here.
                        vault.ensure_initialized(
                            self.store.workspace if slug == unified else None
                        )
                        # Task 7.2: feed Ingest ONLY this slug's slice of the
                        # batch via the SAME _entry_slug used for grouping
                        # (C1 — null-safe; grouping and slicing can never
                        # diverge) — NOT the whole batch. Passing the full
                        # batch would cross-bleed every user's history into
                        # every per-user vault.
                        slug_batch = [
                            e for e in batch if self._entry_slug(e) == slug
                        ]
                        # Same lock key the wiki_note tool takes
                        # (get_vault_lock(vault_slug(session_key))) so
                        # Dream-side Ingest/Lint and the agent-side wiki_note
                        # tool never write one user's vault concurrently
                        # (design H2). Each user's vault locks INDEPENDENTLY
                        # (per-slug lock).
                        async with get_vault_lock(slug):
                            # Task 4 (attachments-ingest): walk workspace/peer/
                            # and media/telegram/ and write any new textual
                            # attachments as inbox/* pages. Idempotent via
                            # per-vault sha256 manifest. The eager hook in
                            # AgentLoop (Task 5) has already handled fresh
                            # deliveries; this catches files that arrived
                            # out-of-band or while the eager hook was down.
                            # v1 routing: unified slug only (the reconciler
                            # walks raw files without sender info; per-user
                            # fan-out requires sidecar metadata deferred to a
                            # follow-up). The inner try/except keeps a
                            # reconciler failure from also skipping Ingest /
                            # Lint for this slug.
                            if slug == unified:
                                try:
                                    await run_attachments_reconcile(vault, slug)
                                except Exception:
                                    logger.exception(
                                        "attachments reconcile failed for vault "
                                        "{} (non-fatal — Ingest+Lint proceed)",
                                        slug,
                                    )
                            await run_ingest(
                                vault, slug_batch, self.provider, self.model,
                                render_template,
                            )
                            run_lint(vault, _date.today())
                            # Refresh dense embeddings (opt-in). Both the flag
                            # AND a non-empty model are required: an unset model
                            # means the dense tier is effectively unconfigured,
                            # so skip silently rather than fail every cycle. The
                            # import stays lazy so a stock install never pulls
                            # numpy/fastembed here.
                            if self.wiki_embeddings and self.wiki_embedding_model:
                                from nanobot.agent.wiki.embeddings import (
                                    refresh_embeddings,
                                )
                                refresh_embeddings(vault, self.wiki_embedding_model)
                    except Exception:
                        logger.exception(
                            "wiki ingest/lint failed for vault {}; other "
                            "vaults + legacy path intact",
                            slug,
                        )

            return True
