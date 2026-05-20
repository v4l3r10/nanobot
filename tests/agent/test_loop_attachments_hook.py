"""Task 5 — eager attachment ingest hook in :meth:`AgentLoop.run`.

The hook fires fire-and-forget the moment an :class:`InboundMessage`
with non-empty ``media`` is consumed from the bus, writing each file
as an ``inbox/<slug>.md`` page into the **unified** vault (symmetric
with the Dream-side reconciler from Task 4). Behaviour contract:

* gated by ``self.context.wiki_enabled`` — strict no-op when off;
* gated by non-empty ``msg.media`` — strict no-op for plain text;
* gated by existence of the unified vault's ``wiki_dir`` — the
  reconciler will catch up on the next Dream sweep;
* exceptions are logged and swallowed — must never break dispatch;
* ``run()`` calls it via ``asyncio.create_task`` so the dispatch path
  is never blocked.

We test the helper method ``_eager_attachment_ingest`` directly (the
full-loop ``run()`` fixture is heavier than necessary; the integration
point is one line — ``asyncio.create_task(self._eager_attachment_ingest(msg))``
— and the unit-level coverage of the helper plus an integration test
that drives ``run()`` for a single tick give the same guarantees with
much less setup).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.wiki.paths import vault_slug
from nanobot.agent.wiki.vault import Vault
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


# Bundled schema for ensure_initialized() to find when called by tests.
def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


def _make_loop(tmp_path: Path, *, wiki_enabled: bool) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="ok", tool_calls=[])
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        wiki_enabled=wiki_enabled,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(  # type: ignore[method-assign]
        return_value=False
    )
    return loop


@pytest.fixture
def _scope_allowed_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Pin the writer's containment roots to ``tmp_path`` so test files
    are accepted. Production passes the real ``get_workspace_path()`` /
    ``get_media_dir()`` — those return user paths that won't contain
    pytest's temp dir."""
    monkeypatch.setattr(
        "nanobot.agent.loop.get_workspace_path", lambda: tmp_path
    )
    monkeypatch.setattr(
        "nanobot.agent.loop.get_media_dir", lambda: tmp_path
    )


def _init_unified_vault(tmp_path: Path) -> Vault:
    """Materialize the unified vault on disk so ``wiki_dir.exists()`` is True."""
    slug = vault_slug("unified:default")
    vault = Vault(tmp_path / "memory" / "users" / slug)
    vault.wiki_dir.mkdir(parents=True, exist_ok=True)
    (vault.wiki_dir / "SCHEMA.md").write_text(
        _bundled_schema_text(), encoding="utf-8"
    )
    return vault


# --- helper: with media -> writes inbox page to unified vault ---------------


@pytest.mark.asyncio
async def test_eager_hook_writes_inbox_page_for_media(
    tmp_path: Path, _scope_allowed_roots: None
) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    media_file = tmp_path / "media" / "telegram" / "snippet.txt"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_text("eager content", encoding="utf-8")

    msg = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="c1",
        content="here is a file",
        media=[str(media_file)],
    )

    await loop._eager_attachment_ingest(msg)

    unified = tmp_path / "memory" / "users" / vault_slug("unified:default") / "wiki"
    page = unified / "inbox" / "snippet.md"
    assert page.exists()
    assert "eager content" in page.read_text(encoding="utf-8")


# --- helper: no media -> no-op ----------------------------------------------


@pytest.mark.asyncio
async def test_eager_hook_noop_when_no_media(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    msg = InboundMessage(
        channel="telegram", sender_id="u", chat_id="c", content="hi"
    )

    await loop._eager_attachment_ingest(msg)

    inbox = (
        tmp_path / "memory" / "users" / vault_slug("unified:default")
        / "wiki" / "inbox"
    )
    assert not inbox.exists() or not list(inbox.iterdir())


# --- helper: wiki_enabled=False -> strict no-op (no vault dir created) ------


@pytest.mark.asyncio
async def test_eager_hook_strict_noop_when_wiki_disabled(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=False)
    assert loop.context.wiki_enabled is False

    media_file = tmp_path / "snippet.txt"
    media_file.write_text("x", encoding="utf-8")
    msg = InboundMessage(
        channel="telegram", sender_id="u", chat_id="c", content="x",
        media=[str(media_file)],
    )

    await loop._eager_attachment_ingest(msg)

    # Strict golden: wiki-OFF turn leaves zero on-disk vault artifacts.
    assert not (tmp_path / "memory" / "users").exists()


# --- helper: wiki_dir not yet initialized -> no-op (reconciler catches up) --


@pytest.mark.asyncio
async def test_eager_hook_noop_when_vault_uninitialized(tmp_path: Path) -> None:
    """First turn after wiki was just enabled: the unified vault's ``wiki_dir``
    doesn't exist yet. The hook must NOT call ensure_initialized (that's
    Dream's job) and must NOT write — the next reconciler sweep handles it."""
    loop = _make_loop(tmp_path, wiki_enabled=True)
    # Deliberately NOT initialising the vault.

    media_file = tmp_path / "snippet.txt"
    media_file.write_text("uninit", encoding="utf-8")
    msg = InboundMessage(
        channel="telegram", sender_id="u", chat_id="c", content="x",
        media=[str(media_file)],
    )

    await loop._eager_attachment_ingest(msg)

    # No vault dir was created by the hook.
    assert not (tmp_path / "memory" / "users").exists()


# --- helper: exceptions logged + swallowed ----------------------------------


@pytest.mark.asyncio
async def test_eager_hook_swallows_writer_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    media_file = tmp_path / "boom.txt"
    media_file.write_text("boom", encoding="utf-8")
    msg = InboundMessage(
        channel="t", sender_id="u", chat_id="c", content="x",
        media=[str(media_file)],
    )

    import nanobot.agent.loop as loop_mod

    boom = MagicMock(side_effect=RuntimeError("kaboom"))
    monkeypatch.setattr(loop_mod, "write_attachment_page", boom)

    # MUST NOT raise.
    await loop._eager_attachment_ingest(msg)
    boom.assert_called_once()


# --- helper: non-string / empty entries in media list are skipped ----------


@pytest.mark.asyncio
async def test_eager_hook_filters_invalid_media_entries(
    tmp_path: Path, _scope_allowed_roots: None
) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    # Only one valid entry; the others are filtered before reaching the writer.
    valid = tmp_path / "ok.txt"
    valid.write_text("ok", encoding="utf-8")
    msg = InboundMessage(
        channel="t", sender_id="u", chat_id="c", content="x",
        media=["", str(valid), None],  # type: ignore[list-item]
    )

    await loop._eager_attachment_ingest(msg)

    page = (
        tmp_path / "memory" / "users" / vault_slug("unified:default")
        / "wiki" / "inbox" / "ok.md"
    )
    assert page.exists()


# --- helper: missing source file is silently skipped ------------------------


@pytest.mark.asyncio
async def test_eager_hook_skips_missing_source_files(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    msg = InboundMessage(
        channel="t", sender_id="u", chat_id="c", content="x",
        media=[str(tmp_path / "does_not_exist.txt")],
    )

    # MUST NOT raise; nothing written.
    await loop._eager_attachment_ingest(msg)
    inbox = (
        tmp_path / "memory" / "users" / vault_slug("unified:default")
        / "wiki" / "inbox"
    )
    assert not inbox.exists() or not list(inbox.iterdir())


# --- integration: run() schedules the hook as a task (fire-and-forget) -----


@pytest.mark.asyncio
async def test_run_schedules_eager_hook_as_task(tmp_path: Path) -> None:
    """The integration leg: a single tick of ``run()`` that consumes an
    ``InboundMessage`` with ``media`` must spawn the eager hook as a task
    (``asyncio.create_task``), NOT ``await`` it inline. We stub the helper
    to record the call and check that ``run()`` returned to its loop body
    without waiting on it (the helper is allowed to be still pending after
    a single bus consume).
    """
    loop = _make_loop(tmp_path, wiki_enabled=True)

    seen: list[InboundMessage] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def _record(m: InboundMessage) -> None:
        seen.append(m)
        started.set()
        # Block until the test releases us — this proves the helper was
        # not awaited inline (otherwise run() would deadlock here).
        await release.wait()

    loop._eager_attachment_ingest = _record  # type: ignore[method-assign]

    # Make _dispatch a no-op so we don't need a full runner stack.
    loop._dispatch = AsyncMock()  # type: ignore[method-assign]

    media_file = tmp_path / "x.txt"
    media_file.write_text("x", encoding="utf-8")
    msg = InboundMessage(
        channel="t", sender_id="u", chat_id="c", content="hello",
        media=[str(media_file)],
    )

    # Start run() in the background and inject one message.
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(msg)
        # The hook should be invoked promptly without blocking dispatch.
        await asyncio.wait_for(started.wait(), timeout=3.0)
        assert seen == [msg]
    finally:
        release.set()
        loop.stop()
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_run_does_not_schedule_hook_when_no_media(tmp_path: Path) -> None:
    """The common case (plain-text message, no media) MUST NOT create a
    task — that's the perf short-circuit the implementation promises."""
    loop = _make_loop(tmp_path, wiki_enabled=True)

    called = False

    async def _record(m: InboundMessage) -> None:
        nonlocal called
        called = True

    loop._eager_attachment_ingest = _record  # type: ignore[method-assign]
    loop._dispatch = AsyncMock()  # type: ignore[method-assign]

    msg = InboundMessage(
        channel="t", sender_id="u", chat_id="c", content="plain text",
    )

    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(msg)
        # Give run() at least one full poll cycle to process the message.
        await asyncio.sleep(0.2)
        assert called is False
    finally:
        loop.stop()
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass
