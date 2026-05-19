"""Task 4 — post-turn cheap MOC refresh trigger in ``AgentLoop._dispatch``.

Three behaviour groups:

  1. ``_refresh_vault_moc(session_key)`` regenerates the session's vault
     ``MEMORY.md`` (deterministic, LLM-free) from on-disk frontmatter AND
     never runs the legacy migration for a per-user vault (no ``.migrated``
     marker is created — proving cross-user-bleed isolation).
  2. With wiki ENABLED and the session's slug marked dirty, driving exactly
     one turn schedules exactly one refresh for THAT slug; not-dirty → none.
  3. With wiki DISABLED, even a dirty slug triggers NO refresh (the
     master-switch gate; golden byte-identity).

The vault for group 1 is REAL on ``tmp_path`` (production
serialize_page + bundled SCHEMA). Groups 2/3 assert the GATE logic
deterministically by monkeypatching ``loop._refresh_vault_moc`` to record
calls — no dependence on background-task timing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.wiki.moc_refresh import mark_vault_dirty, take_dirty
from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.agent.wiki.paths import vault_dir, vault_slug
from nanobot.agent.wiki.vault import Vault
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


def _write_hot_page(vault: Vault, rel: str, *, title: str) -> None:
    page = Page(
        type="people",
        title=title,
        status="hot",
        created="2020-01-01",
        updated="2026-05-18",
        last_touched="2026-05-18",
        tags=[],
        links_out=[],
        pinned=None,
        body="body\n",
    )
    p = vault.wiki_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(serialize_page(page), encoding="utf-8")


def _capture_scheduled(loop: AgentLoop) -> list[str]:
    """Make ``_schedule_background`` run its coro inline and record which
    session key ``_refresh_vault_moc`` was scheduled for.

    Deterministic: the gate's decision (scheduled? for which slug?) is
    asserted synchronously inside ``_dispatch`` instead of racing a real
    background task.
    """
    calls: list[str] = []

    async def _record(session_key: str) -> None:
        calls.append(session_key)

    loop._refresh_vault_moc = _record  # type: ignore[method-assign]

    def _inline(coro) -> None:
        # Drive the (already-created) coroutine to completion synchronously.
        try:
            coro.send(None)
        except StopIteration:
            pass

    loop._schedule_background = _inline  # type: ignore[method-assign]
    return calls


def _make_loop(tmp_path: Path, *, wiki_enabled: bool) -> AgentLoop:
    """A real AgentLoop whose LLM turn is stubbed to a trivial reply."""
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


# --- Group 1: real _refresh_vault_moc regeneration + no legacy migration ----


@pytest.mark.asyncio
async def test_refresh_vault_moc_regenerates_memory_md(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    session_key = "telegram:chat-1"

    vault = Vault(vault_dir(Path(tmp_path), session_key))
    vault.wiki_dir.mkdir(parents=True, exist_ok=True)
    (vault.wiki_dir / "SCHEMA.md").write_text(
        _bundled_schema_text(), encoding="utf-8"
    )
    _write_hot_page(vault, "people/alice.md", title="Alice")

    assert not (vault.root / "MEMORY.md").exists()

    await loop._refresh_vault_moc(session_key)

    moc = (vault.root / "MEMORY.md").read_text(encoding="utf-8")
    assert "[[people/alice]]" in moc
    # No legacy migration ran for a per-user vault: the .migrated marker
    # is the proof that migrate_legacy was (incorrectly) invoked.
    assert not (vault.root / ".migrated").exists()


@pytest.mark.asyncio
async def test_refresh_vault_moc_no_migrated_marker_even_with_legacy_files(
    tmp_path: Path,
) -> None:
    """Even when a global legacy memory/USER.md exists in the workspace,
    a per-user vault refresh must NOT migrate it (no .migrated marker)."""
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "MEMORY.md").write_text(
        "# Memory\n\nSECRET other-user profile data\n", encoding="utf-8"
    )
    (tmp_path / "USER.md").write_text("name: someone-else\n", encoding="utf-8")

    loop = _make_loop(tmp_path, wiki_enabled=True)
    session_key = "discord:guild:chan"

    await loop._refresh_vault_moc(session_key)

    vault_root = vault_dir(Path(tmp_path), session_key)
    assert not (vault_root / ".migrated").exists()
    # And the global legacy blob was NOT fanned into the per-user vault.
    imported = vault_root / "wiki" / "concepts" / "imported-memory.md"
    assert not imported.exists()


@pytest.mark.asyncio
async def test_refresh_vault_moc_swallows_failures(tmp_path: Path) -> None:
    """Best-effort: a failure inside the rebuild must not propagate."""
    loop = _make_loop(tmp_path, wiki_enabled=True)
    import nanobot.agent.loop as loop_mod

    boom = MagicMock(side_effect=RuntimeError("boom"))
    original = loop_mod.rebuild_indexes_and_moc
    loop_mod.rebuild_indexes_and_moc = boom
    try:
        # Must NOT raise.
        await loop._refresh_vault_moc("telegram:chat-err")
    finally:
        loop_mod.rebuild_indexes_and_moc = original
    boom.assert_called_once()


# --- Group 2: wiki ENABLED gate (dirty -> exactly one refresh) -------------


@pytest.mark.asyncio
async def test_dispatch_schedules_refresh_when_enabled_and_dirty(
    tmp_path: Path,
) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    calls = _capture_scheduled(loop)

    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="c1", content="hi"
    )
    sk = loop._effective_session_key(msg)
    mark_vault_dirty(vault_slug(sk))

    await loop._dispatch(msg)

    assert calls == [sk]
    # dirty flag consumed (take-and-clear): a second turn with no new
    # write does NOT re-trigger.
    msg2 = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="c1", content="again"
    )
    await loop._dispatch(msg2)
    assert calls == [sk]


@pytest.mark.asyncio
async def test_dispatch_no_refresh_when_enabled_but_not_dirty(
    tmp_path: Path,
) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=True)
    calls = _capture_scheduled(loop)

    msg = InboundMessage(
        channel="cli", sender_id="u1", chat_id="c-clean", content="hi"
    )
    sk = loop._effective_session_key(msg)
    # Defensively clear any stray dirty mark for this slug.
    take_dirty(vault_slug(sk))

    await loop._dispatch(msg)

    assert calls == []


@pytest.mark.asyncio
async def test_dispatch_refresh_runs_for_non_websocket_channels(
    tmp_path: Path,
) -> None:
    """The trigger is not nested inside the websocket-only block: it
    fires for every channel (here: discord)."""
    loop = _make_loop(tmp_path, wiki_enabled=True)
    calls = _capture_scheduled(loop)

    msg = InboundMessage(
        channel="discord", sender_id="u1", chat_id="c-dc", content="hi"
    )
    sk = loop._effective_session_key(msg)
    mark_vault_dirty(vault_slug(sk))

    await loop._dispatch(msg)

    assert calls == [sk]


# --- Group 3: wiki DISABLED master-switch gate -----------------------------


@pytest.mark.asyncio
async def test_dispatch_no_refresh_when_wiki_disabled_even_if_dirty(
    tmp_path: Path,
) -> None:
    loop = _make_loop(tmp_path, wiki_enabled=False)
    assert loop.context.wiki_enabled is False
    calls = _capture_scheduled(loop)

    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="c-off", content="hi"
    )
    sk = loop._effective_session_key(msg)
    mark_vault_dirty(vault_slug(sk))

    await loop._dispatch(msg)

    assert calls == []
    # The dirty mark was NOT even consumed by the disabled path
    # (the gate short-circuits before take_dirty). Clean up for isolation.
    take_dirty(vault_slug(sk))
