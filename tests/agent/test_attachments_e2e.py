"""Task 7 — End-to-end smoke test for the attachments-wiki-ingest feature.

This is the convergence safety net for the two paths that ingest attachment
files into the wiki:

* Task 5 — the eager hook in :meth:`AgentLoop._eager_attachment_ingest` fires
  the moment an :class:`InboundMessage` with non-empty ``media`` arrives on the
  bus, writing each file as an ``inbox/<slug>.md`` page in the unified vault;
* Task 4 — the Dream-side reconciler ``run_attachments_reconcile`` walks
  conventional workspace dirs on each Dream cycle and writes anything not yet
  in the per-vault manifest.

The two paths share:

* :func:`~nanobot.agent.wiki.attachments_writer.write_attachment_page` — the
  one deterministic writer both paths call;
* ``<vault>/wiki/.ingested_attachments.json`` — the per-vault manifest keyed
  on ``sha256(content)`` that gates the writer's idempotence.

This is the convergence guarantee: a file ingested by the eager hook is
detected as a ``duplicate`` by a subsequent reconciler sweep via the sha256
gate, so the resulting page is byte-stable across the combined eager + Dream
operation. Re-running does not grow the page, does not add a manifest entry,
does not bump dates.

Design choice — **simulate convergence by exercising both entry points
directly**:

* eager path: call the real ``AgentLoop._eager_attachment_ingest`` (the same
  method ``run()`` schedules as a task in production);
* Dream path: call ``run_attachments_reconcile`` directly (the same function
  the Dream loop calls inside its per-slug ``get_vault_lock`` block, see
  ``tests/agent/test_dream_wiki.py::TestDreamRunsAttachmentsReconciler``).

This is functionally equivalent to running a real loop + a real Dream — both
real entry points converge on ``write_attachment_page`` + the manifest, which
is the load-bearing surface — at a fraction of the harness complexity. A
"real loop + real Dream in one test" variant would test a strict superset of
nothing this file already covers (the Dream wiring is locked by
``test_dream_wiki.py`` and the eager-hook scheduling by
``test_loop_attachments_hook.py``).

The single load-bearing claim: **the eager hook writes a page; the reconciler
sees the same source file, computes the same sha256, finds the manifest entry
the eager write made, returns ``duplicate``, and does NOT touch the page**.
The byte-stability assertion would fail if any of:

* Task 5 (eager hook) were reverted — no page would exist before the
  reconciler runs;
* Task 4 (reconciler wiring / iterator) were reverted — the reconciler
  would not see the source file at all (passes trivially, see the second
  test for the assertion that DOES catch this);
* the sha256 dedup in the writer were broken — the reconciler would
  re-write the page (different timestamp body) and the page bytes would
  change.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.wiki.attachments_reconciler import run_attachments_reconcile
from nanobot.agent.wiki.paths import vault_slug
from nanobot.agent.wiki.vault import Vault
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.utils.vault_lock import get_vault_lock


# --------------------------------------------------------------------------- #
# Helpers — mirror the loop fixture in test_loop_attachments_hook.py + the
# vault-init helper in test_attachments_reconciler.py.
# --------------------------------------------------------------------------- #


def _bundled_schema_text() -> str:
    return Path("nanobot/templates/memory/wiki/SCHEMA.md").read_text(encoding="utf-8")


def _make_loop(tmp_path: Path, *, wiki_enabled: bool = True) -> AgentLoop:
    """Build a minimal AgentLoop pointed at ``tmp_path``. Mirrors
    ``test_loop_attachments_hook.py::_make_loop`` — keep in sync."""
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


def _init_unified_vault(tmp_path: Path) -> Vault:
    """Materialize the unified vault on disk so ``wiki_dir.exists()`` is True
    (the eager hook's gate). Mirrors the helper of the same name in
    ``test_loop_attachments_hook.py``."""
    slug = vault_slug("unified:default")
    vault = Vault(tmp_path / "memory" / "users" / slug)
    vault.wiki_dir.mkdir(parents=True, exist_ok=True)
    (vault.wiki_dir / "SCHEMA.md").write_text(
        _bundled_schema_text(), encoding="utf-8"
    )
    return vault


@pytest.fixture
def _scope_allowed_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Pin BOTH the eager-hook writer's containment roots AND the
    Dream-side reconciler's default source roots to ``tmp_path`` so the
    same files are visible to both paths.

    * ``loop.get_workspace_path`` / ``loop.get_media_dir`` — these are the
      ``allowed_roots`` the eager hook passes to the writer's containment
      check (a path under NEITHER would be rejected).
    * ``attachments_reconciler.get_workspace_path`` / ``get_media_dir`` —
      these drive ``_default_sources()``: the reconciler walks
      ``<workspace>/peer`` and ``<media_dir>/telegram``. Without this
      override the reconciler would walk the real user paths and never
      see the test's tmp_path files.
    """
    monkeypatch.setattr(
        "nanobot.agent.loop.get_workspace_path", lambda: tmp_path
    )
    monkeypatch.setattr(
        "nanobot.agent.loop.get_media_dir", lambda: tmp_path
    )
    monkeypatch.setattr(
        "nanobot.agent.wiki.attachments_reconciler.get_workspace_path",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "nanobot.agent.wiki.attachments_reconciler.get_media_dir",
        lambda: tmp_path,
    )


# --------------------------------------------------------------------------- #
# THE SMOKE TEST
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_eager_then_dream_reconcile_is_noop(
    tmp_path: Path, _scope_allowed_roots: None
) -> None:
    """A file ingested by the eager hook is detected as ``duplicate`` by the
    next Dream-side reconciler sweep — the inbox page is byte-stable across
    the two paths.

    Convergence guarantee under combined eager + Dream operation: the manifest
    sha256 gate makes the two paths converge to one page, not two competing
    writes. This is the single load-bearing claim of Task 7.
    """
    # --- arrange: unified vault initialised, source file under workspace/peer --
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    # Drop the file under the peer layout the reconciler walks
    # (workspace/peer/<msg_id>/<file>). The eager hook routes it via
    # InboundMessage.media; the reconciler's peer iterator finds it on disk.
    # Same physical file => same sha256 => same manifest key.
    peer_dir = tmp_path / "peer" / "msg_e2e_001"
    peer_dir.mkdir(parents=True)
    src = peer_dir / "smoke.md"
    src.write_text("end to end content for the smoke test", encoding="utf-8")

    msg = InboundMessage(
        channel="peer",
        sender_id="u1",
        chat_id="c1",
        content="here is a file",
        media=[str(src)],
    )

    # --- act 1: eager hook (Task 5) ------------------------------------------
    await loop._eager_attachment_ingest(msg)

    inbox = (
        tmp_path / "memory" / "users" / vault_slug("unified:default")
        / "wiki" / "inbox"
    )
    page = inbox / "smoke.md"
    assert page.is_file(), (
        "eager hook did not write the inbox page — Task 5 broken"
    )
    page_after_eager = page.read_bytes()
    assert b"end to end content for the smoke test" in page_after_eager

    manifest_path = (
        tmp_path / "memory" / "users" / vault_slug("unified:default")
        / "wiki" / ".ingested_attachments.json"
    )
    manifest_after_eager = manifest_path.read_bytes()

    # --- act 2: Dream-side reconciler (Task 4) -------------------------------
    # Same lock the Dream block holds in production (memory.py per-slug block).
    slug = vault_slug("unified:default")
    async with get_vault_lock(slug):
        report = await run_attachments_reconcile(
            Vault(tmp_path / "memory" / "users" / slug), slug,
        )

    # --- assert: convergence -------------------------------------------------
    # The reconciler saw the source file under peer/ and computed sha256;
    # the sha was already in the manifest (the eager hook put it there); the
    # writer short-circuited with status="duplicate". If the sha256 dedup
    # were broken the reconciler would have re-written the page with a new
    # timestamped Source: header and the bytes would differ.
    assert report.duplicates >= 1, (
        f"reconciler did NOT detect the eager-written file as a duplicate — "
        f"convergence broken (sha256 gate or manifest read failed). "
        f"report={report!r}"
    )
    assert report.created == [], (
        f"reconciler created a NEW page for an already-ingested file — "
        f"manifest gate failed. created={report.created!r}"
    )
    assert report.appended == [], (
        f"reconciler appended to an already-ingested file — sha256 dedup "
        f"failed. appended={report.appended!r}"
    )

    # The load-bearing byte-stability assertion: re-running the second path
    # did NOT mutate the page on disk (no fresh timestamp, no doubled body).
    assert page.read_bytes() == page_after_eager, (
        "inbox page bytes changed after the Dream reconciler ran on a file "
        "already ingested by the eager hook — the writer's sha256 idempotence "
        "guard is broken"
    )

    # Manifest is also byte-stable: a duplicate hit must not append a new
    # entry (the writer short-circuits before _append_entry/_save_manifest).
    assert manifest_path.read_bytes() == manifest_after_eager, (
        "manifest grew on a duplicate sha256 — the writer is not short-"
        "circuiting on the manifest gate as documented"
    )


# --------------------------------------------------------------------------- #
# Supporting test — the reverse direction: reconciler-then-eager is ALSO a
# no-op on the second path. Together with the test above this proves the
# convergence is symmetric (either path may be the "first" writer for a
# given file; the other must be a duplicate). Acceptable per the plan
# ("Adding 1-2 supporting tests is acceptable").
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dream_reconcile_then_eager_is_noop(
    tmp_path: Path, _scope_allowed_roots: None
) -> None:
    """Reverse path: a file first written by the Dream reconciler is detected
    as ``duplicate`` by a subsequent eager-hook call on the SAME file. Proves
    the convergence is symmetric — the manifest sha256 gate doesn't care
    which path made the original write."""
    loop = _make_loop(tmp_path, wiki_enabled=True)
    _init_unified_vault(tmp_path)

    peer_dir = tmp_path / "peer" / "msg_e2e_002"
    peer_dir.mkdir(parents=True)
    src = peer_dir / "reverse.md"
    src.write_text("reverse path content", encoding="utf-8")

    # Reconciler first.
    slug = vault_slug("unified:default")
    async with get_vault_lock(slug):
        report = await run_attachments_reconcile(
            Vault(tmp_path / "memory" / "users" / slug), slug,
        )
    page = (
        tmp_path / "memory" / "users" / slug / "wiki" / "inbox" / "reverse.md"
    )
    assert page.is_file(), (
        "reconciler did not write the inbox page — Task 4 broken"
    )
    assert "reverse path content" in page.read_text(encoding="utf-8")
    assert "inbox/reverse.md" in report.created
    page_after_recon = page.read_bytes()
    manifest_path = (
        tmp_path / "memory" / "users" / slug / "wiki"
        / ".ingested_attachments.json"
    )
    manifest_after_recon = manifest_path.read_bytes()

    # Eager hook on the SAME file: must short-circuit via the manifest.
    msg = InboundMessage(
        channel="peer",
        sender_id="u1",
        chat_id="c1",
        content="same file",
        media=[str(src)],
    )
    await loop._eager_attachment_ingest(msg)

    assert page.read_bytes() == page_after_recon, (
        "inbox page bytes changed after the eager hook ran on a file already "
        "ingested by the reconciler — sha256 idempotence broken in reverse"
    )
    assert manifest_path.read_bytes() == manifest_after_recon, (
        "manifest grew on a duplicate sha256 from the eager path — short-"
        "circuit gate failed in the reverse direction"
    )
