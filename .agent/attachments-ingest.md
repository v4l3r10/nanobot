# Attachments -> Wiki Ingest

## Why

Files delivered through channels (peer, telegram, future) used to sit in the
workspace until the model decided to call `wiki_note` on them. In a focused
long_task the model wouldn't notice -- 12+ hour indexing latency was observed
on real deliveries. Deterministic indexing was needed, decoupled from any
model decision.

## Architecture

Two write paths into the wiki, one shared writer, one shared manifest:

```
                          +------------------------------+
   channel.media[] ------>| AgentLoop.run()              |  eager (< 1s)
                          | -> _eager_attachment_ingest  |
                          +--------------+---------------+
                                         |
                                         v
                          +------------------------------+
                          | write_attachment_page()      |
                          | (attachments_writer.py)      |
                          +--------------+---------------+
                                         ^
                                         |
   workspace/peer/  -----> +------------------------------+
   media/telegram/  -----> | run_attachments_reconcile()  |  scheduled
                           | (attachments_reconciler.py)  |  (Dream cycle)
                           +------------------------------+
                                          ^
                                          | called by memory.py inside
                                          | the per-slug get_vault_lock
```

Both writers funnel through `write_attachment_page` in `attachments_writer.py`.
Both share the per-vault `.ingested_attachments.json` manifest. They coexist
because each closes a gap the other can't: the eager hook gives < 1s latency
for live deliveries but only sees what flows through `InboundMessage.media`;
the reconciler catches files dropped out-of-band (Umanio / peer delivery
that bypasses the bus), files that arrived before the hook was deployed, and
files for which the eager task crashed. The manifest's sha256 gate makes the
overlap a no-op.

## Writer (`nanobot/agent/wiki/attachments_writer.py`)

Pure synchronous function of (source file bytes, vault state, manifest). No
LLM, no line-protocol parsing: the `Page` dataclass is built directly and
serialized via `serialize_page` + `atomic_write_text`.

- Extension classification via `classify_extension`: textual set is
  `{.md, .txt, .json, .yaml, .yml, .csv}`, everything else is recorded as
  `skipped_binary` (manifest entry, no page).
- Containment check: `is_under(resolved_src, root)` for at least one root in
  the caller-supplied `allowed_roots`. Anything escaping is rejected with
  `status="error"`. Defense in depth against symlink games inside the
  source dirs.
- Dedup gates, cheapest first:
  1. `sha256(content)` looked up in the manifest -- no body read on hit.
  2. `_body_already_present(page.body, clamped)` for the rare case where
     the manifest missed but the body is already on the existing page
     (different framing, manual paste, etc.).
- Reuses `_safe_slug`, `_slug_ok`, `_body_already_present`, and the
  `_MAX_BODY_CHARS` clamp from `nanobot/agent/wiki/ingest.py` so attachment
  writes cannot diverge from the model-side ingest pipeline on slug
  containment (C1) or body clamping (C2).
- Schema gate: refuses to write if the vault schema lacks an `inbox` type.
  The one-shot upgrade in `vault.py::_ensure_inbox_type` handles older vaults.
- Output routing in v1: every page lands at `inbox/<slug>.md` in the
  **unified** vault. Per-sender fan-out is deferred (needs sender->slug
  metadata not yet standardized across channels).
- Page body always starts with a `Source: {channel}/{msg_id} - {ts}` header
  so provenance is preserved on append.

The function is sync on purpose: it has no awaitable I/O and is called from
both async (`loop.py`) and async-but-from-an-iterator (`reconciler.py`) sites.
Sync keeps the call surface uniform. The `slug` parameter is documentary --
the caller must already hold `get_vault_lock(slug)`; the writer does not
assert it.

## Reconciler (`nanobot/agent/wiki/attachments_reconciler.py`)

Walks two source layouts:

- **peer**: `<workspace>/peer/<msg_subdir>/<file>` -- `_iter_peer_files`
  treats each `msg_subdir` (named via `safe_filename(msg_id)` by `peer.py`)
  as the `msg_id`. Dot-prefixed subdirs/files are skipped.
- **telegram (flat)**: `<media_dir>/telegram/<file>` -- `_iter_flat_files`
  uses the file stem as `msg_id`. Subdirs and dot-files are skipped.

`_default_sources()` builds the production registry from
`get_workspace_path()` + `get_media_dir()` (config-aware roots). Tests inject
their own list via the `sources` parameter on `run_attachments_reconcile`.

Iteration is OSError-safe at every per-entry step (the directory may mutate
mid-walk on Windows; transient failures skip the entry and the next cycle
catches up).

Called from `memory.py`'s Dream loop, **inside** `get_vault_lock(slug)`,
**before** `run_ingest`, gated to `slug == unified` (matches the writer's
v1 unified-only routing). The inner `try/except` keeps a reconciler failure
from skipping Ingest + Lint for that slug.

## Eager hook (`AgentLoop._eager_attachment_ingest` in `nanobot/agent/loop.py`)

Fire-and-forget `asyncio.create_task` scheduled from `run()` immediately
after `consume_inbound`, gated by `if msg.media:` so the plain-text path
stays zero-cost. The coroutine itself:

- Returns immediately if `self.context.wiki_enabled` is off.
- Returns immediately if the unified vault's `wiki_dir` does not exist yet
  (the wiki was just enabled and Dream's `ensure_initialized` hasn't run --
  the reconciler will pick the files up on the next sweep).
- Acquires `get_vault_lock(slug)` for the unified slug (shares serialization
  with `wiki_note`, `run_ingest`, `run_lint`, `_refresh_vault_moc`).
- For each path in `msg.media`, derives a `msg_id` that matches the
  reconciler's convention so the two paths produce **byte-identical**
  `Source:` headers: if `p.parent.parent.name == msg.channel` it's the
  peer-style per-message subdir layout -> use `p.parent.name`; otherwise
  flat layout -> use `p.stem`.
- Any exception is logged and swallowed -- this code MUST NEVER interfere
  with message dispatch.

## Manifest schema

Path: `<vault>/wiki/.ingested_attachments.json`. The dot prefix keeps
Lint's `rglob("*.md")` from ever seeing it.

```json
{
  "version": 1,
  "entries": [
    {
      "sha256": "<hex>",
      "channel": "peer|telegram|...",
      "msg_id": "<string>",
      "status": "ingested|skipped_binary",
      "path": "<absolute resolved src path>",
      "size": <bytes>,
      "ingested_at": "YYYY-MM-DDTHH:MM:SSZ",
      "page": "inbox/<slug>.md"      // omitted for skipped_binary
    }
  ]
}
```

Corrupt / malformed manifests are logged and replaced with a fresh empty
shell -- the worst case is one cycle of redundant writes (still gated by
`_body_already_present`). Entries are never pruned; growth is linear in the
number of **distinct** attachments received. At current volumes this is fine;
if it ever isn't, the file is trivially compactable by sha.

## Convergence guarantee

Rerun is a no-op and byte-stable:

1. `sha256` is computed first; if the manifest already has that hash, the
   function returns **without reading the file body** -- nothing on disk
   changes.
2. If the manifest missed but the target page already contains the body,
   `_body_already_present(page.body, clamped)` short-circuits the append
   path, records the manifest entry (so step 1 fires next time), and
   returns `duplicate`. The page is not rewritten.
3. The clamp (`_MAX_BODY_CHARS`) and `Source:` header are deterministic
   functions of inputs, so a fresh write of the same bytes at the same
   `msg_id` produces the same serialized page.

## `inbox` schema type

Added by `_ensure_inbox_type` in `nanobot/agent/wiki/vault.py` (idempotent
one-shot upgrade for pre-existing vaults):

```
inbox:     { folder: inbox,     cold_after_days: 30 }
```

`cold_after_days: 30` is the contract with Lint: an inbox page unused for
30 days is cooled (status -> cold), eligible for archival or distillation.
The next Dream cycle's `run_ingest` sees `inbox/*` in `existing_pages` and
can APPEND distilled content into the same page -- the attachment writer's
body clamp is in lockstep with `ingest.py`'s clamp (both import
`_MAX_BODY_CHARS` from the same module), so append events from either side
respect the same ceiling.

## v1 limitations (deferred)

- **Per-sender routing for the reconciler.** Requires sender->slug metadata
  not yet standardized across channels -- v1 routes ALL attachments to the
  unified vault.
- **Binary OCR / Whisper pipeline.** Manifest records non-textual files as
  `skipped_binary`; a future pass can re-process those entries when the
  pipeline lands.
- **Wiki-aware-memory skill nudge.** No "you have N unindexed files" hint
  yet; the model still discovers via `wiki_search` over the inbox folder.
- **`complete_goal` drain hook.** Considered and dropped: the eager hook
  already covers fresh deliveries, and the Dream-cycle reconciler covers
  everything else within one cycle.

## References

- Implementation plan: [`docs/plans/2026-05-20-attachments-wiki-ingest.md`](../docs/plans/2026-05-20-attachments-wiki-ingest.md)
- Writer module: `nanobot/agent/wiki/attachments_writer.py`
- Reconciler module: `nanobot/agent/wiki/attachments_reconciler.py`
- Eager hook: `nanobot/agent/loop.py` (`_eager_attachment_ingest`)
- Dream wiring: `nanobot/agent/memory.py` (in the per-slug loop, before `run_ingest`)
- Schema upgrade: `nanobot/agent/wiki/vault.py` (`_ensure_inbox_type`)
- Shared primitives: `nanobot/agent/wiki/ingest.py` (`_safe_slug`, `_body_already_present`, `_MAX_BODY_CHARS`)
