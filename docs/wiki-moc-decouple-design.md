# Wiki MOC Decouple — Low-Latency Memory Update Design

> Status: VALIDATED (brainstorming, 2026-05-19). Successor refinement of
> [`wiki-tree-memory-design.md`](./wiki-tree-memory-design.md). Targets the
> latency/trigger gap that doc's open questions deferred.

**Goal:** make a durable fact reach the always-injected memory context
promptly, even on a lightly-used bot, without depending on the
session-long → consolidation → Dream-every-2h chain or on model caprice.

**Architecture (one line):** split the two concerns Dream currently fuses —
*index/MOC freshness* (must be low-latency; deterministic; LLM-free) vs.
*curation/quality* (can stay batched; expensive; non-deterministic) — and
drive freshness event-driven off real wiki writes.

---

## 1. Problem & root cause (verified in code)

The wiki has two write paths:

- **Path A — `wiki_note` tool (immediate, model-driven).** Writes the page
  file and the per-type `_index.md` stub in-turn (`wiki_note.py:737-757`).
  Write latency ≈ 0. But it fires only if the model decides to, which
  requires a behavioral directive that is not currently deployed on the
  target bot.
- **Path B — Dream-Lint batch (delayed).** turn → *maybe* consolidation
  (token/message threshold, read at boot → restart-bound;
  `loop.py:1094/1342/1424`) → `history.jsonl` → Dream every 2h or `/dream`
  → Ingest (1 LLM call) + Lint.

**The critical finding:** the only thing *always injected* into the prompt
is the **root `MEMORY.md` MOC** (`context.py`, capped 32k). `wiki_note`
**cannot** rewrite it — *"indexes/MOC files … are automatic and
Dream-only"* (`wiki_note.py:324`). So a freshly written fact is on disk
immediately but **not in the always-on context until Dream/Lint
regenerates the MOC**. Reliable *retrieval* therefore still waits for the
2h batch (or relies on the model choosing to `wiki_note search` — caprice).

Path B is double-gated (consolidation must fire **and** Dream must run); on
a lightly-used personal bot it is effectively dead (`history.jsonl` stays
at 0 bytes).

## 2. Why decoupling works (verified in `lint.py`)

`lint.py` is **entirely deterministic, no LLM, no network**. The
expensive/non-deterministic part of Dream is `ingest.py` (separate
module), *not* Lint. `run_lint` has 7 phases:

1–5. mutating curation: reheat-relocate, stale→cold decay (`should_cool`),
dedup/merge, broken-link audit — these move/merge files.
6. `_regenerate_indexes` — per-type `_index.md` from frontmatter.
7. `_regenerate_moc` — root `MEMORY.md` MOC from frontmatter, capped at
   `schema.moc_max_lines`.

Phases 6+7 are a **pure function of on-disk page frontmatter**, idempotent
(the doc-level invariant: byte-identical on a second run with no input
change). They need no history, no LLM, no consolidation, no Dream.

## 3. Solution — three workstreams

| # | Workstream | Status |
|---|---|---|
| 1 | **Immediate capture:** `wiki_note` behavioral directive in the bot's `AGENTS.md` (when to write, what is durable, search-before-create, recall-before-ignorance). | Text approved |
| 2 | **Cheap MOC rebuild:** standalone, deterministic, LLM-free pass = `_scan` (read-only) → `_regenerate_indexes` → `_regenerate_moc`, **skipping mutating phases 2–5**. Triggered event-driven on real wiki writes. | This design |
| 3 | **Heavy Dream unchanged**, demoted to a quality/cleanup pass (Ingest prose + Karpathy checks + dedup + decay), off the critical path. | No change |

### Two delivery tracks (the no-restart constraint)

- **Track A — runtime, zero restart.** The `AGENTS.md` block lives in the
  bot's *workspace* `AGENTS.md` (a bootstrap file re-read every turn → effect
  next message). Reversible (delete the block). Inert if `wikiEnabled` is
  false (the tool is not even registered — the I-1 `enabled()` gate).
- **Track B — code, on `feat/wiki-tree-memory`.** Workstream #2 is a code
  change → requires a build + one redeploy of the bot. Nothing to do at
  runtime.

## 4. Track B design — the cheap MOC rebuild

### 4.1 The pass

A new callable (working name `rebuild_indexes_and_moc(vault: Vault)`),
extracted from / composed of existing Lint internals:

```
_scan(vault)                 # read-only walk of wiki_dir (incl. .cold/)
  → _regenerate_indexes(...)  # phase 6, pure
  → _regenerate_moc(...)      # phase 7, pure, capped at schema.moc_max_lines
```

It MUST NOT run phases 2–5 (no reheat-relocate, no decay, no dedup, no
file moves). Side effects are confined to `_index.md` files and root
`MEMORY.md`. It reuses Lint's existing `_write_if_changed` so a no-op
write is a no-op (idempotent; `.lint.log` need not be touched, or a
distinct marker is used — decided in the plan).

### 4.2 Trigger (event-driven, approved)

A post-turn hook: if, during the just-finished turn, a `wiki_note`
`create` or `append` succeeded for this session's vault, run the cheap
rebuild **for that vault only**, under the **same per-vault lock**
`get_vault_lock(vault_slug(session_key))` that `wiki_note` and Dream's
wiki block take (serialized against a concurrent Dream on that vault; no
new global lock). Properties: tightest latency (MOC fresh by the next
turn), zero waste (skipped when no write happened), blast radius = one
vault, at most once per turn.

### 4.3 Invariants preserved

- **Master switch:** the hook is gated on the same resolved
  `dream.wiki_enabled` as everything else; `wikiEnabled=false` →
  byte-identical to stock v0.2.0 (golden tests must stay green).
- **Dream idempotence:** the cheap pass never alters page frontmatter or
  page locations, so a later full Dream over unchanged inputs still
  produces its byte-identical result. The cheap pass is itself idempotent.
- **Per-user isolation:** keyed by `vault_slug(session_key)` — never
  touches another user's vault; never runs `migrate_legacy` (that stays
  the `unified_default`-only Dream gate; see
  [[wiki-tree-impl-constraints]]).
- **Concurrency:** per-vault lock, not the global Dream lock — a cheap
  rebuild and a heavy Dream on different vaults do not block each other; on
  the same vault they serialize correctly.

### 4.4 Open implementation items (for the plan)

- Exact extraction shape in `lint.py`: factor phases 6+7 (and the
  read-only `_scan`) into a public callable without duplicating logic and
  without changing `run_lint`'s behavior (Dream still calls the full
  sequence). Verified separable; mechanics decided in `writing-plans`.
- Where the post-turn hook lives and how "a wiki write happened this turn
  for vault X" is signaled from `wiki_note` to the hook (an
  `AgentHook`/`runner` seam vs. a per-vault dirty flag) — chosen in the
  plan against the real `loop.py`/`runner.py` seams.
- `.lint.log` vs. a separate `.moc.log` marker for the cheap pass
  (auditability without polluting Dream's log / its idempotence check).

## 5. Track A — the `AGENTS.md` directive (approved, English)

Appended to the bot workspace `AGENTS.md`. It supplies **policy only** —
the `wiki_note` tool already self-describes its mechanics in its tool
schema, so this must not duplicate them.

```markdown
## Long-term memory (wiki_note)

You have a persistent per-user wiki memory via the `wiki_note` tool. Treat it
as your real memory across conversations — the chat history you see is only a
short recent window, not everything you know.

**Capture — write proactively, don't wait to be asked.**
When the user states a durable fact (who they are, their projects, ongoing
goals, stable preferences, decisions, recurring people/entities), record it
the same turn:
1. `search` for an existing page on that subject first.
2. If one exists, `append` the new fact to it.
3. If not, `create` a leaf page — closest existing type, one subject per page.
Do NOT record ephemera: one-off questions, transient state, things already on
a page, or anything the user did not actually assert.

**Recall — check before answering from ignorance.**
The most relevant pages are already summarized in the memory map in your
context. If the user asks something that depends on past knowledge and the map
does not cover it, `search` the wiki before answering — never claim you don't
know without checking.

Keep pages concise and factual. You cannot merge, move, or delete pages — that
curation is automatic; capture cleanly and a background pass tidies up.
```

Rationale: `search`-before-`create` because the tool refuses
merge/dedup (Dream-only); the *Recall* clause bridges the window between
the immediate write and the cheap MOC rebuild landing.

## 6. Testing

- Unit: `rebuild_indexes_and_moc` regenerates `_index.md`/`MEMORY.md` from
  a hand-built vault; idempotent on a second call; does **not** move
  pages, reheat, decay, or dedup (assert `.cold/` and page locations
  untouched).
- Integration: a `wiki_note create` followed by the post-turn hook makes
  the new page appear in the root `MEMORY.md` — assert it is in the
  context the *next* turn would inject.
- Golden / master switch: `wikiEnabled=false` → the hook is inert, system
  prompt and on-disk layout byte-identical to v0.2.0 (extend the existing
  golden suite; full `pytest -q` regression gate per
  [[regression-gate-baseline]]).
- Isolation: a write in vault A never regenerates vault B's MOC.
- Concurrency: cheap rebuild and Dream contend correctly on the same
  vault's lock; run independently on different vaults.

## 7. Out of scope (YAGNI)

- No change to consolidation / `history.jsonl` / Dream cadence / Ingest.
- No embeddings, SQLite, or semantic retrieval (unchanged project stance).
- No new global config knob unless the plan proves one is needed; reuse
  `dream.wiki_enabled` as the single master switch.
- `.cold/` unbounded growth, no outer Dream timeout — pre-existing,
  accepted residuals (see `wiki-tree-memory-design.md`); not reopened here.
