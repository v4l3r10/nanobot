# Wiki-tree memory (opt-in) — branch overview

This branch ports the **wiki-tree memory** subsystem onto `nightly`. It gives the agent a
navigable, Obsidian-style per-user memory: typed Markdown leaf pages + `[[wikilinks]]` + an
auto-generated Map-of-Content (MOC) index, with hot/cold decay and BM25 (optionally dense)
retrieval. Design rationale and the architecture are in
[`docs/wiki-tree-memory-design.md`](docs/wiki-tree-memory-design.md).

Discussion / RFC: **HKUDS/nanobot#4241**.

## What's in this branch (PR1 scope)

The **engine + tool + read/write path**, fully wired and **opt-in (default off)**:

- `nanobot/agent/wiki/` — the engine (vault, page, schema, ingest, lint, decay, links graph,
  retrieval, embeddings, moc_refresh, migrate, attachments).
- `nanobot/agent/tools/wiki_note.py` — the `wiki_note` agent tool (read/create/append/search/bind).
  Auto-discovered and **gated**: registered only when `wiki_enabled` is true.
- Read path: `agent/context.py` injects the per-user MOC (and the vault's `USER.md`) into the
  system prompt when the wiki is active; otherwise the prompt is byte-identical to stock.
- Write path: the tool writes pages immediately; `agent/loop.py` refreshes the MOC post-turn
  (cheap, decoupled from consolidation) and routes attachments into the vault.
- Config: `dream.wiki_enabled` (master switch, **default false**), `dream.wiki_embeddings`,
  `dream.wiki_embedding_model`, plus `unified_memory` (shared-vault multi-user routing, default off).

Enable it with `agents.defaults.dream.wiki_enabled = true`. With it off, behaviour is unchanged.

## What's intentionally NOT here yet

Per #4241, consolidation was re-architected on the fork around the v0.2.1 two-phase `Dream`
class, which #3990 removed on `nightly`. So this branch deliberately stops at the **dual-pen
write path + read path**, which is fully functional without consolidation:

- **Automatic consolidation (Ingest + Lint on the cron `dream` job)** — the next step, pending the
  Option A/B decision in #4241.
- **Multi-user vault routing + legacy migration** — follows after.

## Notes for reviewers

- Integration was reconciled onto `nightly` via 3-way merge against the v0.2.1 base; the whole
  `nanobot` package compiles (`python -m compileall nanobot`).
- The wiki tests under `tests/agent/wiki/`, `tests/agent/tools/test_wiki_note.py`, and the
  read-path tests (`test_context_wiki.py`, `test_loop_moc_refresh.py`,
  `test_context_memory_skill_wiki.py`, `test_memory_key_resolution.py`) target this scope. The
  consolidation/Dream e2e tests were deferred together with the consolidation work.
- Dense retrieval is optional (off by default, behind the `wiki-search` extra → `fastembed`);
  BM25 works with zero extra dependencies.
