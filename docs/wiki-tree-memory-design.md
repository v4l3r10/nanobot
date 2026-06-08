# Design — Obsidian-style "Wiki-Tree" memory for nanobot

- **Date:** 2026-05-18
- **Status:** validated design (brainstorming complete, 6/6 sections approved)
- **Implementation status: IMPLEMENTED.** The feature is built behind the knob
  `dream.wiki_enabled` (default `false`); milestones M0–M7 complete. The reference behavioral
  spec is the end-to-end suite `tests/agent/test_wiki_e2e.py` (scenarios A–I): any claim about
  implemented behavior should be verified there.
- **Driver author:** valerio.cavagni@gmail.com
- **Related documents:**
  - `REPORT_memoria_nanobot_vs_openhuman.md` (comparison and recommendations P1–P8, H1–H3)
  - `MECCANICHE_MEMORIA_openhuman.md` (exhaustive openhuman reference)
  - Karpathy "LLM Wiki" gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

---

## 0. Context and motivation

A comparison of nanobot (v0.2.0) vs openhuman shows that nanobot's gap **is not storage** but
*retrieval* (pure recency, no semantics), *latency* (long-term knowledge only via the ~2h Dream
batch), and the **absence of multi-user isolation**.

openhuman defects **not** to replicate: unbounded monotonic growth with no forgetting (no
TTL/prune), linear vector scan, `learning.enabled=false` by default.

**Architectural decision:** adopt neither embeddings nor SQLite. We keep Markdown + git + Dream as
the *curated source of truth*, and on top we build a **navigable Obsidian-style wiki** (`[[ ]]`
wikilinks + a Map-of-Content) with **explicit hot/cold forgetting**. The structure follows the
principles of Karpathy's "LLM Wiki" gist: 3 layers (raw sources / LLM-maintained wiki / schema),
3 operations (Ingest, Query, Lint), essentially flat pages with `index.md` + `log.md`, a Lint
that cures contradictions / stale / orphan / missing-link, and the LLM doing the bookkeeping.

---

## 1. Architecture and on-disk layout

A **per-user** vault, derived from the session key (`channel:chat_id` → fs-safe `user_slug`).
This addresses recommendation **P3** (per-user namespacing).

```
workspace/
  SOUL.md                       # GLOBAL — the bot's identity
  AGENTS.md / TOOLS.md          # global
  memory/
    wiki/SCHEMA.md              # global master schema (hand-curated, versioned)
    users/<user_slug>/          # fs-safe user_slug from channel:chat_id
      MEMORY.md                 # root MOC → [[people/_index]] [[projects/_index]] …  (ALWAYS in prompt)
      USER.md                   # per-user profile (fix P3)
      history.jsonl             # = Karpathy's log.md, per-user
      wiki/
        SCHEMA.md               # copy of the master schema
        people/    _index.md  alice.md …
        projects/  _index.md  payment-svc.md …
        concepts/  _index.md  auth-model.md …
        decisions/ _index.md  YYYY-<slug>.md …
        .cold/<type>/…          # decayed pages, out of the MOC, readable by the tool
        .lint.log               # append-only audit of Dream-Lint runs
  .git/                         # dulwich versions everything under memory/
```

- `unified_session=true` → a single `users/unified/` vault (backward-compat with current behavior).
- **Karpathy mapping:** raw sources = session JSONL; `log.md` = `history.jsonl`; `index.md` =
  `MEMORY.md` (root MOC); wiki = `wiki/*.md`; schema = `SCHEMA.md`.
- **Filing by type** (stable), *not* by Area/topic (folder hierarchies are discouraged by
  Karpathy). The knowledge hierarchy lives in the **link graph**, not in folders.
- **Per-page frontmatter:** `type, title, status (hot|cold), created, updated, last_touched,
  tags[], links_out[], pinned (optional)`.

---

## 2. SCHEMA.md — declarative layer

A global master schema, hand-curated and versioned in git; copied per-vault at bootstrap. It
defines:

- The allowed **types** (`people`, `projects`, `concepts`, `decisions`, extensible) and the
  filing folder for each.
- `cold_after_days` per type (defaults: people 180, projects 90, concepts 365, decisions = never).
- **Required** frontmatter fields and slug naming rules.
- A cap on the MOC size (max rows/entries in the root digest).
- Filing rules (which type to assign new content to).

Dream-Lint reads SCHEMA to validate and curate. Changing the forgetting policy or adding a type =
editing SCHEMA, not the code.

---

## 3. Write path (dual pen)

Two pens write to the wiki:

**a) Agent tool `wiki_note`** — operations `read(path)`, `create(type, slug, …)`,
`append(path, text)`, `search(query)`. The agent can **create/append leaf pages** and add the
**stub-link in the `_index.md`** of the type. It cannot: move to `.cold/`, merge duplicates,
rewrite `_index.md`/the root MOC (those are **Dream-only** operations). Every write is **validated
against SCHEMA**: malformed frontmatter or a wrong type → reject with a clear error, so the model
self-corrects.

**b) Dream-Lint (batch, cadence = `dream.interval_h`)** — **Ingest** phase: from `history.jsonl`
it extracts facts, creates/updates pages, records contradictions (without overwriting), updates
`last_touched`.

Hardening: **atomic writes** (`tmp` + `os.replace`) → fix **H1**; **per-vault file lock** → fix
**H2**. Dual-write means knowledge is available **immediately** via the agent tool, while Dream
consolidates afterward: this mitigates latency **P2** without removing Dream.

---

## 4. Read path

- The **MOC** (per-user `MEMORY.md`) is **always in the prompt**, but small: links to the per-type
  `_index` pages + a recent-hot digest. It replaces the current monolithic `MEMORY.md` injection
  and shortens the last-50 `history.jsonl` replay to a minimal recent tail.
- **Model-driven** navigation via the `wiki_note.read` tool following wikilinks (no embeddings).
- A `search` operation (keyword/tag/recency over **hot + cold**) for discoverability of pages not
  linked from the MOC.
- **Reheat-on-read:** a read updates `last_touched`; if the page was `cold` it returns to
  `status: hot`; the next Lint physically removes it from `.cold/`.
  - **Hand-off reheat / `.cold/` (contract pinned for Lint, milestone 4.3):**
    - Reading a page whose path is under `.cold/` rehydrates it **in place**
      (`status: cold`→`hot`, `last_touched`=today), leaving it **physically** in `.cold/`.
    - It is **Lint's** job (a Dream phase, milestone 4.3) to **physically relocate out** of
      `.cold/` any page with `status: hot` still sitting under a `.cold` component (relocation
      key: `status == "hot" and ".cold" in path.parts`).
    - `append` instead **refuses** paths under `.cold/` (cold pages only reactivate via `read`);
      this *read-reheats / append-refuses* asymmetry is intentional.

This replaces recommendation **P1** (semantic index) with explicit wikilink navigation + MOC +
keyword search — a deliberate choice: no vector store.

---

## 5. Hot/cold forgetting + Lint

- `cold_after_days` per type from SCHEMA; once the inactivity threshold (`last_touched`) is
  crossed, Lint moves the page to `.cold/<type>/`, out of the MOC but still readable by the tool.
- `pinned: true` → immune to decay.
- **No automatic hard-delete.** `.cold/` is the floor; git keeps history; only a **merge** (dedup)
  or a manual deletion truly removes a page. A cap on `.cold/` = **YAGNI, deferred**.
- **Lint** (Karpathy's 4 checks + dedup): contradictions (recorded, not overwritten), stale → cold,
  orphan (page unreachable from the MOC), missing/broken link (via `links_out`), dedup/merge of
  duplicate pages. Then it **regenerates** each type's `_index.md` and the root `MEMORY.md` **from
  the frontmatter** (deterministic).
- `.lint.log` = append-only audit of each run (what was cooled, merged, repaired).

This realizes the **admission gate** (**P7**: cheap signals + threshold, no embeddings) and the
**hierarchical/per-topic Dream** (**P8**: Lint operates per type).

---

## 6. Hooking into nanobot code, edge cases, testing

**Hook points (indicative references, nanobot v0.2.0):**

- `agent/context.py:37-76` + `:63-71` — inject the per-user MOC instead of the monolithic
  `MEMORY.md`; reduce the last-50 replay to a small recent tail.
- `agent/memory.py:205-226` + `MemoryStore` — per-user paths; `MEMORY.md` becomes a generated
  artifact; fix atomic writes (H1) here and in the tool.
- `agent/memory.py:785-1087` (`Dream`) + `templates/agent/dream_phase{1,2}.md` — Dream →
  SCHEMA-driven Ingest+Lint.
- `session/manager.py` — derive `user_slug` from the session key; honor `unified_session`.
- **New tool `wiki_note`** in `agent/tools/`, sandboxed to the workspace like
  `ReadFileTool`/`EditFileTool`, registered in the tool registry.
- `utils/gitstore.py:45-391` — extend tracked paths to `memory/users/**` (so `/dream-log` and
  `/dream-restore` cover the wiki).
- `config/schema.py` — knobs: enable wiki, `cold_after_days` defaults (overridden from SCHEMA),
  per-user vault toggle (tied to `unified_session`), Lint cadence = `dream.interval_h`.

**Edge cases:** one-time migration of legacy `MEMORY.md`/`USER.md`/`history.jsonl` → per-user
vault (the first Dream bootstraps the wiki from the existing MEMORY.md); wiki disabled → identical
behavior to today (backward-compat); malformed page → the tool validates against SCHEMA and
rejects, Lint repairs anyway; slug collision on reheat → Lint resolves via merge/suffix; per-vault
lock, Dream skips a busy vault and retries; Dream backlog (P2) → tool writes keep knowledge
available, optional opportunistic trigger.

**Testing:**
- *Unit:* SCHEMA parser/validator; frontmatter round-trip; decay predicate; orphan/broken-link
  detector; dedup/merge; deterministic MOC/`_index` regeneration; atomic write; `user_slug`
  derivation.
- *Integration:* turn → `wiki_note.create` → next turn sees it via navigation; Dream Ingest from a
  synthetic `history.jsonl` integrates+links; decay → `.cold`; reheat via search+read;
  contradiction recorded not overwritten; post-Dream git commit contains the expected tree.
- *Multi-user isolation:* two slugs don't contaminate each other; `unified_session` collapses to
  one vault.
- *Golden regression:* "wiki off = current nanobot".
- *Migration:* legacy workspace → migrated vault with no data loss, git history preserved.

---

## 7. Mapping of report recommendations

| Recommendation | How this design addresses it |
|---|---|
| **P1** semantic index | Replaced by wikilink navigation + MOC + `search` keyword (explicit choice: no embeddings) |
| **P2** Dream latency | Mitigated by dual-write: the agent tool makes knowledge available immediately |
| **P3** per-user namespacing | `users/<user_slug>/` vault + per-user `USER.md` |
| **P7** admission gate | SCHEMA validation at write + admission in Lint (cheap signals) |
| **P8** hierarchical/per-topic Dream | Lint operates per type, regenerates per-type `_index` |
| **H1** atomic writes | `tmp` + `os.replace` in the tool and in `memory.py` |
| **H2** file-lock | Per-vault lock; Dream skips busy vaults |

---

## 8. What we do NOT do (YAGNI)

- No embeddings / vector store / SQLite.
- No cap or physical eviction on `.cold/` (deferred).
- No automatic hard-delete.
- No folder hierarchy by Area/topic (the hierarchy lives in the link graph; physical filing stays
  by type, stable).

---

## 9. Open questions

- **Location/versioning of this document:** the working tree where this design originated is not a
  git repo; the `nanobot/` clone is a third-party repo (HKUDS/nanobot). To decide: leave the design
  at the workspace root with the other artifacts, or initialize a dedicated git repo for the
  workspace.
  - **Resolved (M7).** The document lives in-repo as `docs/wiki-tree-memory-design.md` (flat
    Markdown, next to `docs/memory.md` / `docs/configuration.md`), versioned on the feature branch.
    The operator/user doc is the "Wiki-tree memory (per-user navigable memory)" section added to
    `docs/memory.md` (Task 7.4). The plan working doc stays out of the repo (`docs/plans/`,
    gitignored).
- The `search` format detail (keyword/tag/recency ranking) to be fixed during implementation
  planning.
  - **Resolved (M2, Task 2.4).** `wiki_note search` (`agent/tools/wiki_note.py`) is plain
    keyword/tag/recency over hot **and** cold pages (`iter_pages(include_cold=True)`), no
    embeddings/SQLite. Three mutually-exclusive modes: empty query → most recent by `last_touched`
    desc (relpath asc tiebreak); `tag:` prefix → exact case-insensitive tag match; otherwise →
    case-insensitive substring with field-presence scoring (title 3, tag 2, body 1), ordered score
    desc / `last_touched` desc / relpath asc. Capped at 20 results with an overflow note. `search`
    is read-only and does **not** reheat (only `read` rehydrates a cold page).

---

## 10. Closure mapping — open questions → milestones

The remaining design decisions left open during brainstorming were closed as follows (reference
the `tests/agent/test_wiki_e2e.py` suite, scenarios A–I):

- **Session-key plumbing in the ContextBuilder (§6):** explicit `session_key` argument passed to
  `build_system_prompt` (no ContextVar) — resolved **Task 6.1** (`agent/context.py`); with wiki off
  or no key the prompt is byte-identical to pre-wiki (scenario I).
- **Multi-user history attribution (§1, §6):** `append_history` tagged with `session_key` +
  full read-side routing via `_entry_slug` (`agent/memory.py`) — resolved **Task 7.2**; each vault
  receives only its slice of the batch, no cross-bleed (scenario B).
- **Legacy unified-only migration (§6 edge case):** `migrate_legacy` runs only for
  `unified_default`, gated `slug == unified` at the Dream call-site; per-user vaults never receive
  the global blob (invariant C1) — resolved **Task 7.1** (`agent/wiki/migrate.py`), enforced under
  live routing (scenario G).
- **Hand-off reheat / `.cold/` (§4):** pinned contract — `read` rehydrates in place, the next Lint
  relocates out of `.cold/` (key `status == "hot" and ".cold" in path.parts`); `append` refuses
  cold paths — resolved **Task 2.3 / 4.3** (`agent/wiki/lint.py`, `wiki_note.py`), full round-trip
  verified (scenario C).
- **Cap/eviction on `.cold/` (§5, §8):** confirmed **YAGNI / deferred** per the design — no
  hard-delete, `.cold/` + git remain the floor.
- **Atomic writes (H1) and per-vault lock (H2):** `tmp` + `os.replace` (`utils/atomic.py`)
  everywhere in the tool, in Ingest, in Lint and in migrate; per-vault lock
  (`utils/vault_lock.get_vault_lock`) shared between Dream-Ingest and `wiki_note` — resolved
  **M0/M2/M4** (scenarios F, H).
- **Ingest/Lint idempotence on Dream retry (§6 testing):** substance guard C2 in Ingest
  (`_body_already_present`) + Lint writes only if bytes change — resolved **Task 4.3/4.4**; a
  re-delivered batch is byte-stable (scenario F).
- **Git versioning of the wiki (§6):** `GitStore` dynamically scans `memory/users/**` and includes
  it in `auto_commit`/`revert`; `/dream-log` and `/dream-restore` cover the wiki with per-commit
  inverses — resolved **Task 5.1/5.2** (`utils/gitstore.py`, `command/builtin.py`), scenario D.
- **Golden "wiki off = current nanobot":** `dream.wiki_enabled=false` default, zero
  `memory/users/` artifacts, prompt on the global `MEMORY.md` — resolved **Task 3.1 / 6.1**,
  enforced end-to-end (scenario I).
