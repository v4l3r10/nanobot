# Memory in nanobot

nanobot's memory is built on a simple belief: memory should feel alive, but it should not feel chaotic.

Good memory is not a pile of notes. It is a quiet system of attention. It notices what is worth keeping, lets go of what no longer needs the spotlight, and turns lived experience into something calm, durable, and useful.

That is the shape of memory in nanobot.

## The Design

nanobot does not treat memory as one giant file.

It separates memory into layers, because different kinds of remembering deserve different tools:

- `session.messages` holds the living short-term conversation.
- `memory/history.jsonl` is the running archive of compressed past turns.
- `SOUL.md`, `USER.md`, and `memory/MEMORY.md` are the durable knowledge files.
- `GitStore` records how those durable files change over time.

This keeps the system light in the moment, but reflective over time.

## The Flow

Memory moves through nanobot in two stages.

### Stage 1: Consolidator

When a conversation grows large enough to pressure the context window, nanobot does not try to carry every old message forever.

Instead, the `Consolidator` summarizes the oldest safe slice of the conversation and appends that summary to `memory/history.jsonl`.

This file is:

- append-only
- cursor-based
- optimized for machine consumption first, human inspection second

Each line is a JSON object:

```json
{"cursor": 42, "timestamp": "2026-04-03 00:02", "content": "- User prefers dark mode\n- Decided to use PostgreSQL"}
```

It is not the final memory. It is the material from which final memory is shaped.

### Stage 2: Dream

`Dream` is the slower, more thoughtful layer. It runs on a cron schedule by default and can also be triggered manually.

Dream reads:

- new entries from `memory/history.jsonl`
- the current `SOUL.md`
- the current `USER.md`
- the current `memory/MEMORY.md`

Then it works in two phases:

1. It studies what is new and what is already known.
2. It edits the long-term files surgically, not by rewriting everything, but by making the smallest honest change that keeps memory coherent.

This is why nanobot's memory is not just archival. It is interpretive.

## The Files

```text
workspace/
├── SOUL.md              # The bot's long-term voice and communication style
├── USER.md              # Stable knowledge about the user
└── memory/
    ├── MEMORY.md        # Project facts, decisions, and durable context
    ├── history.jsonl    # Append-only history summaries
    ├── .cursor          # Consolidator write cursor
    ├── .dream_cursor    # Dream consumption cursor
    └── .git/            # Version history for long-term memory files
```

These files play different roles:

- `SOUL.md` remembers how nanobot should sound.
- `USER.md` remembers who the user is and what they prefer.
- `MEMORY.md` remembers what remains true about the work itself.
- `history.jsonl` remembers what happened on the way there.

## Why `history.jsonl`

The old `HISTORY.md` format was pleasant for casual reading, but it was too fragile as an operational substrate.

`history.jsonl` gives nanobot:

- stable incremental cursors
- safer machine parsing
- easier batching
- cleaner migration and compaction
- a better boundary between raw history and curated knowledge

You can still search it with familiar tools:

```bash
# grep
grep -i "keyword" memory/history.jsonl

# jq
cat memory/history.jsonl | jq -r 'select(.content | test("keyword"; "i")) | .content' | tail -20

# Python
python -c "import json; [print(json.loads(l).get('content','')) for l in open('memory/history.jsonl','r',encoding='utf-8') if l.strip() and 'keyword' in l.lower()][-20:]"
```

The difference is philosophical as much as technical:

- `history.jsonl` is for structure
- `SOUL.md`, `USER.md`, and `MEMORY.md` are for meaning

## Commands

Memory is not hidden behind the curtain. Users can inspect and guide it.

| Command | What it does |
|---------|--------------|
| `/dream` | Run Dream immediately |
| `/dream-log` | Show the latest Dream memory change |
| `/dream-log <sha>` | Show a specific Dream change |
| `/dream-restore` | List recent Dream memory versions |
| `/dream-restore <sha>` | Restore memory to the state before a specific change |

These commands exist for a reason: automatic memory is powerful, but users should always retain the right to inspect, understand, and restore it.

## Versioned Memory

After Dream changes long-term memory files, nanobot can record that change with `GitStore`.

This gives memory a history of its own:

- you can inspect what changed
- you can compare versions
- you can restore a previous state

That turns memory from a silent mutation into an auditable process.

## Configuration

Dream is configured under `agents.defaults.dream`:

```json
{
  "agents": {
    "defaults": {
      "dream": {
        "intervalH": 2,
        "modelOverride": null,
        "maxBatchSize": 20,
        "maxIterations": 10
      }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `intervalH` | How often Dream runs, in hours |
| `modelOverride` | Optional Dream-specific model override |
| `maxBatchSize` | How many history entries Dream processes per run |
| `maxIterations` | The tool budget for Dream's editing phase |

In practical terms:

- `modelOverride: null` means Dream uses the same model as the main agent. Set it only if you want Dream to run on a different model.
- `maxBatchSize` controls how many new `history.jsonl` entries Dream consumes in one run. Larger batches catch up faster; smaller batches are lighter and steadier.
- `maxIterations` limits how many read/edit steps Dream can take while updating `SOUL.md`, `USER.md`, and `MEMORY.md`. It is a safety budget, not a quality score.
- `intervalH` is the normal way to configure Dream. Internally it runs as an `every` schedule, not as a cron expression.
- `wikiEnabled` (default `false`) turns on the per-user wiki-tree layer described in [Wiki-tree memory](#wiki-tree-memory-per-user-navigable-memory). `lintCadenceH` is reserved for a future decoupled Lint cadence and does not change behavior today.

Legacy note:

- Older source-based configs may still contain `dream.cron`. nanobot continues to honor it for backward compatibility, but new configs should use `intervalH`.
- Older source-based configs may still contain `dream.model`. nanobot continues to honor it for backward compatibility, but new configs should use `modelOverride`.

## Wiki-tree memory (per-user navigable memory)

### What it is

The wiki-tree layer is an optional, Obsidian-style per-user Markdown wiki built on top of the memory described above. It does not replace `SOUL.md`, `USER.md`, `MEMORY.md`, or `history.jsonl` — it adds a navigable knowledge tree the agent can walk on its own.

- The agent reads and writes leaf pages through the `wiki_note` tool (read / create / append / search).
- The Dream cycle's **Ingest** phase distills new `history.jsonl` entries into pages; its **Lint** phase curates them (decay, dedup, broken-link audit) and regenerates the navigation indexes.
- It is **off by default**. A stock install behaves exactly like pre-wiki nanobot.

The structure follows the "LLM Wiki" idea: pages filed by type, navigation through wikilinks and a small Map-of-Content (MOC), no embeddings and no database. Knowledge hierarchy lives in the link graph, not in folders.

### How to enable

Wiki-tree memory is gated by a single config knob under `agents.defaults.dream`:

```json
{
  "agents": {
    "defaults": {
      "dream": {
        "wikiEnabled": true
      }
    }
  }
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `wikiEnabled` | `false` | Master switch for the whole wiki subsystem (Ingest/Lint + per-user MOC prompt injection). |
| `lintCadenceH` | `null` | Reserved for a future decoupled Lint cadence. **Current behavior:** Lint runs on every Dream wiki cycle regardless of this value; the field is accepted but does not yet change cadence. This is a planned refinement, not current behavior. |

Enablement is operationally automatic: set `wikiEnabled: true` and the next Dream cycle does the rest (migration + first MOC). No manual vault setup is needed.

> Transient on first enable: between flipping the knob and the first Dream-Lint cycle, a user's prompt has the usual SOUL / AGENTS / TOOLS content but not yet wiki memory or profile. This is normal and self-resolves on the first Dream cycle that runs after enabling.

### The master switch (golden guarantee)

With `wikiEnabled: false` (the default), nanobot is byte-identical to pre-wiki nanobot:

- no `memory/users/` directory is ever created;
- the system prompt uses the global `memory/MEMORY.md` exactly as before;
- the Dream cycle behaves exactly as before (same cursor, compaction, and git path).

It is safe to ship the feature dark and leave it off indefinitely.

### Per-user vaults

When enabled, each session gets its own *vault*:

```text
workspace/memory/users/<vault_slug>/
├── MEMORY.md          # the per-user MOC — this is what gets injected into THAT user's prompt
├── USER.md            # per-user profile (the vault owns the profile when wiki is active)
├── .lint.log          # append-only audit of every Lint run
├── .migrated          # one-shot migration marker (unified_default only)
└── wiki/
    ├── SCHEMA.md       # this vault's schema (copied from the bundled master on first use)
    ├── people/    _index.md  alice.md …
    ├── projects/  _index.md  payment-svc.md …
    ├── concepts/  _index.md  auth-model.md …
    ├── decisions/ _index.md  …
    └── .cold/<type>/…  # decayed pages, dropped from the MOC, still readable by the tool
```

`vault_slug` maps the session key `channel:chat_id` to a filesystem-safe directory name by replacing `:` with `_` and stripping unsafe characters — e.g. `telegram:1` → `telegram_1`. Vaults are strictly isolated: one user's pages, indexes, and MOC never bleed into another's.

### Unified session collapse

`unified_session: true` routes every channel into the single `unified_default` vault (single-user, multi-device). The same collapse is the back-compat floor for anything that is not a clean per-user key: legacy untagged history, and any malformed `session_key` (JSON `null`, a non-string like `123`, an empty string, a list/dict) all route to `unified_default`. This routing is total — a bad `session_key` value can never crash the Dream cycle; it just lands in the unified vault.

### Migration (one-time, automatic)

The first wiki-enabled Dream cycle migrates the legacy global memory into the wiki:

- the global `memory/MEMORY.md` becomes one `concepts/imported-memory.md` page, and the root `USER.md` is copied into the vault's `USER.md`;
- this happens **only for the `unified_default` vault** (the C1 invariant). Per-user vaults never receive the global blob, so a multi-user deployment cannot leak one user's profile into another's vault;
- it is strictly one-shot per vault, guarded by the `.migrated` marker;
- the legacy files are **never deleted** — they remain on disk as the git-history floor;
- a stock/template `MEMORY.md` is detected and not imported as if it were real memory.

So enablement is just the config flip: the next Dream cycle migrates and Lint builds the MOC.

### Decay and reheat lifecycle

Each type has a `cold_after_days` budget in `SCHEMA.md`. On Lint:

- a hot page whose `last_touched` is older than its type's budget is moved to `wiki/.cold/<type>/` and dropped from the MOC;
- `pinned: true` in a page's frontmatter makes it immune to decay;
- an explicit `wiki_note` read of a cold page reheats it in place (status flips back to hot, `last_touched` is bumped); the **next** Lint relocates it out of `.cold/` back to its hot home and re-indexes it into the MOC;
- `append` deliberately refuses `.cold/` paths — cold pages reheat only via `read`.

Nothing is hard-deleted: `.cold/` plus git history are the floor. The only thing that truly removes a page is a Lint dedup/merge or a manual deletion.

### SCHEMA.md knobs

The bundled master schema ships at `nanobot/templates/memory/wiki/SCHEMA.md` and is copied into each vault on first use:

```yaml
types:
  people:    { folder: people,    cold_after_days: 180 }
  projects:  { folder: projects,  cold_after_days: 90 }
  concepts:  { folder: concepts,  cold_after_days: 365 }
  decisions: { folder: decisions, cold_after_days: null }
required_frontmatter: [type, title, status, created, updated, last_touched]
moc_max_lines: 120
```

- `types` defines the allowed page types, their filing folder, and per-type decay (`cold_after_days: null` = never decays);
- `required_frontmatter` is the admission gate — a page missing any of these keys (or with an unknown `type`) is refused by the `wiki_note` tool;
- `moc_max_lines` is the soft cap on the regenerated MOC.

To customize types or decay policy for a vault, edit that vault's `wiki/SCHEMA.md` (a per-vault `SCHEMA.md` overrides the bundled master and is never overwritten once present). Editing the bundled master changes the default for vaults created afterward.

### `/dream-log` and `/dream-restore`

These commands now cover the wiki as well as the legacy memory files, because `GitStore` versions everything under `memory/users/**`:

- `/dream-log` shows the latest Dream change, including wiki page changes;
- `/dream-restore` with no arguments lists recent commits;
- `/dream-restore <sha>` is a true per-commit inverse over the tracked memory tree: it undoes exactly that commit's changes (including wiki pages it added/modified) while preserving later and unrelated pages and the legacy files;
- restoring a commit with nothing to undo (e.g. the first version) returns a plain "Nothing to undo" message — that is informational, not an error.

### Resilience and operational notes

- **Per-user isolation of failures:** if Ingest or Lint raises for one vault, the failure is logged with that vault's slug and swallowed. The other users still ingest, the legacy memory path and Dream cursor are unaffected, and that user's batch stays in `history.jsonl` for a later cycle.
- **Idempotent on crash-resume:** re-delivering the same history batch with the same model output is a byte-stable no-op (Ingest skips text already present; Lint only writes when bytes would change).
- **Per-vault lock (H2):** the same per-vault async lock serializes same-user Dream-Ingest against the `wiki_note` tool, while distinct users' vaults proceed concurrently.
- **Total read-side routing:** the `history.jsonl` `session_key` field is read defensively — any malformed value routes to `unified_default` rather than raising.

> ⚠ **Production prerequisite (known caveat, not yet implemented).** The process-wide Dream-run lock is held across the Ingest LLM call, and there is currently **no outer Dream timeout** — only the provider SDK's default request timeout. A wedged LLM call would therefore hold the global Dream lock and stall subsequent Dream cycles. Before enabling `wikiEnabled: true` in a production deployment, ensure a bounded provider request timeout (or keep the gate off). This is a known operational caveat and a planned hardening, not a blocker for the gated-off default.

## In Practice

What this means in daily use is simple:

- conversations can stay fast without carrying infinite context
- durable facts can become clearer over time instead of noisier
- the user can inspect and restore memory when needed

Memory should not feel like a dump. It should feel like continuity.

That is what this design is trying to protect.
