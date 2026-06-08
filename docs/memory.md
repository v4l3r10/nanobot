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

Then it edits the long-term files surgically in a single pass — not by rewriting everything, but by making the smallest honest change that keeps memory coherent.

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
      "unifiedMemory": false,
      "dream": {
        "intervalH": 2,
        "modelOverride": null,
        "maxBatchSize": 20,
        "maxIterations": 10,
        "wikiEnabled": false,
        "wikiEmbeddings": false,
        "wikiEmbeddingModel": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "lintCadenceH": null
      }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `intervalH` | How often Dream runs, in hours |
| `cron` | Cron expression override (takes precedence over `intervalH`) |
| `modelOverride` | Optional Dream-specific model override *(pending implementation)* |
| `maxBatchSize` | *(Deprecated — not used)* |
| `maxIterations` | *(Deprecated — not used)* |
| `wikiEnabled` | Master switch for [wiki-tree memory](#wiki-tree-memory-opt-in) (default `false`) |
| `wikiEmbeddings` | Add dense-embedding retrieval over the vault, on top of BM25 (default `false`; needs the `wiki-search` extra) |
| `wikiEmbeddingModel` | Embedding model id used when `wikiEmbeddings` is on |
| `lintCadenceH` | How often the deterministic vault Lint runs, in hours (`null` = each consolidation). *Reserved for the consolidation pass.* |
| `unifiedMemory` *(on `agents.defaults`, not `dream`)* | Share one memory/wiki vault across users while keeping per-user chat sessions (default `false`) |

In practical terms:

- `intervalH` is the normal way to configure Dream frequency. Internally it runs as an `every` schedule.
- `cron` overrides `intervalH` when set, allowing precise cron expressions (e.g. `0 */4 * * *`).
- `modelOverride` is reserved for a future release. Currently Dream uses the same model as the main agent.
- `maxBatchSize` and `maxIterations` are preserved for config compatibility but no longer affect behavior.
- `wikiEnabled` turns on the wiki-tree memory layer described below. With it off (the default), the
  system prompt and memory behavior are byte-identical to a stock install.
- All keys accept their snake_case form too (`wiki_enabled`, `unified_memory`, …) — both spellings work.

## Wiki-tree memory (opt-in)

When `wikiEnabled` is on, durable knowledge is stored not as flat prose but as a navigable,
Obsidian-style **per-user vault**: typed Markdown leaf pages (one subject per page) connected by
`[[wikilinks]]`, with an auto-generated **Map-of-Content (MOC)** index. It is opt-in and
default-off; the full design rationale is in
[`wiki-tree-memory-design.md`](wiki-tree-memory-design.md).

How it changes the flow:

- **Read path.** Instead of pasting the whole memory into every turn, only the compact MOC index
  (plus the vault's `USER.md`) is injected at context-build time. Per-turn cost stays roughly
  constant as the vault grows; the agent navigates from the index to the specific page it needs.
- **Write path ("dual pen").** The `wiki_note` tool writes pages immediately during a turn, so
  nothing is lost between consolidations. The MOC is refreshed cheaply after each turn.
- **Retrieval.** A `search` over the vault ranks pages with BM25, and — when `wikiEmbeddings` is on
  — fuses in dense-embedding similarity. This is what makes long-term facts *recalled by topic*,
  not just stored.
- **Hot/cold decay.** Pages untouched past a per-type threshold move to a cold area and reheat on
  read, so the working set stays bounded without anything being deleted.

The vault lives under `memory/users/<key>/` and is auto-committed via `GitStore`, so wiki memory is
versioned and restorable just like the rest.

> **Status:** this opt-in layer ships the engine, the `wiki_note` tool, and the read/write path.
> Automatic consolidation (distilling history into pages on the Dream cron) is tracked separately —
> see HKUDS/nanobot#4241. Until then, pages are written by the tool during conversations.

## In Practice

What this means in daily use is simple:

- conversations can stay fast without carrying infinite context
- durable facts can become clearer over time instead of noisier
- the user can inspect and restore memory when needed

Memory should not feel like a dump. It should feel like continuity.

That is what this design is trying to protect.
