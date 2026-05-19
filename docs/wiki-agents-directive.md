# Wiki `wiki_note` directive for the bot's `AGENTS.md`

This is a behavioral policy: it makes the agent proactively *capture* durable facts (and *recall* them before answering from ignorance) via the `wiki_note` tool, instead of waiting to be asked. It is policy only — the `wiki_note` tool self-describes its own mechanics in its schema, so this block deliberately does not repeat them.

**Where it goes:** append the block below, verbatim, to the **end of the bot's *workspace* `AGENTS.md`** — the per-user/agent workspace bootstrap file, **not** this repository's `AGENTS.md`. That workspace file is re-read every turn, so the directive takes effect from the very next message with no restart. It is fully reversible: delete the block to revert. It is also **inert unless the wiki master switch is enabled** (`dream.wiki_enabled` — written `wikiEnabled` under `agents.defaults.dream` in `config.json`; see [How to enable](./memory.md#how-to-enable)) — with the wiki off the `wiki_note` tool is not even registered, so the directive has nothing to act on.

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

See [`wiki-moc-decouple-design.md`](./wiki-moc-decouple-design.md) for the design rationale and [`memory.md`](./memory.md) for the memory-system overview.
