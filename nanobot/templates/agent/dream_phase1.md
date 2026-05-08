You have THREE equally important tasks:
1. Extract new facts from conversation history
2. Deduplicate existing memory files — find and flag redundant, overlapping, or stale content even if NOT mentioned in history
3. Capture today's significant events into the day's journal note

Output one line per finding:
[FILE] atomic fact (not already in memory)
[FILE-REMOVE] reason for removal
[SKILL] kebab-case-name: one-line description of the reusable pattern
[DAILY] section: short bullet for today's journal note

Files: USER (identity, preferences), SOUL (bot behavior, tone), MEMORY (knowledge, project context), DAILY (today's events/decisions/conversations)

Rules:
- Atomic facts: "has a cat named Luna" not "discussed pet care"
- Corrections: [USER] location is Tokyo, not Osaka
- Capture confirmed approaches the user validated

Deduplication — scan ALL memory files for these redundancy patterns:
- Same fact stated in multiple places (e.g., "communicates in Chinese" in both USER.md and multiple MEMORY.md entries)
- Overlapping or nested sections covering the same topic
- Information in MEMORY.md that is already captured in USER.md or SOUL.md (MEMORY.md should not duplicate permanent-file content)
- Verbose entries that can be condensed without losing information
For each duplicate found, output [FILE-REMOVE] for the less authoritative copy (prefer keeping facts in their canonical location)

Staleness — MEMORY.md lines may have a ``← Nd`` suffix showing days since last modification:
- SOUL.md and USER.md have no age annotations — they are permanent, only update with corrections
- Age only indicates when content was last touched, not whether it should be removed
- Use content judgment: user habits/preferences/personality traits are permanent regardless of age
- Only prune content that is objectively outdated: passed events, resolved tracking, superseded approaches
- Lines with ``← Nd`` (N>{{ stale_threshold_days }}) deserve closer review but are NOT automatically removable
- When removing: prefer deleting individual items over entire sections

Skill discovery — flag [SKILL] when ALL of these are true:
- A specific, repeatable workflow appeared 2+ times in the conversation history
- It involves clear steps (not vague preferences like "likes concise answers")
- It is substantial enough to warrant its own instruction set (not trivial like "read a file")
- Do not worry about duplicates — the next phase will check against existing skills

Daily journal — flag [DAILY] for today's significant events under one of these sections:
- Conversazioni — meaningful exchanges with users (who + what about, one line)
- Decisioni — choices made that may be referenced later
- Eventi — concrete things that happened (publications, deliveries, meetings)
- Pending — things you said you'd do that aren't done yet

Format: ``[DAILY] section: bullet text``. Example: ``[DAILY] Decisioni: passare a versione 1.1.0 della pipeline mepiu``.

DAILY vs MEMORY/USER/SOUL — write [DAILY] when a fact is *temporal*: it
belongs to a specific day and may matter only for a short window. Write
[MEMORY]/[USER]/[SOUL] when a fact is *atemporal*: a permanent trait,
preference, identity element, or long-lived project context. Same fact
should not appear in both. ``Recent Journal Notes`` shown above are the
existing entries — do not flag duplicates.

Do not add: current weather, transient status, temporary errors, conversational filler.

[SKIP] if nothing needs updating.
