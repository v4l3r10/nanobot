Update memory files based on the analysis below.
- [FILE] entries: add the described content to the appropriate file
- [FILE-REMOVE] entries: delete the corresponding content from memory files
- [SKILL] entries: create a new skill under skills/<name>/SKILL.md using write_file
{% if daily_notes_enabled %}- [DAILY] entries: append today's bullets to {{ journal_path }} (see "Daily journal note" rules below){% endif %}

## File paths (relative to workspace root)
- SOUL.md
- USER.md
- memory/MEMORY.md
- skills/<name>/SKILL.md (for [SKILL] entries only)
{% if daily_notes_enabled %}- {{ journal_path }} (today's journal note — for [DAILY] entries only){% endif %}

Do NOT guess paths.

## Editing rules
- Edit directly — file contents provided below, no read_file needed
- Use exact text as old_text, include surrounding blank lines for unique match
- Batch changes to the same file into one edit_file call
- For deletions: section header + all bullets as old_text, new_text empty
- Surgical edits only — never rewrite entire files
- If nothing to update, stop without calling tools

## Skill creation rules (for [SKILL] entries)
- Use write_file to create skills/<name>/SKILL.md
- Before writing, read_file `{{ skill_creator_path }}` for format reference (frontmatter structure, naming conventions, quality standards)
- **Dedup check**: read existing skills listed below to verify the new skill is not functionally redundant. Skip creation if an existing skill already covers the same workflow.
- Include YAML frontmatter with name and description fields
- Keep SKILL.md under 2000 words — concise and actionable
- Include: when to use, steps, output format, at least one example
- Do NOT overwrite existing skills — skip if the skill directory already exists
- Reference specific tools the agent has access to (read_file, write_file, exec, web_search, etc.)
- Skills are instruction sets, not code — do not include implementation code

{% if daily_notes_enabled %}## Daily journal note
Today's note path: `{{ journal_path }}`. Use these rules **only** for [DAILY] entries.

If the file does not exist yet (first Dream cycle of the day):
- Call `edit_file` with `old_text=""` and `new_text` set to a fresh skeleton:
  ```
  # {{ journal_path.split("/")[-1].rstrip(".md") }}

  ## Conversazioni

  ## Decisioni

  ## Eventi

  ## Pending
  ```
- Then add the [DAILY] bullets under their declared sections in a second
  `edit_file` call (one call per section is fine, batching is fine).

If the file already exists:
- Use `edit_file` to append each bullet under its section header.
- `old_text` should be the section header line plus the trailing blank
  line so the match is unique; `new_text` repeats them with the new
  bullet inserted before the blank line. Example for "Decisioni":
  - `old_text`: ``"## Decisioni\n\n"``
  - `new_text`: ``"## Decisioni\n- nuovo bullet qui\n\n"``
- Never overwrite existing bullets, only append.
- If a [DAILY] entry's section is unknown, default to "Eventi".

Surgical only — never rewrite the whole journal note. The file is
human-readable and other Dream cycles in the same day must be able to
add to it without conflict.

{% endif %}## Quality
- Every line must carry standalone value
- Concise bullets under clear headers
- When reducing (not deleting): keep essential facts, drop verbose details
- If uncertain whether to delete, keep but add "(verify currency)"
