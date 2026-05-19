"""Tests for Task 2 — gated substitution of the `memory` always-skill body.

When the wiki read path is active for a call (``wiki_enabled`` AND a usable
``session_key``), the legacy ``memory`` always-skill body (grep
``memory/history.jsonl``, "Managed by Dream", "Do NOT edit ...") is stale and
misleading: the per-user vault MOC + ``wiki_note`` are the real mechanism. The
single place that knows the wiki is on (``ContextBuilder.build_system_prompt``)
substitutes ONLY the ``memory`` body with the rendered wiki template, keeping
the exact ``### Skill: <name>`` / ``\\n\\n---\\n\\n`` wrapper shape
``load_skills_for_context`` produces. The wiki-OFF path is byte-identical to
before (it still calls ``load_skills_for_context(always_skills)`` unchanged).

Wiring mirrors ``tests/agent/test_context_wiki.py``: a real tmp workspace, the
ContextBuilder-owned MemoryStore, a vault MOC written under the slug-resolved
vault dir, and ``build_system_prompt(session_key=...)``. Always-skills resolve
from the real bundled ``nanobot/skills`` dir (SkillsLoader's BUILTIN_SKILLS_DIR
is package-relative, not workspace-relative), so ``memory`` and ``my`` are
present regardless of the tmp workspace.
"""

from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.wiki.paths import vault_dir

# Mirror the test_context_wiki.py helpers.
GLOBAL_MEMORY = "GLOBAL-MEMORY-FACT: the sky is teal."
VAULT_MOC = "# MOC\n\n## Recent\n- [[note-alpha]] vault-only durable fact\n"


def _populate_workspace(tmp_path: Path) -> None:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text(GLOBAL_MEMORY, encoding="utf-8")
    (tmp_path / "USER.md").write_text("I am the root user.", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("Be kind.", encoding="utf-8")


def _write_vault_moc(tmp_path: Path, session_key: str, content: str) -> Path:
    vdir = vault_dir(tmp_path, session_key)
    vdir.mkdir(parents=True, exist_ok=True)
    moc = vdir / "MEMORY.md"
    moc.write_text(content, encoding="utf-8")
    return moc


def _build_prompt(tmp_path: Path, *, wiki_enabled: bool, session_key: str) -> str:
    _populate_workspace(tmp_path)
    if wiki_enabled:
        _write_vault_moc(tmp_path, session_key, VAULT_MOC)
    builder = ContextBuilder(workspace=tmp_path, wiki_enabled=wiki_enabled)
    return builder.build_system_prompt(session_key=session_key)


def _memory_skill_section(prompt: str) -> str:
    """Return the body of the ``### Skill: memory`` block.

    The legacy-absence assertions must be scoped to the skill body, NOT the
    whole prompt: the (out-of-scope, untouched) ``agent/identity.md`` template
    independently mentions ``memory/history.jsonl`` in its Workspace section,
    so a whole-prompt ``not in`` check would conflate the two. This feature
    only governs the ``memory`` always-skill body, so that is what we inspect.
    The ``# Active Skills`` block joins skills with ``\\n\\n---\\n\\n``; the
    memory skill body runs from ``### Skill: memory`` to the next ``---``
    separator (or end of the block).
    """
    marker = "### Skill: memory"
    start = prompt.index(marker)
    rest = prompt[start:]
    end = rest.find("\n\n---\n\n")
    return rest if end == -1 else rest[:end]


def test_wiki_off_memory_skill_is_legacy_verbatim(tmp_path):
    """wiki-off: the memory skill section equals the raw SKILL.md body
    (frontmatter stripped) — unchanged by this feature."""
    prompt = _build_prompt(tmp_path, wiki_enabled=False, session_key="telegram:1")
    mem_skill = _memory_skill_section(prompt)
    # Legacy markers present in the memory skill body, wiki directive absent.
    assert "memory/history.jsonl" in mem_skill
    assert "Managed by Dream" in mem_skill
    assert "Do NOT edit SOUL.md, USER.md, or MEMORY.md" in mem_skill
    assert "wiki_note" not in prompt  # no other always-skill mentions it
    # Body equals the raw SKILL.md with frontmatter stripped (verbatim path).
    raw = (Path(__file__).parents[2] / "nanobot" / "skills" / "memory" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    stripped = raw.split("---", 2)[2].strip()
    assert mem_skill == f"### Skill: memory\n\n{stripped}"
    # Wrapper shape preserved.
    assert "### Skill: memory" in prompt
    assert "# Active Skills" in prompt


def test_wiki_on_memory_skill_is_wiki_aware(tmp_path):
    prompt = _build_prompt(tmp_path, wiki_enabled=True, session_key="telegram:1")
    mem_skill = _memory_skill_section(prompt)
    assert "wiki_note" in prompt
    assert "search the wiki before" in prompt.lower() or "before answering" in prompt.lower()
    # Legacy stale guidance is NOT injected into the memory skill body when
    # wiki active (carry-forward 1: ALL legacy markers absent — exact
    # substrings from skills/memory/SKILL.md; scoped to the skill body since
    # the out-of-scope identity template independently names history.jsonl).
    assert "memory/history.jsonl" not in mem_skill
    assert "Managed by Dream" not in mem_skill
    assert "Do NOT edit SOUL.md, USER.md, or MEMORY.md" not in mem_skill
    # Wrapper shape preserved.
    assert "### Skill: memory" in prompt
    assert "# Active Skills" in prompt


def test_wiki_on_other_always_skills_unchanged(tmp_path):
    """Substitution is scoped to `memory` only — other always-skills
    (e.g. `my`) still render from their SKILL.md verbatim."""
    prompt = _build_prompt(tmp_path, wiki_enabled=True, session_key="telegram:1")
    assert "### Skill: my" in prompt
    assert "Self-Awareness" in prompt  # from my/SKILL.md body


def test_wiki_on_memory_body_renders_after_moc_section(tmp_path):
    """Carry-forward 2 (positional): the wiki memory skill body renders AFTER
    the injected ``# Memory`` MOC section, so the template's reference to "the
    ``# Memory`` section above" is true in the assembled prompt."""
    prompt = _build_prompt(tmp_path, wiki_enabled=True, session_key="telegram:1")
    moc_idx = prompt.index("# Memory\n\n")
    skill_idx = prompt.index("### Skill: memory")
    assert moc_idx < skill_idx
    # The MOC section is the vault MOC (not the global MEMORY.md fallback).
    assert "vault-only durable fact" in prompt
    assert "GLOBAL-MEMORY-FACT" not in prompt


def _active_skills_section(prompt: str) -> str:
    """Return the EXACT ``# Active Skills`` part of the assembled prompt.

    ``build_system_prompt`` joins top-level parts with ``\\n\\n---\\n\\n`` and
    the part right after ``# Active Skills`` is the ``# Skills`` summary
    (``agent/skills_section.md``). The Active-Skills content itself joins
    individual skills with the same ``\\n\\n---\\n\\n``, so the section runs
    from the ``# Active Skills`` header up to the part boundary that starts the
    next top-level section (``\\n\\n---\\n\\n# Skills``), or end of prompt.
    """
    start = prompt.index("# Active Skills")
    rest = prompt[start:]
    boundary = rest.find("\n\n---\n\n# Skills")
    return rest if boundary == -1 else rest[:boundary]


def test_wiki_off_path_is_byte_identical_to_no_substitution(tmp_path):
    """Task 3 wiki-OFF byte-identity golden: with wiki off, the *entire*
    ``# Active Skills`` section of the full system prompt is byte-identical to
    ``# Active Skills\\n\\n`` + ``load_skills_for_context(get_always_skills())``
    rebuilt from the unmodified ``SKILL.md`` files — proving the Task 2
    substitution branch (``if wiki_active and "memory" in always_skills``) is
    provably NOT taken and the non-wiki code path is unchanged by this feature.
    Deterministic; no mocks beyond the existing harness; exact equality of the
    isolated section slice (not a weaker containment check)."""
    _populate_workspace(tmp_path)
    builder = ContextBuilder(workspace=tmp_path, wiki_enabled=False)
    prompt = builder.build_system_prompt(session_key="telegram:1")

    always = builder.skills.get_always_skills()
    assert "memory" in always  # sanity: resolves under the test harness
    expected = builder.skills.load_skills_for_context(always)
    assert _active_skills_section(prompt) == f"# Active Skills\n\n{expected}"
