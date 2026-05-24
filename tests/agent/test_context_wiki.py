"""Tests for Task 6.1 — per-user vault MOC injection into the system prompt.

These exercise ``ContextBuilder.build_system_prompt`` with the ``wiki_enabled``
gate. The single most safety-critical invariant is the golden guarantee:
``wiki_enabled=False`` (and the param omitted) must produce a byte-identical
system prompt to the pre-6.1 code path. The remaining tests pin the
wiki-enabled behavior (vault MOC instead of global MEMORY.md, USER.md
de-duplication, smaller recency tail, safe fallbacks).
"""

from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.wiki.paths import vault_dir, vault_slug

# ---------------------------------------------------------------------------
# Helpers — mirror the existing test_context_builder.py pattern (real tmp
# workspace + the ContextBuilder-owned MemoryStore).
# ---------------------------------------------------------------------------

# Raw global memory/MEMORY.md content. Note MemoryStore.get_memory_context()
# wraps this in "## Long-term Memory\n{...}", so the rendered # Memory section
# is "# Memory\n\n## Long-term Memory\n{GLOBAL_MEMORY}" — keep this constant
# free of that heading to avoid a doubled-heading red herring.
GLOBAL_MEMORY = "GLOBAL-MEMORY-FACT: the sky is teal."
GLOBAL_MEMORY_RENDERED = f"# Memory\n\n## Long-term Memory\n{GLOBAL_MEMORY}"
VAULT_MOC = "# MOC\n\n## Recent\n- [[note-alpha]] vault-only durable fact\n"
ROOT_USER = "I am the GLOBAL user profile. Name: Root."
VAULT_USER = "I am the VAULT user profile. Name: Vaulted."


def _populate_workspace(
    tmp_path: Path,
    *,
    global_memory: str = GLOBAL_MEMORY,
    root_user: str | None = ROOT_USER,
    soul: str | None = "Be kind.",
    history_entries: int = 0,
) -> None:
    """Build a representative workspace: non-template global MEMORY.md, an
    optional root USER.md / SOUL.md, and some unprocessed history."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text(global_memory, encoding="utf-8")
    if root_user is not None:
        (tmp_path / "USER.md").write_text(root_user, encoding="utf-8")
    if soul is not None:
        (tmp_path / "SOUL.md").write_text(soul, encoding="utf-8")
    if history_entries:
        store = ContextBuilder(workspace=tmp_path).memory
        for i in range(history_entries):
            store.append_history(f"history entry {i}")


def _write_vault_moc(tmp_path: Path, session_key: str, content: str) -> Path:
    vdir = vault_dir(tmp_path, session_key)
    vdir.mkdir(parents=True, exist_ok=True)
    moc = vdir / "MEMORY.md"
    moc.write_text(content, encoding="utf-8")
    return moc


def _write_vault_user(tmp_path: Path, session_key: str, content: str) -> Path:
    vdir = vault_dir(tmp_path, session_key)
    vdir.mkdir(parents=True, exist_ok=True)
    u = vdir / "USER.md"
    u.write_text(content, encoding="utf-8")
    return u


def _memory_section(prompt: str) -> str | None:
    """Return the text of the ``# Memory`` part, or None if absent."""
    for part in prompt.split("\n\n---\n\n"):
        if part.startswith("# Memory\n\n"):
            return part
    return None


def _history_section(prompt: str) -> str | None:
    for part in prompt.split("\n\n---\n\n"):
        if part.startswith("# Recent History\n\n"):
            return part
    return None


# ---------------------------------------------------------------------------
# Golden guarantee — wiki OFF must be byte-identical to the original path
# ---------------------------------------------------------------------------


class TestWikiDisabledGolden:
    def test_wiki_disabled_prompt_is_golden(self, tmp_path):
        """wiki_enabled=False (and omitted) == the verbatim original path.

        Characterization: pins the concrete expected sections so any drift
        in the wiki-off path fails. Also asserts the explicit-False prompt
        and the param-omitted prompt are byte-identical to each other and
        to a wiki-off prompt built with an (ignored) session_key.
        """
        _populate_workspace(tmp_path, history_entries=5)

        omitted = ContextBuilder(workspace=tmp_path)
        explicit = ContextBuilder(workspace=tmp_path, wiki_enabled=False)

        p_omitted = omitted.build_system_prompt()
        p_explicit_false = explicit.build_system_prompt()
        # A session_key passed while wiki is OFF must be ignored entirely.
        p_false_with_key = explicit.build_system_prompt(session_key="unified:default")

        assert p_omitted == p_explicit_false == p_false_with_key

        # Concrete expected sections (zero-regression guard).
        mem = _memory_section(p_omitted)
        assert mem is not None
        assert mem == GLOBAL_MEMORY_RENDERED
        assert "GLOBAL-MEMORY-FACT" in mem
        # USER.md handled exactly as before — via the bootstrap block.
        assert "## USER.md" in p_omitted
        assert ROOT_USER in p_omitted
        assert "## SOUL.md" in p_omitted
        # History tail uses 50 (all 5 entries present here).
        hist = _history_section(p_omitted)
        assert hist is not None
        assert hist.count("- ") == 5

    def test_wiki_disabled_history_tail_uses_50(self, tmp_path):
        _populate_workspace(tmp_path, history_entries=70)
        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=False)
        prompt = builder.build_system_prompt()
        hist = _history_section(prompt)
        assert hist is not None
        lines = [ln for ln in hist.splitlines() if ln.startswith("- ")]
        assert len(lines) == ContextBuilder._MAX_RECENT_HISTORY == 50


# ---------------------------------------------------------------------------
# Wiki ENABLED behavior
# ---------------------------------------------------------------------------


class TestWikiEnabledMemory:
    def test_wiki_enabled_injects_vault_moc_not_global(self, tmp_path):
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)

        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")

        mem = _memory_section(prompt)
        assert mem is not None
        assert mem == f"# Memory\n\n{VAULT_MOC}"
        assert "vault-only durable fact" in mem
        assert "GLOBAL-MEMORY-FACT" not in prompt

    def test_wiki_enabled_empty_vault_skips_memory_section(self, tmp_path):
        _populate_workspace(tmp_path)
        # No vault MOC created at all.
        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")

        assert _memory_section(prompt) is None
        # The global MEMORY.md must NOT be used as a fallback.
        assert "GLOBAL-MEMORY-FACT" not in prompt

    def test_wiki_enabled_blank_vault_moc_skips_memory_section(self, tmp_path):
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", "   \n\n  \n")
        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")
        assert _memory_section(prompt) is None
        assert "GLOBAL-MEMORY-FACT" not in prompt

    def test_wiki_enabled_no_session_key_falls_back(self, tmp_path):
        """wiki_enabled=True but session_key=None → behave as wiki-off."""
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)

        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt()  # no session_key

        # Falls back to the global memory path; never crashes.
        mem = _memory_section(prompt)
        assert mem is not None
        assert mem == GLOBAL_MEMORY_RENDERED
        assert "vault-only durable fact" not in prompt

    def test_wiki_enabled_falls_back_equals_disabled(self, tmp_path):
        """No session_key with wiki on == the wiki-off prompt byte-for-byte."""
        _populate_workspace(tmp_path, history_entries=3)
        on = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        off = ContextBuilder(workspace=tmp_path, wiki_enabled=False)
        assert on.build_system_prompt() == off.build_system_prompt()


class TestWikiEnabledUserDedup:
    def test_wiki_enabled_user_md_not_double_injected(self, tmp_path):
        """With a global root USER.md and a vault USER.md, the wiki prompt
        carries the USER profile at most once and it is the VAULT one."""
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)
        _write_vault_user(tmp_path, "unified:default", VAULT_USER)

        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")

        # Global root USER.md must NOT appear.
        assert ROOT_USER not in prompt
        # Vault USER.md appears exactly once.
        assert prompt.count(VAULT_USER) == 1
        # SOUL.md is untouched.
        assert "## SOUL.md" in prompt
        assert "Be kind." in prompt

    def test_wiki_enabled_no_vault_user_omits_global(self, tmp_path):
        """If no vault USER.md exists, the global root USER.md is NOT
        injected either (no stale double)."""
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)
        # No vault USER.md.
        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")

        assert ROOT_USER not in prompt
        assert "## USER.md" not in prompt
        # Other bootstrap files unaffected.
        assert "## SOUL.md" in prompt


class TestWikiEnabledHistoryTail:
    def test_wiki_enabled_history_tail_capped_small(self, tmp_path):
        _populate_workspace(tmp_path, history_entries=25)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)

        on = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        off = ContextBuilder(workspace=tmp_path, wiki_enabled=False)

        on_hist = _history_section(on.build_system_prompt(session_key="unified:default"))
        off_hist = _history_section(off.build_system_prompt())

        assert on_hist is not None
        assert off_hist is not None
        on_lines = [ln for ln in on_hist.splitlines() if ln.startswith("- ")]
        off_lines = [ln for ln in off_hist.splitlines() if ln.startswith("- ")]
        assert len(on_lines) == ContextBuilder._MAX_RECENT_HISTORY_WIKI == 10
        assert len(off_lines) == 25  # all entries, < 50 cap


# ---------------------------------------------------------------------------
# Vault path resolution
# ---------------------------------------------------------------------------


class TestWikiSectionSizeCap:
    """M1 — the vault MOC / USER reads are deterministically size-capped so a
    corrupted / pre-Lint / hand-edited vault cannot blow up the hot-path
    prompt on every turn. ``truncate_text`` semantics: when over the cap the
    section content is ``text[:cap] + "\\n... (truncated)"``.
    """

    _SUFFIX = "\n... (truncated)"

    def test_wiki_moc_is_size_capped(self, tmp_path):
        cap = ContextBuilder._MAX_MEMORY_CHARS
        oversized = "M" * (cap * 3)
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", oversized)

        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")

        mem = _memory_section(prompt)
        assert mem is not None
        body = mem[len("# Memory\n\n"):]
        # Capped to the prefix + the stable truncation suffix, deterministic.
        assert body == oversized[:cap] + self._SUFFIX
        assert len(body) == cap + len(self._SUFFIX)
        # Far below the raw 3x size that would otherwise be injected verbatim.
        assert len(body) < len(oversized)

    def test_wiki_user_is_size_capped(self, tmp_path):
        cap = ContextBuilder._MAX_MEMORY_CHARS
        oversized = "U" * (cap * 3)
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)
        _write_vault_user(tmp_path, "unified:default", oversized)

        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")

        user_part = None
        for part in prompt.split("\n\n---\n\n"):
            if part.startswith("## USER.md\n\n"):
                user_part = part
                break
        assert user_part is not None
        body = user_part[len("## USER.md\n\n"):]
        assert body == oversized[:cap] + self._SUFFIX
        assert len(body) == cap + len(self._SUFFIX)

    def test_wiki_under_cap_moc_is_verbatim(self, tmp_path):
        """A normal-sized MOC is injected verbatim (cap is a ceiling only)."""
        _populate_workspace(tmp_path)
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)
        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="unified:default")
        mem = _memory_section(prompt)
        assert mem == f"# Memory\n\n{VAULT_MOC}"


class TestWikiEmptyStringSessionKey:
    """M3 — the read block and the ``wiki_active`` branch must share ONE
    predicate so a falsy-but-not-None key (``""``) can never make them
    disagree (read skipped, branch active ⇒ user loses BOTH memory and
    profile). ``session_key=""`` must behave exactly like the safe
    ``session_key=None`` fallback (= wiki-off byte-for-byte)."""

    def test_empty_string_session_key_falls_back_safely(self, tmp_path):
        _populate_workspace(tmp_path, history_entries=3)
        # A vault exists for the unified slug, but the EMPTY key must not
        # resolve to it (and must not crash) — it falls back to wiki-off.
        _write_vault_moc(tmp_path, "unified:default", VAULT_MOC)

        on = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        off = ContextBuilder(workspace=tmp_path, wiki_enabled=False)

        p_empty = on.build_system_prompt(session_key="")
        p_none = on.build_system_prompt(session_key=None)
        p_off = off.build_system_prompt()

        # Empty key == None key == wiki-off, byte-for-byte.
        assert p_empty == p_none == p_off
        # Concretely: global memory present, global USER present, vault hidden.
        mem = _memory_section(p_empty)
        assert mem == GLOBAL_MEMORY_RENDERED
        assert "## USER.md" in p_empty
        assert ROOT_USER in p_empty
        assert "vault-only durable fact" not in p_empty


class TestUnifiedSessionVaultPath:
    def test_unified_session_uses_unified_default_vault(self, tmp_path):
        assert vault_slug("unified:default") == "unified_default"
        vdir = vault_dir(tmp_path, "unified:default")
        assert vdir == tmp_path / "memory" / "users" / "unified_default"

    def test_wiki_enabled_resolves_vault_via_slug(self, tmp_path):
        _populate_workspace(tmp_path)
        # Write the MOC at the slug-resolved location for a channel session.
        _write_vault_moc(tmp_path, "telegram:42", "# T-MOC\n\nvault for telegram")
        builder = ContextBuilder(workspace=tmp_path, wiki_enabled=True)
        prompt = builder.build_system_prompt(session_key="telegram:42")
        mem = _memory_section(prompt)
        assert mem is not None
        assert "vault for telegram" in mem


# ---------------------------------------------------------------------------
# Layer 2 — per-interlocutor sender card
# ---------------------------------------------------------------------------


def _seed_person(tmp_path, slug, sender_ids, summary, title="Eugenio"):
    """Write a hot people page with sender-card frontmatter into a vault."""
    from nanobot.agent.wiki.vault import Vault
    from nanobot.agent.wiki.page import Page, serialize_page
    from nanobot.utils.atomic import atomic_write_text

    v = Vault(tmp_path / "memory" / "users" / slug)
    v.ensure_initialized()
    p = Page(
        type="people", title=title, status="hot",
        created="2026-05-24", updated="2026-05-24", last_touched="2026-05-24",
        summary=summary, sender_ids=sender_ids, body="full bio here",
    )
    atomic_write_text(v.page_path("people", title.lower()), serialize_page(p))


class TestResolveSenderCard:
    def test_resolves_by_numeric_prefix(self, tmp_path):
        _seed_person(tmp_path, "unified_default",
                     ["telegram:136150230"], "giornalista, IT informale")
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        line = cb.resolve_sender_card(
            sender_id="136150230|eugenio_user", channel="telegram",
            vault_key="unified:default")
        assert line is not None
        assert "Eugenio" in line and "giornalista" in line

    def test_resolves_by_full_sender_id(self, tmp_path):
        _seed_person(tmp_path, "unified_default",
                     ["telegram:136150230|eugenio_user"], "giornalista")
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        line = cb.resolve_sender_card(
            sender_id="136150230|eugenio_user", channel="telegram",
            vault_key="unified:default")
        assert line is not None and "Eugenio" in line

    def test_no_binding_returns_none(self, tmp_path):
        _seed_person(tmp_path, "unified_default",
                     ["telegram:999"], "someone else")
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        assert cb.resolve_sender_card(
            sender_id="136150230|x", channel="telegram",
            vault_key="unified:default") is None

    def test_disabled_returns_none(self, tmp_path):
        _seed_person(tmp_path, "unified_default", ["telegram:1"], "x")
        cb = ContextBuilder(tmp_path, wiki_enabled=False)
        assert cb.resolve_sender_card("1|a", "telegram", "unified:default") is None

    def test_no_vault_key_returns_none(self, tmp_path):
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        assert cb.resolve_sender_card("1|a", "telegram", None) is None

    def test_page_without_summary_is_not_resolved(self, tmp_path):
        # Bound id but no summary -> nothing useful to inject -> None.
        _seed_person(tmp_path, "unified_default", ["telegram:55"], "")
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        assert cb.resolve_sender_card("55|x", "telegram", "unified:default") is None


class TestSenderCardInTail:
    def test_bound_sender_line_in_tail_not_prefix(self, tmp_path):
        _seed_person(tmp_path, "unified_default",
                     ["telegram:136150230"], "giornalista, IT informale")
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        msgs = cb.build_messages(
            history=[], current_message="hi", channel="telegram",
            chat_id="c1", sender_id="136150230|eugenio_user",
            memory_key="unified:default")
        system = msgs[0]["content"]
        tail = msgs[-1]["content"]
        assert "Eugenio" in tail and "giornalista" in tail   # tail carries it
        assert "Eugenio (giornalista" not in system          # NOT in cacheable prefix

    def test_group_prefix_identical_tail_differs(self, tmp_path):
        _seed_person(tmp_path, "unified_default",
                     ["telegram:136150230"], "giornalista, IT informale")
        _seed_person(tmp_path, "unified_default",
                     ["telegram:999"], "altro", title="Rocco")
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        a = cb.build_messages(history=[], current_message="m", channel="telegram",
                              chat_id="g", sender_id="136150230|e", memory_key="unified:default")
        b = cb.build_messages(history=[], current_message="m", channel="telegram",
                              chat_id="g", sender_id="999|r", memory_key="unified:default")
        assert a[0]["content"] == b[0]["content"]   # cacheable prefix identical
        assert a[-1]["content"] != b[-1]["content"] # tail differs per sender
        assert "Eugenio" in a[-1]["content"]
        assert "Rocco" in b[-1]["content"]

    def test_unbound_sender_tail_has_no_card_line(self, tmp_path):
        cb = ContextBuilder(tmp_path, wiki_enabled=True)
        msgs = cb.build_messages(history=[], current_message="hi", channel="telegram",
                                 chat_id="c1", sender_id="42|nobody", memory_key="unified:default")
        assert "Sender:" not in msgs[-1]["content"]  # opt-in by data presence

    def test_wiki_off_tail_has_no_card_line(self, tmp_path):
        _seed_person(tmp_path, "unified_default", ["telegram:7"], "x")
        cb = ContextBuilder(tmp_path, wiki_enabled=False)
        msgs = cb.build_messages(history=[], current_message="hi", channel="telegram",
                                 chat_id="c1", sender_id="7|a", memory_key="unified:default")
        assert "Sender:" not in msgs[-1]["content"]
