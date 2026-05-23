"""Tests for the Dream class — two-phase memory consolidation via AgentRunner."""

import json

import pytest

from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.memory import Dream, MemoryStore
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.utils.gitstore import LineAge


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path)
    s.write_soul("# Soul\n- Helpful")
    s.write_user("# User\n- Developer")
    s.write_memory("# Memory\n- Project X active")
    return s


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def dream(store, mock_provider, mock_runner):
    d = Dream(store=store, provider=mock_provider, model="test-model", max_batch_size=5)
    d._runner = mock_runner
    return d


def _make_run_result(
    stop_reason="completed",
    final_content=None,
    tool_events=None,
    usage=None,
):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


class TestDreamRun:
    async def test_noop_when_no_unprocessed_history(self, dream, mock_provider, mock_runner, store):
        """Dream should not call LLM when there's nothing to process."""
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()

    async def test_calls_runner_for_unprocessed_entries(self, dream, mock_provider, mock_runner, store):
        """Dream should call AgentRunner when there are unprocessed history entries."""
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == 10
        assert spec.fail_on_tool_error is False

    async def test_advances_dream_cursor(self, dream, mock_provider, mock_runner, store):
        """Dream should advance the cursor after processing."""
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_processed_history(self, dream, mock_provider, mock_runner, store):
        """Dream should compact history after processing."""
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        # After Dream, cursor is advanced and 3, compact keeps last max_history_entries
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_skill_phase_uses_builtin_skill_creator_path(self, dream, mock_provider, mock_runner, store):
        """Dream should point skill creation guidance at the builtin skill-creator template."""
        store.append_history("Repeated workflow one")
        store.append_history("Repeated workflow two")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKILL] test-skill: test description")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        spec = mock_runner.run.call_args[0][0]
        system_prompt = spec.initial_messages[0]["content"]
        expected = str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        assert expected in system_prompt

    async def test_skill_write_tool_accepts_workspace_relative_skill_path(self, dream, store):
        """Dream skill creation should allow skills/<name>/SKILL.md relative to workspace root."""
        write_tool = dream._tools.get("write_file")
        assert write_tool is not None

        result = await write_tool.execute(
            path="skills/test-skill/SKILL.md",
            content="---\nname: test-skill\ndescription: Test\n---\n",
        )

        assert "Successfully wrote" in result
        assert (store.workspace / "skills" / "test-skill" / "SKILL.md").exists()

    async def test_phase1_prompt_includes_line_age_annotations(self, dream, mock_provider, mock_runner, store):
        """Phase 1 prompt should have per-line age suffixes in MEMORY.md when git is available."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        # Init git so line_ages works
        store.git.init()
        store.git.auto_commit("initial memory state")

        await dream.run()

        # The MEMORY.md section should not crash and should contain the memory content
        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_annotates_only_memory_not_soul_or_user(self, dream, mock_provider, mock_runner, store):
        """SOUL.md and USER.md should never have age annotations — they are permanent."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        store.git.init()
        store.git.auto_commit("initial state")

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        # The ← suffix should only appear in MEMORY.md section
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        soul_section = user_msg.split("## Current SOUL.md")[1].split("## Current USER.md")[0]
        user_section = user_msg.split("## Current USER.md")[1]
        # SOUL and USER should not contain age arrows
        assert "\u2190" not in soul_section
        assert "\u2190" not in user_section

    async def test_phase1_prompt_works_without_git(self, dream, mock_provider, mock_runner, store):
        """Phase 1 should work fine even if git is not initialized (no age annotations)."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        # Should still succeed — just without age annotations
        mock_provider.chat_with_retry.assert_called_once()
        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_prompt_carries_age_suffix_for_stale_lines(
        self, dream, mock_provider, mock_runner, store,
    ):
        """End-to-end: ages >14d must appear verbatim in the LLM prompt, ages ≤14d must not."""
        # MEMORY.md fixture has 2 non-blank lines ("# Memory" and "- Project X active").
        # Inject four ages to cover threshold boundaries: >14 suffix, ==14 no suffix, <14 no suffix.
        store.write_memory("# Memory\n- Project X active\n- fresh item\n- edge case line")
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        fake_ages = [
            LineAge(age_days=30),   # "# Memory"        → should get ← 30d
            LineAge(age_days=20),   # "- Project X..."  → should get ← 20d
            LineAge(age_days=14),   # "- fresh item"    → ==14, threshold is strictly >14, no suffix
            LineAge(age_days=5),    # "- edge case..."  → no suffix
        ]
        with patch.object(store.git, "line_ages", return_value=fake_ages):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190 30d" in memory_section
        assert "\u2190 20d" in memory_section
        assert "\u2190 14d" not in memory_section
        assert "\u2190 5d" not in memory_section

    async def test_phase1_skips_annotation_when_disabled(
        self, dream, mock_provider, mock_runner, store,
    ):
        """`annotate_line_ages=False` must bypass the git lookup entirely and keep MEMORY.md raw."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        dream.annotate_line_ages = False
        # line_ages must be bypassed entirely — verify with a spy rather than a
        # raising side_effect, because _annotate_with_ages catches Exception
        # (which swallows AssertionError) and would hide an accidental call.
        with patch.object(store.git, "line_ages") as mock_line_ages:
            await dream.run()
            mock_line_ages.assert_not_called()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "\u2190" not in user_msg

    async def test_phase1_skips_annotation_on_line_ages_length_mismatch(
        self, dream, mock_provider, mock_runner, store,
    ):
        """If ages length != lines length (dirty working tree), skip annotation instead of mis-tagging."""
        # MEMORY.md has 2 non-blank lines but we hand back only 1 age → mismatch.
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch.object(store.git, "line_ages", return_value=[LineAge(age_days=999)]):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        # No age arrow at all — we refused to annotate rather than tag the wrong line.
        assert "\u2190" not in memory_section

    async def test_phase1_prompt_uses_threshold_from_template_var(
        self, dream, mock_provider, mock_runner, store,
    ):
        """System prompt should reference the stale-threshold constant, not a hardcoded 14."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        system_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][0]["content"]
        # The template renders with stale_threshold_days=14 → LLM must see "N>14"
        assert "N>14" in system_msg


class TestDreamPromptCaps:
    """Dream's Phase 1/2 prompt must not be poisoned by a legacy oversized
    history entry or a runaway MEMORY.md. Without caps, a single pre-#3412
    raw_archive dump in history.jsonl would make every subsequent Dream run
    exceed the context window and silently advance the cursor past real work.
    """

    async def test_phase1_caps_huge_memory_file(
        self, dream, mock_provider, mock_runner, store,
    ):
        """A MEMORY.md much larger than _MEMORY_FILE_MAX_CHARS must be truncated
        in the prompt preview (full content is still reachable via read_file)."""
        store.write_memory("M" * (dream._MEMORY_FILE_MAX_CHARS * 5))
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert len(memory_section) < dream._MEMORY_FILE_MAX_CHARS + 500

    async def test_phase1_caps_huge_history_entry(
        self, dream, mock_provider, mock_runner, store,
    ):
        """A legacy oversized history entry (e.g. pre-#3412 raw_archive dump)
        must not explode the Phase 1 prompt — each entry is capped in the
        preview, even though the JSONL record itself stays full-size."""
        # Bypass the append_history cap by writing directly, simulating a
        # record that was written by an older nanobot build before any caps.
        store.history_file.write_text(
            json.dumps({
                "cursor": 1,
                "timestamp": "2026-04-01 10:00",
                "content": "H" * (dream._HISTORY_ENTRY_PREVIEW_MAX_CHARS * 8),
            }) + "\n",
            encoding="utf-8",
        )
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        history_section = user_msg.split("## Conversation History\n")[1].split("\n\n## Current Date")[0]
        assert len(history_section) < dream._HISTORY_ENTRY_PREVIEW_MAX_CHARS + 500


class TestDreamConfigurableCaps:
    """The four prompt-preview caps (memory/soul/user/history) can be overridden
    at __init__ so DreamConfig values from config.json take effect. The class
    constants remain as backward-compatible defaults for callers that don't
    pass them explicitly."""

    def test_init_defaults_match_class_constants(self, store, mock_provider):
        d = Dream(store=store, provider=mock_provider, model="m")

        assert d.memory_file_max_chars == Dream._MEMORY_FILE_MAX_CHARS
        assert d.soul_file_max_chars == Dream._SOUL_FILE_MAX_CHARS
        assert d.user_file_max_chars == Dream._USER_FILE_MAX_CHARS
        assert (
            d.history_entry_preview_max_chars == Dream._HISTORY_ENTRY_PREVIEW_MAX_CHARS
        )

    def test_init_accepts_custom_caps(self, store, mock_provider):
        d = Dream(
            store=store,
            provider=mock_provider,
            model="m",
            memory_file_max_chars=50_000,
            soul_file_max_chars=8_000,
            user_file_max_chars=8_000,
            history_entry_preview_max_chars=2_000,
            max_tool_result_chars=64_000,
        )

        assert d.memory_file_max_chars == 50_000
        assert d.soul_file_max_chars == 8_000
        assert d.user_file_max_chars == 8_000
        assert d.history_entry_preview_max_chars == 2_000
        assert d.max_tool_result_chars == 64_000

    def test_init_max_tool_result_chars_default(self, store, mock_provider):
        """Default mirrors AgentDefaults.max_tool_result_chars (16_000) so Dream's
        Phase 2 tool-result truncation matches the historical behavior when the
        config doesn't override it."""
        d = Dream(store=store, provider=mock_provider, model="m")

        assert d.max_tool_result_chars == 16_000

    async def test_phase1_respects_custom_memory_cap(
        self, store, mock_provider, mock_runner,
    ):
        """A small custom memory_file_max_chars truncates more aggressively."""
        d = Dream(
            store=store,
            provider=mock_provider,
            model="m",
            max_batch_size=5,
            memory_file_max_chars=1_000,
        )
        d._runner = mock_runner

        store.write_memory("M" * 10_000)
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert len(memory_section) < 1_500
        assert "(truncated)" in memory_section

    async def test_phase1_zero_cap_disables_truncation(
        self, store, mock_provider, mock_runner,
    ):
        """memory_file_max_chars=0 is the truncate_text sentinel for 'no cap'.
        The full memory file must reach the prompt preview unchanged."""
        d = Dream(
            store=store,
            provider=mock_provider,
            model="m",
            max_batch_size=5,
            memory_file_max_chars=0,
        )
        d._runner = mock_runner

        big_memory = "Mline\n" * 10_000  # ~60KB — would truncate under the 32K default
        store.write_memory(big_memory)
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "(truncated)" not in memory_section
        assert memory_section.count("Mline") == 10_000

    async def test_phase1_respects_custom_history_cap(
        self, store, mock_provider, mock_runner,
    ):
        """history_entry_preview_max_chars caps each entry independently."""
        d = Dream(
            store=store,
            provider=mock_provider,
            model="m",
            max_batch_size=5,
            history_entry_preview_max_chars=200,
        )
        d._runner = mock_runner

        store.history_file.write_text(
            json.dumps({
                "cursor": 1,
                "timestamp": "2026-04-01 10:00",
                "content": "H" * 5_000,
            }) + "\n",
            encoding="utf-8",
        )
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        history_section = user_msg.split("## Conversation History\n")[1].split("\n\n## Current Date")[0]
        assert len(history_section) < 400
        assert "(truncated)" in history_section


class TestDreamTimezone:
    """Phase 1 current_date and the journal note path both depend on the
    configured timezone. Without it, the day rolls over at UTC midnight even
    when the agent runs in Europe/Rome — wrong file, wrong date in prompt."""

    def test_today_returns_iso_date(self, store, mock_provider):
        import re

        d = Dream(store=store, provider=mock_provider, model="t")

        assert re.match(r"^\d{4}-\d{2}-\d{2}$", d._today())

    def test_today_with_invalid_timezone_falls_back_silently(self, store, mock_provider):
        """A typo in agents.defaults.timezone must not crash Dream."""
        import re

        d = Dream(store=store, provider=mock_provider, model="t", timezone="Not/Real")

        assert re.match(r"^\d{4}-\d{2}-\d{2}$", d._today())

    def test_today_uses_configured_timezone(self, store, mock_provider):
        """At the same UTC instant, UTC and Asia/Tokyo can show different dates."""
        from datetime import datetime as _dt, timezone as _tz
        from unittest.mock import patch

        fixed_utc = _dt(2026, 5, 8, 23, 30, tzinfo=_tz.utc)

        def fake_now(tz=None):
            return fixed_utc.astimezone(tz) if tz else fixed_utc.astimezone()

        with patch("nanobot.agent.memory.datetime") as mock_dt:
            mock_dt.now.side_effect = fake_now

            d_utc = Dream(store=store, provider=mock_provider, model="t", timezone="UTC")
            d_tokyo = Dream(store=store, provider=mock_provider, model="t", timezone="Asia/Tokyo")

            # 23:30 UTC on May 8 → 08:30 May 9 in Tokyo (UTC+9)
            assert d_utc._today() == "2026-05-08"
            assert d_tokyo._today() == "2026-05-09"

    async def test_run_uses_today_for_current_date_section(self, store, mock_provider, mock_runner):
        """Phase 1 prompt must show the TZ-aware date, not raw UTC."""
        from datetime import datetime as _dt, timezone as _tz
        from unittest.mock import patch

        fixed_utc = _dt(2026, 5, 8, 23, 30, tzinfo=_tz.utc)
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch("nanobot.agent.memory.datetime") as mock_dt:
            def fake_now(tz=None):
                return fixed_utc.astimezone(tz) if tz else fixed_utc.astimezone()
            mock_dt.now.side_effect = fake_now
            mock_dt.fromtimestamp = _dt.fromtimestamp  # used by line_ages

            d = Dream(
                store=store, provider=mock_provider, model="m",
                timezone="Asia/Tokyo",
            )
            d._runner = mock_runner
            await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        assert "## Current Date\n2026-05-09" in user_msg


class TestDreamJournalInjection:
    """Phase 1 must surface the most recent journal notes so Dream has temporal
    context when deciding what to add/remove/forward."""

    async def test_journal_section_omitted_when_disabled(self, store, mock_provider, mock_runner):
        store.append_history("anything")
        store.write_journal("2026-05-08", "# 2026-05-08\n- yesterday's stuff")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = Dream(
            store=store, provider=mock_provider, model="m",
            daily_notes_enabled=False,
        )
        d._runner = mock_runner
        await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        assert "Recent Journal Notes" not in user_msg

    async def test_journal_section_omitted_when_no_notes(self, dream, mock_provider, mock_runner, store):
        store.append_history("anything")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        assert "Recent Journal Notes" not in user_msg

    async def test_journal_section_includes_recent_notes_newest_first(self, dream, mock_provider, mock_runner, store):
        store.append_history("anything")
        store.write_journal("2026-05-06", "older content")
        store.write_journal("2026-05-08", "newest content")
        store.write_journal("2026-05-07", "middle content")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        # Default daily_notes_context_days=2 → only newest two
        await dream.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        assert "Recent Journal Notes" in user_msg
        idx_newest = user_msg.find("2026-05-08.md")
        idx_middle = user_msg.find("2026-05-07.md")
        idx_older = user_msg.find("2026-05-06.md")
        assert idx_newest != -1 and idx_middle != -1
        assert idx_newest < idx_middle  # newest first
        assert idx_older == -1  # context_days=2 caps the window

    async def test_journal_section_respects_context_days(self, store, mock_provider, mock_runner):
        store.append_history("anything")
        for day in (4, 5, 6, 7, 8):
            store.write_journal(f"2026-05-0{day}", f"day {day} content")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = Dream(
            store=store, provider=mock_provider, model="m",
            daily_notes_context_days=3,
        )
        d._runner = mock_runner
        await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        for keep in ("2026-05-08", "2026-05-07", "2026-05-06"):
            assert f"{keep}.md" in user_msg
        for drop in ("2026-05-05", "2026-05-04"):
            assert f"{drop}.md" not in user_msg

    async def test_journal_section_truncated_to_max_chars(self, store, mock_provider, mock_runner):
        store.append_history("anything")
        big = "X" * 5_000
        store.write_journal("2026-05-08", big)
        store.write_journal("2026-05-07", big)
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = Dream(
            store=store, provider=mock_provider, model="m",
            daily_notes_max_chars=500,
        )
        d._runner = mock_runner
        await d.run()

        user_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
        # Extract the journal section between its header and the next "## " header
        start = user_msg.index("## Recent Journal Notes")
        end = user_msg.index("## Current MEMORY.md", start)
        journal_block = user_msg[start:end]
        # Header line itself adds ~40 chars; bound the body within cap + small slack
        assert len(journal_block) < 700

    async def test_phase1_prompt_documents_daily_verb(self, dream, mock_provider, mock_runner, store):
        """The system prompt must teach the LLM the [DAILY] verb."""
        store.append_history("anything")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        system_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][0]["content"]
        assert "[DAILY]" in system_msg
        assert "Daily journal" in system_msg or "daily journal" in system_msg.lower()


class TestDreamPhase2DailyJournal:
    """Phase 2's system prompt must teach the LLM how to write/append the
    journal note for today, using edit_file with old_text='' for create."""

    async def test_phase2_prompt_includes_today_journal_path(self, dream, mock_provider, mock_runner, store):
        store.append_history("anything")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[DAILY] Eventi: shipped")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        phase2_system = mock_runner.run.call_args[0][0].initial_messages[0]["content"]
        # Path must follow memory/journal/YYYY-MM-DD.md, today's date.
        today = dream._today()
        assert f"memory/journal/{today}.md" in phase2_system

    async def test_phase2_prompt_documents_create_and_append_rules(self, dream, mock_provider, mock_runner, store):
        store.append_history("anything")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[DAILY] Eventi: shipped")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        phase2_system = mock_runner.run.call_args[0][0].initial_messages[0]["content"]
        # Create rule: edit_file with old_text=""
        assert "old_text=\"\"" in phase2_system or 'old_text=""' in phase2_system
        # Standard skeleton sections must be in the prompt so the LLM seeds
        # the file consistently across days.
        for section in ("Conversazioni", "Decisioni", "Eventi", "Pending"):
            assert f"## {section}" in phase2_system

    async def test_phase2_prompt_omits_daily_section_when_disabled(self, store, mock_provider, mock_runner):
        store.append_history("anything")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        d = Dream(
            store=store, provider=mock_provider, model="m",
            daily_notes_enabled=False,
        )
        d._runner = mock_runner
        await d.run()

        phase2_system = mock_runner.run.call_args[0][0].initial_messages[0]["content"]
        assert "Daily journal note" not in phase2_system
        assert "memory/journal/" not in phase2_system

    async def test_edit_file_can_create_journal_when_missing(self, dream, store):
        """Sanity: the existing edit_file tool already supports old_text=''
        create-on-empty inside workspace, so no new tool wiring is needed."""
        edit_tool = dream._tools.get("edit_file")
        assert edit_tool is not None

        target = "memory/journal/2026-05-08.md"
        result = await edit_tool.execute(
            path=target,
            old_text="",
            new_text="# 2026-05-08\n\n## Eventi\n",
        )

        assert "Successfully created" in result
        assert (store.workspace / target).exists()
        assert "# 2026-05-08" in (store.workspace / target).read_text(encoding="utf-8")


def test_dream_has_wiki_embedding_attrs_defaulting_off(dream):
    # Reuse the module's `dream` fixture, which builds:
    #   Dream(store=store, provider=mock_provider, model="test-model", max_batch_size=5)
    assert dream.wiki_embeddings is False
    assert dream.wiki_embedding_model == ""

