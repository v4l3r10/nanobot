from nanobot.config.schema import DreamConfig


def test_dream_config_defaults_to_interval_hours() -> None:
    cfg = DreamConfig()

    assert cfg.interval_h == 2
    assert cfg.cron is None


def test_dream_config_builds_every_schedule_from_interval() -> None:
    cfg = DreamConfig(interval_h=3)

    schedule = cfg.build_schedule("UTC")

    assert schedule.kind == "every"
    assert schedule.every_ms == 3 * 3_600_000
    assert schedule.expr is None


def test_dream_config_honors_legacy_cron_override() -> None:
    cfg = DreamConfig.model_validate({"cron": "0 */4 * * *"})

    schedule = cfg.build_schedule("UTC")

    assert schedule.kind == "cron"
    assert schedule.expr == "0 */4 * * *"
    assert schedule.tz == "UTC"
    assert cfg.describe_schedule() == "cron 0 */4 * * * (legacy)"


def test_dream_config_dump_uses_interval_h_and_hides_legacy_cron() -> None:
    cfg = DreamConfig.model_validate({"intervalH": 5, "cron": "0 */4 * * *"})

    dumped = cfg.model_dump(by_alias=True)

    assert dumped["intervalH"] == 5
    assert "cron" not in dumped


def test_dream_config_uses_model_override_name_and_accepts_legacy_model() -> None:
    cfg = DreamConfig.model_validate({"model": "openrouter/sonnet"})

    dumped = cfg.model_dump(by_alias=True)

    assert cfg.model_override == "openrouter/sonnet"
    assert dumped["modelOverride"] == "openrouter/sonnet"
    assert "model" not in dumped


def test_dream_config_default_prompt_caps_match_dream_class_constants() -> None:
    """Defaults must match Dream's historical class constants for backward compatibility."""
    from nanobot.agent.memory import Dream

    cfg = DreamConfig()

    assert cfg.memory_file_max_chars == Dream._MEMORY_FILE_MAX_CHARS == 32_000
    assert cfg.soul_file_max_chars == Dream._SOUL_FILE_MAX_CHARS == 16_000
    assert cfg.user_file_max_chars == Dream._USER_FILE_MAX_CHARS == 16_000
    assert (
        cfg.history_entry_preview_max_chars
        == Dream._HISTORY_ENTRY_PREVIEW_MAX_CHARS
        == 4_000
    )


def test_dream_config_accepts_camel_case_prompt_caps() -> None:
    cfg = DreamConfig.model_validate({
        "memoryFileMaxChars": 50_000,
        "soulFileMaxChars": 8_000,
        "userFileMaxChars": 8_000,
        "historyEntryPreviewMaxChars": 2_000,
    })

    assert cfg.memory_file_max_chars == 50_000
    assert cfg.soul_file_max_chars == 8_000
    assert cfg.user_file_max_chars == 8_000
    assert cfg.history_entry_preview_max_chars == 2_000


def test_dream_config_accepts_snake_case_prompt_caps() -> None:
    """populate_by_name=True on Base lets snake_case keys also work."""
    cfg = DreamConfig.model_validate({
        "memory_file_max_chars": 64_000,
        "history_entry_preview_max_chars": 8_000,
    })

    assert cfg.memory_file_max_chars == 64_000
    assert cfg.history_entry_preview_max_chars == 8_000


def test_dream_config_zero_disables_cap() -> None:
    """A cap of 0 is the truncate_text sentinel for 'no truncation'."""
    cfg = DreamConfig(memory_file_max_chars=0, history_entry_preview_max_chars=0)

    assert cfg.memory_file_max_chars == 0
    assert cfg.history_entry_preview_max_chars == 0


def test_dream_config_rejects_negative_prompt_caps() -> None:
    """Negative caps are nonsensical; ge=0 enforces this."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DreamConfig(memory_file_max_chars=-1)


def test_dream_config_dump_uses_camel_case_for_prompt_caps() -> None:
    cfg = DreamConfig(memory_file_max_chars=24_000)

    dumped = cfg.model_dump(by_alias=True)

    assert dumped["memoryFileMaxChars"] == 24_000
    assert dumped["soulFileMaxChars"] == 16_000
    assert dumped["userFileMaxChars"] == 16_000
    assert dumped["historyEntryPreviewMaxChars"] == 4_000
