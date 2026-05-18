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


class TestWikiDreamConfig:
    def test_wiki_defaults_off(self):
        from nanobot.config.schema import DreamConfig
        c = DreamConfig()
        assert c.wiki_enabled is False
        assert c.lint_cadence_h is None

    def test_wiki_camelcase_json_loads(self):
        from nanobot.config.schema import DreamConfig
        c = DreamConfig.model_validate({"wikiEnabled": True, "lintCadenceH": 6})
        assert c.wiki_enabled is True
        assert c.lint_cadence_h == 6

    def test_wiki_enabled_round_trips(self):
        from nanobot.config.schema import DreamConfig
        c = DreamConfig(wiki_enabled=True)
        dumped = c.model_dump()
        assert dumped["wiki_enabled"] is True

    def test_lint_cadence_validation(self):
        import pytest
        from pydantic import ValidationError
        from nanobot.config.schema import DreamConfig
        with pytest.raises(ValidationError):
            DreamConfig(lint_cadence_h=0)   # ge=1
