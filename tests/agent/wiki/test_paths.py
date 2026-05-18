from pathlib import Path
from nanobot.agent.wiki.paths import vault_slug, vault_dir

def test_slug_from_session_key():
    assert vault_slug("telegram:12345678") == "telegram_12345678"

def test_slug_unified():
    assert vault_slug("unified:default") == "unified_default"

def test_slug_sanitizes_unsafe_chars():
    assert vault_slug('a/b:c*?') == "a_b_c__"

def test_vault_dir_layout(tmp_path):
    d = vault_dir(tmp_path, "telegram:1")
    assert d == tmp_path / "memory" / "users" / "telegram_1"
