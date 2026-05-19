from __future__ import annotations
from pathlib import Path
from nanobot.utils.helpers import safe_filename


def vault_slug(session_key: str) -> str:
    """Mirror SessionManager.safe_key: replace ':' then strip unsafe chars."""
    return safe_filename(session_key.replace(":", "_"))


def vault_dir(workspace: Path, session_key: str) -> Path:
    return Path(workspace) / "memory" / "users" / vault_slug(session_key)
