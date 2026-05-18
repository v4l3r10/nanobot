from pathlib import Path
from nanobot.utils.atomic import atomic_write_text

def test_atomic_write_creates_file_and_parents(tmp_path):
    target = tmp_path / "a" / "b" / "note.md"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"

def test_atomic_write_overwrites_without_partial_state(tmp_path):
    target = tmp_path / "note.md"
    atomic_write_text(target, "v1")
    atomic_write_text(target, "v2-longer-content")
    assert target.read_text(encoding="utf-8") == "v2-longer-content"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]
