"""Crash-safe text writes: temp file in the same dir + fsync + os.replace.

Mirrors the existing atomic pattern in MemoryStore._write_entries
(agent/memory.py) so behaviour is consistent across the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
