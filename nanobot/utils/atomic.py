"""Crash-safe text writes: temp file in the same dir + fsync + os.replace.

Mirrors the existing atomic pattern in MemoryStore._write_entries
(agent/memory.py) so behaviour is consistent across the codebase:
fsync the temp file, os.replace, fsync the parent directory (skipped
on Windows where opening a directory raises PermissionError), and
unlink the temp file on any failure so no stale ``.tmp`` is left.
"""
from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

        # fsync the directory so the rename is durable.
        # On Windows, opening a directory with O_RDONLY raises
        # PermissionError — skip the dir sync there (NTFS
        # journals metadata synchronously).
        with suppress(PermissionError):
            fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
