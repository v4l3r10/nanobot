"""Per-vault asyncio locks so the wiki_note tool and Dream-Lint never
write the same user's vault concurrently. asyncio (not threading) —
Dream and the agent loop share one event loop."""
from __future__ import annotations

import asyncio
from weakref import WeakValueDictionary

_locks: "WeakValueDictionary[str, asyncio.Lock]" = WeakValueDictionary()


def get_vault_lock(slug: str) -> asyncio.Lock:
    lock = _locks.get(slug)
    if lock is None:
        lock = asyncio.Lock()
        _locks[slug] = lock
    return lock
