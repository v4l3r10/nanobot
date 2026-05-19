"""Process-global "this vault received a wiki_note write this turn" signal.

Single process, single asyncio event loop: plain ``set`` add/remove are
atomic between awaits, so no lock is needed. This is ONLY an optimization
(skip the post-turn cheap MOC rebuild for vaults nobody wrote to this
turn); correctness never depends on it -- a missed mark merely defers the
MOC refresh to the next write or the next Dream cycle.
"""

_dirty: set[str] = set()


def mark_vault_dirty(slug: str) -> None:
    """Record that ``slug``'s vault got a wiki write (called by wiki_note)."""
    _dirty.add(slug)


def take_dirty(slug: str) -> bool:
    """Return True and clear iff ``slug`` was marked since the last take."""
    try:
        _dirty.remove(slug)
        return True
    except KeyError:
        return False
