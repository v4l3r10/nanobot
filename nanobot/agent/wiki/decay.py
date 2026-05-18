"""Pure hot/cold decay predicate for wiki pages.

The Lint engine (Task 4.3) uses :func:`should_cool` to decide whether a
``hot`` page has gone stale and should be moved to ``.cold/``. This module
is intentionally a single deterministic, side-effect-free function: no I/O,
no logging, no LLM, no mutation of ``page``. Keeping the decision pure is
what makes Lint testable and idempotent.

Boundary convention: a page cools only when its age is **strictly greater
than** the configured ``cold_after_days`` limit. A page whose
``last_touched`` is *exactly* ``cold_after_days`` ago does NOT cool yet --
it cools the following day. This convention is locked by
``test_exactly_at_limit_boundary``.
"""

from __future__ import annotations

import datetime as dt

from nanobot.agent.wiki.page import Page
from nanobot.agent.wiki.schema import Schema

__all__ = ["should_cool"]


def _parse_iso_date(value: str) -> dt.date | None:
    """Parse ``value`` as an ISO ``YYYY-MM-DD`` date.

    Returns ``None`` on any malformed/unparsable input instead of raising,
    so the caller can apply the safe "don't cool on bad data" default.
    """
    try:
        return dt.date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def should_cool(page: Page, schema: Schema, today: dt.date) -> bool:
    """Whether ``page`` should transition ``hot`` -> ``cold`` as of ``today``.

    Pure predicate. Rules are evaluated in this exact order:

    1. Already ``cold`` -> ``False`` (no-op; never re-decays).
    2. ``page.pinned is True`` -> ``False`` (pinned pages are immune).
    3. ``page.type`` not a known schema type -> ``False``. This gate is
       applied *before* consulting ``cold_after_days`` because
       :meth:`Schema.cold_after_days` returns ``None`` for BOTH an unknown
       type and a known never-decaying type (Task 1.3 review finding); we
       must not infer "unknown" from ``None``.
    4. ``cold_after_days`` is ``None`` -> ``False`` (type configured to
       never decay).
    5. ``page.last_touched`` is not a valid ISO date -> ``False`` (safe
       default: a corrupt date is a separate concern; Lint must not evict
       a page just because its date is unparsable).
    6. Otherwise cool iff ``(today - last_touched).days`` is **strictly
       greater than** the limit (see module docstring for the boundary
       convention).

    ``page`` is never mutated; the result depends only on the arguments.
    """
    if page.status == "cold":
        return False

    if page.pinned is True:
        return False

    if not schema.is_known_type(page.type):
        return False

    limit = schema.cold_after_days(page.type)
    if limit is None:
        return False

    last_touched = _parse_iso_date(page.last_touched)
    if last_touched is None:
        return False

    age_days = (today - last_touched).days
    return age_days > limit
