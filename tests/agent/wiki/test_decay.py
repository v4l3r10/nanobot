import datetime as dt

from nanobot.agent.wiki.decay import should_cool
from nanobot.agent.wiki.page import Page
from nanobot.agent.wiki.schema import load_schema

SCHEMA = load_schema('''```yaml
types:
  people:    { folder: people,    cold_after_days: 180 }
  decisions: { folder: decisions, cold_after_days: null }
required_frontmatter: [type, title, status, created, updated, last_touched]
moc_max_lines: 120
```''')

TODAY = dt.date(2026, 5, 18)


def _p(type="people", status="hot", last_touched="2026-05-18", pinned=None):
    return Page(type=type, title="T", status=status, created="2020-01-01",
                updated=last_touched, last_touched=last_touched,
                tags=[], links_out=[], pinned=pinned, body="b\n")


def test_pinned_never_cools():
    p = _p(last_touched="2000-01-01", pinned=True)
    assert should_cool(p, SCHEMA, TODAY) is False


def test_never_decaying_type_never_cools():
    p = _p(type="decisions", last_touched="2000-01-01")
    assert should_cool(p, SCHEMA, TODAY) is False


def test_stale_beyond_limit_cools():
    p = _p(last_touched="2025-10-30")  # ~200 days before 2026-05-18, limit 180
    assert should_cool(p, SCHEMA, TODAY) is True


def test_recent_within_limit_does_not_cool():
    p = _p(last_touched="2026-02-07")  # ~100 days, within 180
    assert should_cool(p, SCHEMA, TODAY) is False


def test_already_cold_is_noop():
    p = _p(status="cold", last_touched="2000-01-01")
    assert should_cool(p, SCHEMA, TODAY) is False


def test_unknown_type_does_not_cool():
    p = _p(type="aliens", last_touched="2000-01-01")
    assert should_cool(p, SCHEMA, TODAY) is False


def test_exactly_at_limit_boundary():
    # last_touched exactly `cold_after_days` ago: define & lock the boundary.
    p = _p(last_touched=(TODAY - dt.timedelta(days=180)).isoformat())
    # document the chosen convention in the impl; assert it here:
    assert should_cool(p, SCHEMA, TODAY) is False  # strictly GREATER THAN limit cools


def test_malformed_last_touched_does_not_cool():
    p = _p(last_touched="not-a-date")
    assert should_cool(p, SCHEMA, TODAY) is False  # safe default: don't cool on bad data
