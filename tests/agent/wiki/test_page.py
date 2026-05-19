import pytest
from nanobot.agent.wiki.page import Page, parse_page, serialize_page


def _page():
    return Page(
        type="people", title="Alice", status="hot",
        created="2026-05-18", updated="2026-05-18", last_touched="2026-05-18",
        tags=["team", "eng"], links_out=["projects/payment-svc"],
        pinned=False, body="Alice leads payments.\n",
    )


def test_round_trip():
    p = _page()
    text = serialize_page(p)
    assert text.startswith("---\n")
    p2 = parse_page(text)
    assert p2 == p


def test_parse_tolerates_missing_optional_pinned():
    p = _page(); p.pinned = None
    p2 = parse_page(serialize_page(p))
    assert p2.pinned is None


def test_parse_rejects_unknown_status():
    bad = serialize_page(_page()).replace("status: hot", "status: lukewarm")
    with pytest.raises(ValueError):
        parse_page(bad)
