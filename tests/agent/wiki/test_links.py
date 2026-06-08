"""Tests for body-derived wikilink extraction/resolution (links.py)."""
from __future__ import annotations

from nanobot.agent.wiki.links import (
    build_adjacency,
    build_page_index,
    parse_wikilinks,
    resolve_links,
)
from nanobot.agent.wiki.page import Page


def _page(last_touched="2026-05-20", links_out=None):
    return Page(
        type="people", title="T", status="hot",
        created="2020-01-01", updated=last_touched, last_touched=last_touched,
        links_out=list(links_out or []),
        body="b\n",
    )


# --- parse_wikilinks (pure) ---

def test_parse_extracts_targets_in_order():
    body = "See [[people/alice]] and [[projects/pay]]."
    assert parse_wikilinks(body) == ["people/alice", "projects/pay"]


def test_parse_strips_alias_and_md_suffix():
    body = "[[people/alice|Alice]] and [[projects/pay.md]]"
    assert parse_wikilinks(body) == ["people/alice", "projects/pay"]


def test_parse_dedups_preserving_order():
    body = "[[a/x]] [[a/x]] [[b/y]]"
    assert parse_wikilinks(body) == ["a/x", "b/y"]


def test_parse_ignores_malformed_and_empty():
    body = "[[]] [[ ]] [single] [[good/one]] [[no close"
    assert parse_wikilinks(body) == ["good/one"]


# --- resolve_links ---

def test_resolve_keeps_canonical_even_if_target_absent():
    # forward link: target may not exist yet; _broken_links audits it
    assert resolve_links(["projects/pay"], {}) == ["projects/pay"]


def test_resolve_rejects_unsafe_canonical():
    bad = ["../etc/hosts", "people/_index", ".hidden/x", "a/b/c", "people/SCHEMA"]
    assert resolve_links(bad, {}) == []


def test_resolve_drops_self_reference():
    assert resolve_links(["people/alice"], {}, owner_ref="people/alice") == []


def test_resolve_bare_slug_against_index():
    index = build_page_index([("people/alice.md", _page())])
    assert resolve_links(["alice"], index) == ["people/alice"]


def test_resolve_bare_slug_zero_match_omitted():
    assert resolve_links(["ghost"], {}) == []


def test_resolve_bare_slug_ambiguous_prefers_most_recent_then_folder():
    index = build_page_index([
        ("people/x.md", _page(last_touched="2026-01-01")),
        ("projects/x.md", _page(last_touched="2026-05-01")),
    ])
    # projects/x is more recently touched -> wins
    assert resolve_links(["x"], index) == ["projects/x"]


def test_resolve_bare_slug_ambiguous_same_date_folder_ascending():
    index = build_page_index([
        ("zeta/x.md", _page(last_touched="2026-05-01")),
        ("alpha/x.md", _page(last_touched="2026-05-01")),
    ])
    assert resolve_links(["x"], index) == ["alpha/x"]


def test_resolve_output_sorted_and_deduped():
    index = build_page_index([("people/alice.md", _page())])
    out = resolve_links(["projects/pay", "people/alice", "alice", "projects/pay"], index)
    assert out == ["people/alice", "projects/pay"]


def test_resolve_bare_slug_resolves_to_cold_target():
    # build_page_index is fed cold pages too by the caller; a .cold/ relpath
    # still yields a canonical folder/slug (the cold component is stripped by
    # the caller, so here we just confirm normal folder/slug resolution).
    index = build_page_index([("concepts/note.md", _page())])
    assert resolve_links(["note"], index) == ["concepts/note"]


# --- build_adjacency ---

def test_adjacency_undirected_out_and_backlink():
    pages = [
        ("people/alice.md", _page(links_out=["projects/pay"])),
        ("projects/pay.md", _page(links_out=[])),
    ]
    adj = build_adjacency(pages)
    assert adj["people/alice.md"] == {"projects/pay.md"}
    assert adj["projects/pay.md"] == {"people/alice.md"}  # backlink emerges


def test_adjacency_resolves_cold_target_by_canonical_ref():
    pages = [
        ("people/alice.md", _page(links_out=["projects/pay"])),
        (".cold/projects/pay.md", _page(links_out=[])),
    ]
    adj = build_adjacency(pages)
    assert adj["people/alice.md"] == {".cold/projects/pay.md"}
    assert adj[".cold/projects/pay.md"] == {"people/alice.md"}


def test_adjacency_prefers_hot_relpath_over_cold_at_same_ref():
    # A hot copy AND a cold copy of the same canonical ref coexist; a third page
    # links to that ref. The edge MUST resolve to the hot relpath — '.' < letters
    # would otherwise make ".cold/projects/pay.md" win the relpath sort.
    pages = [
        ("people/alice.md", _page(links_out=["projects/pay"])),
        ("projects/pay.md", _page(links_out=[])),
        (".cold/projects/pay.md", _page(links_out=[])),
    ]
    adj = build_adjacency(pages)
    assert adj["people/alice.md"] == {"projects/pay.md"}
    assert adj["projects/pay.md"] == {"people/alice.md"}
    assert ".cold/projects/pay.md" not in adj


def test_adjacency_skips_unresolvable_ref():
    pages = [("people/alice.md", _page(links_out=["projects/ghost"]))]
    adj = build_adjacency(pages)
    assert adj.get("people/alice.md", set()) == set()


def test_adjacency_drops_self_reference():
    pages = [("people/alice.md", _page(links_out=["people/alice"]))]
    adj = build_adjacency(pages)
    assert adj.get("people/alice.md", set()) == set()


def test_adjacency_empty_when_no_links():
    pages = [("people/alice.md", _page()), ("people/bob.md", _page())]
    assert build_adjacency(pages) == {}
