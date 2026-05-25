"""Body-derived wikilink graph for the wiki tree.

Pages reference each other with Obsidian-style ``[[folder/slug]]`` links in
their Markdown body. The body is authoritative; a page's ``links_out``
frontmatter is a *derived projection* holding only resolved, canonical
``folder/slug`` refs (the exact shape ``lint._broken_links`` audits).

``parse_wikilinks`` is pure (body in, raw targets out). ``resolve_links`` turns
raw targets into canonical refs: canonical refs are validated for shape/safety
and kept even if the target does not exist yet (a forward link); bare ``slug``
refs are resolved against a page index (``build_page_index``), the
most-recently-touched page winning an ambiguous slug. An EMPTY index is valid:
bare slugs then resolve to nothing and only canonical refs survive — that is
the eager write-path mode.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from nanobot.agent.wiki.page import Page
from nanobot.agent.wiki.vault import _COLD_COMPONENT, _NON_PAGE_NAMES

__all__ = ["build_page_index", "parse_wikilinks", "resolve_links"]

# A wikilink ``[[ ... ]]`` with no nested brackets inside the target.
_WIKILINK = re.compile(r"\[\[([^\[\]]+)\]\]")

# Reserved page stems (e.g. SCHEMA, _index) from the authoritative non-page
# set — never link to a structural file.
_RESERVED_STEMS = {n[: -len(".md")] for n in _NON_PAGE_NAMES if n.endswith(".md")}


def parse_wikilinks(body: str) -> list[str]:
    """Extract raw ``[[...]]`` targets from a body (pure, order-preserving).

    Per match: take the text before any ``|`` (alias), strip, drop a single
    trailing ``.md`` (often copied from a search result path). Empty targets
    dropped; duplicates removed preserving first-seen order. No validation or
    resolution here — returns ``folder/slug`` or bare ``slug`` strings.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _WIKILINK.finditer(body):
        target = m.group(1).split("|", 1)[0].strip()
        if target.endswith(".md"):
            target = target[: -len(".md")].strip()
        if not target or target in seen:
            continue
        seen.add(target)
        out.append(target)
    return out


def build_page_index(
    pages: Iterable[tuple[str, Page]],
) -> dict[str, list[tuple[str, str, str]]]:
    """Map ``slug.lower()`` -> list of ``(ref, last_touched, folder)`` candidates.

    ``ref`` is the canonical ``folder/slug`` (relpath minus ``.md``). Built from
    ``(relpath, Page)`` pairs. The caller passes plain ``folder/slug`` relpaths
    (a cold page is exposed by its canonical relpath, not its ``.cold/`` path).
    """
    index: dict[str, list[tuple[str, str, str]]] = {}
    for relpath, page in pages:
        if not relpath.endswith(".md"):
            continue
        ref = relpath[: -len(".md")]
        if "/" not in ref:
            continue
        folder, slug = ref.rsplit("/", 1)
        index.setdefault(slug.lower(), []).append((ref, page.last_touched, folder))
    return index


def _is_safe_ref(folder: str, slug: str) -> bool:
    """Whether ``folder/slug`` is a safe, non-structural content-page ref."""
    if not folder or not slug:
        return False
    for seg in (folder, slug):
        if seg != seg.strip() or seg.startswith(".") or seg == "..":
            return False
        if any(ord(ch) < 0x20 for ch in seg):
            return False
    if folder == _COLD_COMPONENT or slug == _COLD_COMPONENT:
        return False
    if f"{slug}.md" in _NON_PAGE_NAMES or slug in _RESERVED_STEMS:
        return False
    return True


def _resolve_one(
    target: str,
    index: dict[str, list[tuple[str, str, str]]],
    owner_ref: str | None,
) -> str | None:
    if "/" in target:
        parts = target.split("/")
        if len(parts) != 2:
            return None
        folder, slug = parts
        if not _is_safe_ref(folder, slug):
            return None
        ref = f"{folder}/{slug}"
        return None if ref == owner_ref else ref
    # Bare slug: resolve against the index (candidates exclude the owner).
    candidates = [c for c in index.get(target.lower(), []) if c[0] != owner_ref]
    if not candidates:
        return None
    # Tie-break: most recent last_touched, then folder ascending. Two stable
    # sorts (least-significant key first).
    by_folder = sorted(candidates, key=lambda c: c[2])
    best = sorted(by_folder, key=lambda c: c[1], reverse=True)[0]
    return best[0]


def resolve_links(
    targets: list[str],
    index: dict[str, list[tuple[str, str, str]]],
    owner_ref: str | None = None,
) -> list[str]:
    """Resolve raw targets to a sorted, deduped list of canonical ``folder/slug``.

    Canonical refs are validated (and kept even if absent). Bare slugs resolve
    against ``index`` (empty index → none resolve). Self-references (== owner_ref)
    dropped. Deterministic: output is sorted.
    """
    out: list[str] = []
    seen: set[str] = set()
    for t in targets:
        ref = _resolve_one(t, index, owner_ref)
        if ref and ref not in seen:
            seen.add(ref)
            out.append(ref)
    return sorted(out)
