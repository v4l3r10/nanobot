"""Tests for the wiki_note agent tool (Task 2.1: read/create + session routing;
Task 2.2: SCHEMA admission gate, _index stub, dup refusal, vault lock)."""

import asyncio
from types import SimpleNamespace

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.wiki_note import (
    WikiNoteTool,
    _TAG_MAX_LEN,
    _TAGS_MAX,
    _normalize_tags,
)
from nanobot.agent.wiki.page import Page, parse_page, serialize_page
from nanobot.agent.wiki.paths import vault_dir, vault_slug
from nanobot.utils.vault_lock import get_vault_lock


def _ctx(tmp_path):
    """Build a fake ToolContext-like object with exactly what _FsTool.create reads.

    _FsTool.create (filesystem.py:41-56) reads:
      ctx.config.restrict_to_workspace, ctx.config.exec.sandbox,
      ctx.workspace, ctx.file_state_store
    """
    return SimpleNamespace(
        config=SimpleNamespace(
            restrict_to_workspace=True,
            exec=SimpleNamespace(sandbox=False),
        ),
        workspace=str(tmp_path),
        file_state_store=None,
    )


def _tool(tmp_path):
    t = WikiNoteTool.create(_ctx(tmp_path))
    t.set_context(
        RequestContext(channel="telegram", chat_id="1", session_key="telegram:1")
    )
    return t


async def test_read_missing_page_returns_clear_message(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(operation="read", path="people/nobody.md")
    assert "not found" in out.lower()
    # Must be a clean string, not a raised exception leaking through.
    assert "Traceback" not in out


async def test_create_then_read(tmp_path):
    t = _tool(tmp_path)
    created = await t.execute(
        operation="create",
        type="people",
        slug="alice",
        title="Alice",
        body="Leads payments.",
    )
    assert "alice" in created.lower()
    out = await t.execute(operation="read", path="people/alice.md")
    assert "Leads payments." in out
    # Serialized page text must include the frontmatter the model needs.
    assert "title: Alice" in out
    assert "status: hot" in out


async def test_unknown_operation_returns_error_string(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(operation="frobnicate")
    assert isinstance(out, str)
    assert "unknown operation" in out.lower()
    assert "frobnicate" in out


def test_tool_is_auto_discoverable():
    # Concrete (no abstract methods left), correct name, core scope, and
    # discoverable so loader.ToolLoader.discover() picks it up.
    assert issubclass(WikiNoteTool, Tool)
    assert not getattr(WikiNoteTool, "__abstractmethods__", None)
    t = WikiNoteTool.__new__(WikiNoteTool)
    assert t.name == "wiki_note"
    assert "core" in WikiNoteTool._scopes
    assert WikiNoteTool._plugin_discoverable


async def test_create_refuses_schema_folder_traversal(tmp_path):
    """A malicious per-vault SCHEMA.md whose type folder escapes the vault
    must be refused by create, and must NOT write any file outside the vault.

    This exercises the latent sandbox hole that goes live in Task 2.2 once
    SCHEMA.md is resolved per-vault: ``Schema.folder(type)`` returns whatever
    string the SCHEMA declares, and create previously fed it straight to
    ``atomic_write_text`` (which does ``parent.mkdir(parents=True)``) with no
    containment check — unlike read, which is guarded by ``is_under``.
    """
    t = _tool(tmp_path)
    # The tool routes session_key="telegram:1" to this vault dir.
    vault_root = vault_dir(tmp_path, "telegram:1")
    wiki_dir = vault_root / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    # Malicious per-vault SCHEMA.md: the 'evil' type's folder climbs out of
    # the vault entirely, targeting a sibling of tmp_path.
    schema_md = (
        "```yaml\n"
        "version: 1\n"
        "types:\n"
        "  evil:\n"
        '    folder: "../../../../../escape"\n'
        "    cold_after_days: null\n"
        "```\n"
    )
    (wiki_dir / "SCHEMA.md").write_text(schema_md, encoding="utf-8")

    out = await t.execute(
        operation="create",
        type="evil",
        slug="x",
        title="X",
        body="pwn",
    )

    # (a) The tool must refuse with a clean model-readable error string.
    assert isinstance(out, str)
    assert out.lower().startswith("error:") or "refus" in out.lower()
    assert "Traceback" not in out

    # (b) Nothing may have been written outside the vault. Walk every
    # ancestor of the vault up to the filesystem root and assert no rogue
    # 'escape' directory (or x.md inside one) was created anywhere.
    for ancestor in [tmp_path, *tmp_path.parents]:
        rogue = ancestor / "escape"
        assert not rogue.exists(), f"escaped write created {rogue}"
        assert not (ancestor / "escape" / "x.md").exists()


def test_session_routing_isolates_vaults(tmp_path):
    """Two sessions must resolve to distinct vault directories (multi-user isolation)."""
    a = WikiNoteTool.create(_ctx(tmp_path))
    a.set_context(RequestContext(channel="telegram", chat_id="1", session_key="telegram:1"))
    b = WikiNoteTool.create(_ctx(tmp_path))
    b.set_context(RequestContext(channel="telegram", chat_id="2", session_key="telegram:2"))
    assert a._vault().root != b._vault().root


# --- Task 2.2: SCHEMA admission gate, _index stub, dup refusal, vault lock ---


def _wiki_dir(tmp_path):
    return vault_dir(tmp_path, "telegram:1") / "wiki"


def _all_md_files(wiki_dir):
    if not wiki_dir.is_dir():
        return []
    return sorted(wiki_dir.rglob("*.md"))


async def test_create_unknown_type_rejected(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(operation="create", type="aliens", slug="x", title="X")
    assert "unknown type" in out.lower()
    assert "Traceback" not in out
    # Nothing may have been written anywhere under the vault wiki dir.
    wiki_dir = _wiki_dir(tmp_path)
    written = [p for p in _all_md_files(wiki_dir) if p.name != "SCHEMA.md"]
    assert written == [], f"unexpected files written: {written}"


async def test_create_appends_index_stub(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(
        operation="create", type="people", slug="alice", title="Alice", body="hi"
    )
    assert "alice" in out.lower()
    index = _wiki_dir(tmp_path) / "people" / "_index.md"
    assert index.is_file(), "people/_index.md should be created"
    text = index.read_text(encoding="utf-8")
    assert "# people index" in text
    assert "[[people/alice]]" in text


async def test_create_duplicate_refused_no_overwrite(tmp_path):
    t = _tool(tmp_path)
    first = await t.execute(
        operation="create",
        type="people",
        slug="alice",
        title="Alice",
        body="ORIGINAL",
    )
    assert "alice" in first.lower()
    second = await t.execute(
        operation="create",
        type="people",
        slug="alice",
        title="Alice",
        body="CHANGED",
    )
    assert "already exists" in second.lower()
    page = await t.execute(operation="read", path="people/alice.md")
    assert "ORIGINAL" in page
    assert "CHANGED" not in page


async def test_create_index_append_only(tmp_path):
    t = _tool(tmp_path)
    index = _wiki_dir(tmp_path) / "people" / "_index.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("# people index\n\n- [[people/bob]]\n", encoding="utf-8")
    await t.execute(
        operation="create", type="people", slug="alice", title="Alice", body="hi"
    )
    text = index.read_text(encoding="utf-8")
    assert "[[people/bob]]" in text, "pre-existing bob stub must be preserved"
    assert "[[people/alice]]" in text, "new alice stub must be appended"
    # bob must come before alice (not reordered).
    assert text.index("[[people/bob]]") < text.index("[[people/alice]]")


async def test_create_index_stub_idempotent(tmp_path):
    t = _tool(tmp_path)
    await t.execute(
        operation="create", type="people", slug="alice", title="Alice", body="hi"
    )
    # Manually delete the page file but keep _index.md, then recreate.
    page_file = _wiki_dir(tmp_path) / "people" / "alice.md"
    page_file.unlink()
    await t.execute(
        operation="create", type="people", slug="alice", title="Alice", body="again"
    )
    index = _wiki_dir(tmp_path) / "people" / "_index.md"
    text = index.read_text(encoding="utf-8")
    assert text.count("[[people/alice]]") == 1, (
        f"stub line must appear exactly once, got:\n{text}"
    )


async def test_create_concurrent_same_vault_serialized(tmp_path):
    t = _tool(tmp_path)
    r1, r2 = await asyncio.gather(
        t.execute(
            operation="create", type="people", slug="alice", title="Alice", body="a"
        ),
        t.execute(
            operation="create", type="people", slug="bob", title="Bob", body="b"
        ),
    )
    assert "alice" in r1.lower()
    assert "bob" in r2.lower()
    index = _wiki_dir(tmp_path) / "people" / "_index.md"
    text = index.read_text(encoding="utf-8")
    # The per-vault lock must prevent a lost append: BOTH stubs present.
    assert "[[people/alice]]" in text
    assert "[[people/bob]]" in text


async def test_create_acquires_vault_lock(tmp_path):
    """The create write-path must run inside the per-vault async lock so a
    concurrent Dream-Lint pass cannot interleave a half-written page/index."""
    t = _tool(tmp_path)
    lock = get_vault_lock(vault_slug("telegram:1"))
    await lock.acquire()
    try:
        task = asyncio.ensure_future(
            t.execute(
                operation="create",
                type="people",
                slug="alice",
                title="Alice",
                body="hi",
            )
        )
        # While we hold the lock, create must NOT complete (it is blocked
        # waiting on the same lock) and must NOT have written the page.
        await asyncio.sleep(0.05)
        assert not task.done(), "create completed without acquiring the vault lock"
        page_file = _wiki_dir(tmp_path) / "people" / "alice.md"
        assert not page_file.exists(), "page written before acquiring the lock"
    finally:
        lock.release()
    out = await task
    assert "alice" in out.lower()
    assert (_wiki_dir(tmp_path) / "people" / "alice.md").exists()


# --- Task 2.2 review follow-up: slug sanitization (I1) + M2 gate coverage ---


async def test_create_rejects_newline_slug(tmp_path):
    """A slug containing a newline must be rejected with a friendly error and
    must NOT corrupt the MOC into a two-physical-line stub (I1).

    Before the fix, ``safe_filename`` left ``\\n`` untouched: the page
    filename became ``alice\\nbob.md`` and the ``_index.md`` stub became the
    two physical lines ``- [[people/alice`` / ``bob]]``, breaking the
    whole-line idempotency check Lint 4.3 / context 6.1 rely on.
    """
    t = _tool(tmp_path)
    out = await t.execute(
        operation="create",
        type="people",
        slug="alice\nbob",
        title="X",
        body="b",
    )
    assert isinstance(out, str)
    assert out.startswith("Error:"), out
    assert "invalid slug" in out.lower()
    assert "Traceback" not in out

    wiki_dir = _wiki_dir(tmp_path)
    # No page filename anywhere may contain a newline.
    for p in _all_md_files(wiki_dir):
        assert "\n" not in p.name, f"page filename contains newline: {p!r}"

    # The _index.md must either not exist or contain no broken two-physical-
    # line stub (an opening ``[[people/alice`` with no closing ``]]`` on the
    # same line).
    index = wiki_dir / "people" / "_index.md"
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            if "[[people/alice" in line:
                assert "]]" in line, (
                    f"broken two-line stub in _index.md: {line!r}"
                )


async def test_create_sanitizes_slug_separators(tmp_path):
    """Path separators / whitespace in a slug collapse to a single safe
    component and the page + a single well-formed stub are written."""
    t = _tool(tmp_path)
    out = await t.execute(
        operation="create",
        type="people",
        slug="a/b c",
        title="X",
        body="body",
    )
    assert out.startswith("Created page"), out
    # 'a/b c' -> safe_filename '/' -> '_' giving 'a_b c', then our rule maps
    # path-sep/whitespace to '-' and collapses: 'a-b-c'.
    assert "a-b-c.md" in out

    wiki_dir = _wiki_dir(tmp_path)
    page = wiki_dir / "people" / "a-b-c.md"
    assert page.is_file(), f"sanitized page not written; files: {_all_md_files(wiki_dir)}"

    index = wiki_dir / "people" / "_index.md"
    text = index.read_text(encoding="utf-8")
    stub_lines = [
        ln for ln in text.splitlines() if "[[people/a-b-c]]" in ln
    ]
    assert stub_lines == ["- [[people/a-b-c]]"], (
        f"expected one well-formed stub line, got: {text!r}"
    )


async def test_frontmatter_gate_rejects_when_schema_requires_extra(tmp_path):
    """A per-vault SCHEMA.md whose required_frontmatter names a key the tool
    never sets must make the (now real) frontmatter gate refuse, writing
    nothing. This makes the M2 gate genuinely covered."""
    t = _tool(tmp_path)
    wiki_dir = _wiki_dir(tmp_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    schema_md = (
        "```yaml\n"
        "version: 1\n"
        "types:\n"
        "  people:\n"
        '    folder: "people"\n'
        "    cold_after_days: null\n"
        "required_frontmatter:\n"
        "  - type\n"
        "  - title\n"
        "  - owner\n"
        "```\n"
    )
    (wiki_dir / "SCHEMA.md").write_text(schema_md, encoding="utf-8")

    out = await t.execute(
        operation="create",
        type="people",
        slug="alice",
        title="Alice",
        body="hi",
    )
    assert out.startswith("Error:"), out
    assert "owner" in out
    assert "Traceback" not in out

    # Nothing may have been written (no page, no index).
    written = [p for p in _all_md_files(wiki_dir) if p.name != "SCHEMA.md"]
    assert written == [], f"unexpected files written: {written}"


# --- Task 2.2 re-review: reserved/hidden slug, length clamp, Unicode Cc/Cf ---


async def test_create_rejects_reserved_slug(tmp_path):
    """A slug whose resulting basename collides with a vault-structural,
    non-page name (``SCHEMA.md``, ``_index.md``) must be refused outright
    (R1). Otherwise the page is permanently invisible to Ingest/Lint
    (``Vault._NON_PAGE_NAMES`` matches by basename regardless of folder) yet
    a dangling ``[[people/SCHEMA]]`` stub gets added to ``_index.md`` — a
    silently-broken phantom page. Nothing may be written and the index must
    contain NO dangling stub for the rejected slug.
    """
    t = _tool(tmp_path)
    for bad in ("SCHEMA", "_index"):
        out = await t.execute(
            operation="create",
            type="people",
            slug=bad,
            title="X",
            body="b",
        )
        assert isinstance(out, str)
        assert out.startswith("Error:"), out
        assert "reserved" in out.lower() or "invalid" in out.lower()
        assert "Traceback" not in out

    wiki_dir = _wiki_dir(tmp_path)
    # No colliding page file may exist (people/SCHEMA.md, people/_index.md
    # as a *page* — the only _index.md allowed is the MOC itself).
    assert not (wiki_dir / "people" / "SCHEMA.md").exists()
    # The MOC must not exist at all (nothing legitimate was created), or if
    # it somehow does it must carry no dangling stub for a rejected slug.
    index = wiki_dir / "people" / "_index.md"
    if index.exists():
        text = index.read_text(encoding="utf-8")
        assert "[[people/SCHEMA]]" not in text
        assert "[[people/_index]]" not in text


async def test_create_rejects_dotcold_and_hidden_slug(tmp_path):
    """A slug equal to the cold-marker component (``.cold``) or any slug
    whose sanitized name starts with ``.`` (hidden/structural) must be
    refused; nothing may be written (R1)."""
    t = _tool(tmp_path)
    for bad in (".cold", ".hidden"):
        out = await t.execute(
            operation="create",
            type="people",
            slug=bad,
            title="X",
            body="b",
        )
        assert isinstance(out, str)
        assert out.startswith("Error:"), out
        assert "reserved" in out.lower() or "invalid" in out.lower()
        assert "Traceback" not in out

    wiki_dir = _wiki_dir(tmp_path)
    written = [p for p in _all_md_files(wiki_dir) if p.name != "SCHEMA.md"]
    assert written == [], f"unexpected files written: {written}"


async def test_create_clamps_long_slug(tmp_path):
    """A ~300-char slug must be deterministically clamped to a single
    safe component <= the cap (no raw OSError/WinError, cross-platform
    identical), with exactly one well-formed _index stub. Running twice
    yields the same name (deterministic)."""
    t = _tool(tmp_path)
    long_slug = "a" * 300
    out = await t.execute(
        operation="create",
        type="people",
        slug=long_slug,
        title="X",
        body="body",
    )
    assert out.startswith("Created page"), out
    assert "Traceback" not in out
    assert "WinError" not in out

    wiki_dir = _wiki_dir(tmp_path)
    people = wiki_dir / "people"
    pages = [p for p in people.glob("*.md") if p.name != "_index.md"]
    assert len(pages) == 1, f"expected one page, got: {pages}"
    name = pages[0].name
    stem = name[:-3]  # strip .md
    assert "/" not in stem and "\\" not in stem, "must be single component"
    assert len(stem) <= 80, f"slug stem not clamped: {len(stem)} chars"
    assert set(stem) == {"a"}, f"unexpected clamp content: {stem!r}"

    index = people / "_index.md"
    text = index.read_text(encoding="utf-8")
    stub_lines = [ln for ln in text.splitlines() if "[[people/" in ln]
    assert stub_lines == [f"- [[people/{stem}]]"], (
        f"expected one well-formed stub, got: {text!r}"
    )

    # Deterministic: a second tool/session with the same long slug must
    # produce the identical clamped name.
    t2 = WikiNoteTool.create(_ctx(tmp_path))
    t2.set_context(
        RequestContext(channel="telegram", chat_id="9", session_key="telegram:9")
    )
    out2 = await t2.execute(
        operation="create",
        type="people",
        slug=long_slug,
        title="X",
        body="body",
    )
    assert out2.startswith("Created page"), out2
    assert stem in out2, f"non-deterministic clamp: {out!r} vs {out2!r}"


async def test_create_rejects_unicode_bidi_slug(tmp_path):
    """A slug containing a Unicode bidi-override / format char (U+202E)
    must be refused (R3): it survives into filenames + MOC wikilinks.
    Nothing may be written."""
    t = _tool(tmp_path)
    out = await t.execute(
        operation="create",
        type="people",
        slug="‮evil",
        title="X",
        body="b",
    )
    assert isinstance(out, str)
    assert out.startswith("Error:"), out
    assert "invalid slug" in out.lower()
    assert "Traceback" not in out

    wiki_dir = _wiki_dir(tmp_path)
    written = [p for p in _all_md_files(wiki_dir) if p.name != "SCHEMA.md"]
    assert written == [], f"unexpected files written: {written}"


async def test_create_allows_accented_slug(tmp_path):
    """Benign Unicode letters (accented latin, CJK) must still pass through
    _safe_slug and create successfully — _safe_slug must NOT ASCII-only the
    slug, only reject Cc/Cf control/format chars (R3)."""
    t = _tool(tmp_path)
    out = await t.execute(
        operation="create",
        type="people",
        slug="café-notes",
        title="X",
        body="body",
    )
    assert out.startswith("Created page"), out
    assert "Traceback" not in out

    wiki_dir = _wiki_dir(tmp_path)
    people = wiki_dir / "people"
    pages = [p for p in people.glob("*.md") if p.name != "_index.md"]
    assert len(pages) == 1, f"expected one page, got: {pages}"
    stem = pages[0].name[:-3]
    assert "caf" in stem and stem.endswith("notes")
    assert "/" not in stem and "\\" not in stem

    index = people / "_index.md"
    text = index.read_text(encoding="utf-8")
    stub_lines = [ln for ln in text.splitlines() if "[[people/" in ln]
    assert stub_lines == [f"- [[people/{stem}]]"], (
        f"expected one well-formed stub, got: {text!r}"
    )


async def test_create_allows_legitimate_design_slugs(tmp_path):
    """Sanity: ordinary design slugs (payment-svc, 2026-some-decision,
    auth-model) must still create cleanly through the hardened seam."""
    t = _tool(tmp_path)
    for slug in ("payment-svc", "2026-some-decision", "auth-model"):
        out = await t.execute(
            operation="create",
            type="concepts",
            slug=slug,
            title="X",
            body="b",
        )
        assert out.startswith("Created page"), (slug, out)
        assert f"{slug}.md" in out, (slug, out)


# --- Task 2.3: append operation + reheat-on-read ---


import datetime  # noqa: E402


def _today():
    return datetime.date.today().isoformat()


async def test_append_adds_text_bumps_freshness(tmp_path):
    """append grows the body (one separator newline) and bumps updated +
    last_touched to today, leaving every other frontmatter field byte-equal."""
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="alice",
        title="Alice",
        body="placeholder",
    )
    page_file = _wiki_dir(tmp_path) / "people" / "alice.md"
    # Overwrite on disk with controlled, stale frontmatter.
    stale = Page(
        type="people",
        title="Alice",
        status="hot",
        created="2020-01-01",
        updated="2020-01-01",
        last_touched="2020-01-01",
        tags=["x"],
        links_out=["people/bob"],
        pinned=True,
        body="OLD.\n",
    )
    page_file.write_text(serialize_page(stale), encoding="utf-8")

    out = await t.execute(
        operation="append", path="people/alice.md", text="NEW LINE"
    )
    assert "Traceback" not in out
    assert not out.startswith("Error:"), out

    page = parse_page(page_file.read_text(encoding="utf-8"))
    assert "OLD." in page.body
    assert "NEW LINE" in page.body
    assert page.updated == _today()
    assert page.last_touched == _today()
    # Everything else byte-identical.
    assert page.created == "2020-01-01"
    assert page.title == "Alice"
    assert page.type == "people"
    assert page.status == "hot"
    assert page.tags == ["x"]
    assert page.links_out == ["people/bob"]
    assert page.pinned is True


async def test_append_missing_page_errors(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(
        operation="append", path="people/ghost.md", text="x"
    )
    assert out.startswith("Error:"), out
    assert "not found" in out.lower()
    assert "Traceback" not in out
    wiki_dir = _wiki_dir(tmp_path)
    written = [p for p in _all_md_files(wiki_dir) if p.name != "SCHEMA.md"]
    assert written == [], f"unexpected files written: {written}"


async def test_append_refuses_reserved_path(tmp_path):
    t = _tool(tmp_path)
    # Create a real page so the people/ folder + _index.md exist.
    await t.execute(
        operation="create",
        type="people",
        slug="alice",
        title="Alice",
        body="hi",
    )
    people = _wiki_dir(tmp_path) / "people"
    index = people / "_index.md"
    schema_md = people.parent / "SCHEMA.md"
    schema_md.write_text("# fake schema\n", encoding="utf-8")

    before_index = index.read_text(encoding="utf-8")
    before_schema = schema_md.read_text(encoding="utf-8")

    out1 = await t.execute(
        operation="append", path="people/_index.md", text="x"
    )
    assert out1.startswith("Error:"), out1
    assert "Traceback" not in out1
    assert index.read_text(encoding="utf-8") == before_index

    out2 = await t.execute(
        operation="append", path="SCHEMA.md", text="x"
    )
    assert out2.startswith("Error:"), out2
    assert "Traceback" not in out2
    assert schema_md.read_text(encoding="utf-8") == before_schema


async def test_append_refuses_cold_path(tmp_path):
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="old",
        title="Old",
        body="archived",
    )
    src = _wiki_dir(tmp_path) / "people" / "old.md"
    cold_dir = _wiki_dir(tmp_path) / ".cold" / "people"
    cold_dir.mkdir(parents=True, exist_ok=True)
    cold_page = Page(
        type="people",
        title="Old",
        status="cold",
        created="2020-01-01",
        updated="2020-01-01",
        last_touched="2020-01-01",
        body="archived\n",
    )
    cold_file = cold_dir / "old.md"
    cold_file.write_text(serialize_page(cold_page), encoding="utf-8")
    src.unlink()

    before = cold_file.read_text(encoding="utf-8")
    out = await t.execute(
        operation="append", path=".cold/people/old.md", text="x"
    )
    assert out.startswith("Error:"), out
    assert "Traceback" not in out
    assert cold_file.read_text(encoding="utf-8") == before


async def test_read_reheats_cold_page(tmp_path):
    t = _tool(tmp_path)
    people = _wiki_dir(tmp_path) / "people"
    people.mkdir(parents=True, exist_ok=True)
    frozen = people / "frozen.md"
    cold_page = Page(
        type="people",
        title="Frozen",
        status="cold",
        created="2020-01-01",
        updated="2020-01-01",
        last_touched="2020-01-01",
        body="thawing\n",
    )
    frozen.write_text(serialize_page(cold_page), encoding="utf-8")

    out = await t.execute(operation="read", path="people/frozen.md")
    assert "status: hot" in out
    assert "Traceback" not in out
    on_disk = parse_page(frozen.read_text(encoding="utf-8"))
    assert on_disk.status == "hot"
    assert on_disk.last_touched == _today()

    # Second read of a now-hot page must NOT rewrite the file.
    mtime_after_reheat = frozen.stat().st_mtime_ns
    bytes_after_reheat = frozen.read_text(encoding="utf-8")
    out2 = await t.execute(operation="read", path="people/frozen.md")
    assert "status: hot" in out2
    assert frozen.stat().st_mtime_ns == mtime_after_reheat, (
        "hot read rewrote the file"
    )
    assert frozen.read_text(encoding="utf-8") == bytes_after_reheat
    assert out2 == bytes_after_reheat


async def test_read_hot_page_does_not_rewrite(tmp_path):
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="bob",
        title="Bob",
        body="payments",
    )
    page_file = _wiki_dir(tmp_path) / "people" / "bob.md"
    mtime_before = page_file.stat().st_mtime_ns
    out = await t.execute(operation="read", path="people/bob.md")
    assert "payments" in out
    assert page_file.stat().st_mtime_ns == mtime_before, (
        "hot read performed a write"
    )


async def test_append_then_read_roundtrip(tmp_path):
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="carol",
        title="Carol",
        body="start",
    )
    appended = await t.execute(
        operation="append", path="people/carol.md", text="EXTRA42"
    )
    assert not appended.startswith("Error:"), appended
    out = await t.execute(operation="read", path="people/carol.md")
    assert "EXTRA42" in out
    assert "start" in out


async def test_append_rejects_whitespace_only_text(tmp_path):
    """append must reject whitespace-only text (``"   \\n"``) the same way it
    rejects missing/empty text: appending only blank lines would just inject
    noise into the body. The page body must be byte-unchanged (M3)."""
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="x",
        title="X",
        body="keep",
    )
    page_file = _wiki_dir(tmp_path) / "people" / "x.md"
    before = page_file.read_text(encoding="utf-8")

    out = await t.execute(
        operation="append", path="people/x.md", text="   \n"
    )
    assert out.startswith("Error:"), out
    assert "text" in out.lower()
    assert "Traceback" not in out

    # Body must be byte-unchanged: no blank line appended.
    after = page_file.read_text(encoding="utf-8")
    assert after == before, f"whitespace append mutated the page:\n{after!r}"


async def test_read_cold_path_reheats_in_place(tmp_path):
    """Reading a page whose path is under ``.cold/`` reheats it IN PLACE
    (``status: cold``→``hot``, ``last_touched``=today) while leaving the file
    physically under ``.cold/``. The physical relocation out of ``.cold/`` is
    Lint's job (Task 4.3) — this locks the documented reheat/``.cold/``
    hand-off contract (design §4)."""
    t = _tool(tmp_path)
    cold_dir = _wiki_dir(tmp_path) / ".cold" / "people"
    cold_dir.mkdir(parents=True, exist_ok=True)
    cold_file = cold_dir / "old.md"
    cold_page = Page(
        type="people",
        title="Old",
        status="cold",
        created="2020-01-01",
        updated="2020-01-01",
        last_touched="2020-01-01",
        body="archived\n",
    )
    cold_file.write_text(serialize_page(cold_page), encoding="utf-8")

    out = await t.execute(operation="read", path=".cold/people/old.md")
    assert "Traceback" not in out
    assert "status: hot" in out

    # Reheated IN PLACE: the file is still at the .cold/ path, flipped hot.
    assert cold_file.exists(), "cold-path file must remain at the .cold/ path"
    on_disk = parse_page(cold_file.read_text(encoding="utf-8"))
    assert on_disk.status == "hot"
    assert on_disk.last_touched == _today()

    # No hot-location copy was created (relocation is Lint's job, not read's).
    hot_copy = _wiki_dir(tmp_path) / "people" / "old.md"
    assert not hot_copy.exists(), (
        "read must NOT relocate out of .cold/ (that is Lint/Task 4.3)"
    )


# --- Task 2.4: search operation (keyword/tag/recency over hot + cold) ---


def _seed_page(tmp_path, relpath, page):
    """Write ``serialize_page(page)`` to ``wiki/<relpath>`` for telegram:1."""
    target = _wiki_dir(tmp_path) / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_page(page), encoding="utf-8")
    return target


def _page(**kw):
    """A Page with sane defaults; override per test."""
    base = dict(
        type="people",
        title="T",
        status="hot",
        created="2026-01-01",
        updated="2026-01-01",
        last_touched="2026-01-01",
        tags=[],
        links_out=[],
        pinned=None,
        body="",
    )
    base.update(kw)
    return Page(**base)


async def test_search_keyword_matches_hot_and_cold(tmp_path):
    t = _tool(tmp_path)
    _seed_page(
        tmp_path,
        "people/alice.md",
        _page(type="people", title="Alice", body="handles the payment flow"),
    )
    _seed_page(
        tmp_path,
        "projects/pay.md",
        _page(type="projects", title="Payment Service", body="misc"),
    )
    _seed_page(
        tmp_path,
        ".cold/concepts/legacy.md",
        _page(
            type="concepts",
            title="Legacy",
            status="cold",
            body="legacy payment code",
        ),
    )

    out = await t.execute(operation="search", query="payment")
    assert "Traceback" not in out
    assert "people/alice.md" in out
    assert "projects/pay.md" in out
    assert ".cold/concepts/legacy.md" in out
    # The cold one is marked (cold); the hot ones are not.
    for line in out.splitlines():
        if ".cold/concepts/legacy.md" in line:
            assert "(cold)" in line
        if "people/alice.md" in line or "projects/pay.md" in line:
            assert "(cold)" not in line


async def test_search_ranks_title_over_body(tmp_path):
    t = _tool(tmp_path)
    _seed_page(
        tmp_path,
        "concepts/a.md",
        _page(type="concepts", title="Widget design", body="nothing here"),
    )
    _seed_page(
        tmp_path,
        "concepts/b.md",
        _page(
            type="concepts",
            title="Other",
            body="a longer note that only mentions widget once among many other words",
        ),
    )
    out = await t.execute(operation="search", query="widget")
    assert "concepts/a.md" in out
    assert "concepts/b.md" in out
    # BM25 length normalization: 'widget' is prominent in the short, focused A
    # but incidental in the longer B (same term frequency, larger document), so
    # A scores higher and ranks first.
    assert out.index("concepts/a.md") < out.index("concepts/b.md")


async def test_search_ranks_by_term_relevance_multiword(tmp_path):
    t = _tool(tmp_path)
    _seed_page(
        tmp_path, "projects/logistica.md",
        _page(type="projects", title="Logistica",
              body="piano logistica magazzino spedizioni logistica"),
    )
    _seed_page(
        tmp_path, "people/bob.md",
        _page(type="people", title="Bob",
              body="bob ogni tanto parla di logistica durante le riunioni"),
    )
    # No page contains the exact phrase "logistica magazzino" (old substring
    # ranker would find nothing); BM25 tokenizes and ranks the focused page first.
    out = await t.execute(operation="search", query="logistica magazzino")
    assert "projects/logistica.md" in out
    assert "people/bob.md" in out
    assert out.index("projects/logistica.md") < out.index("people/bob.md")


async def test_search_tag_filter(tmp_path):
    t = _tool(tmp_path)
    _seed_page(
        tmp_path,
        "people/x.md",
        _page(type="people", title="X", tags=["eng", "team"]),
    )
    _seed_page(
        tmp_path,
        "people/y.md",
        _page(type="people", title="Y", tags=["ops"]),
    )
    _seed_page(
        tmp_path,
        "people/z.md",
        _page(type="people", title="Z", tags=["ENG"]),
    )
    out = await t.execute(operation="search", query="tag:eng")
    assert "people/x.md" in out
    assert "people/y.md" not in out
    # Case-insensitive: a page tagged "ENG" is matched by tag:eng.
    assert "people/z.md" in out


async def test_search_empty_query_returns_recent(tmp_path):
    t = _tool(tmp_path)
    _seed_page(
        tmp_path,
        "people/old.md",
        _page(type="people", title="Old", last_touched="2026-01-01"),
    )
    _seed_page(
        tmp_path,
        "people/mid.md",
        _page(type="people", title="Mid", last_touched="2026-03-01"),
    )
    _seed_page(
        tmp_path,
        "people/new.md",
        _page(type="people", title="New", last_touched="2026-05-01"),
    )
    for q in ("", None):
        out = (
            await t.execute(operation="search", query=q)
            if q is not None
            else await t.execute(operation="search")
        )
        assert "people/new.md" in out
        assert "people/mid.md" in out
        assert "people/old.md" in out
        # Most-recent first.
        assert out.index("people/new.md") < out.index("people/mid.md")
        assert out.index("people/mid.md") < out.index("people/old.md")


async def test_search_caps_at_20(tmp_path):
    t = _tool(tmp_path)
    for i in range(25):
        _seed_page(
            tmp_path,
            f"concepts/p{i:02d}.md",
            _page(type="concepts", title=f"P{i}", body="has a zebra in it"),
        )
    out = await t.execute(operation="search", query="zebra")
    assert "Traceback" not in out
    page_lines = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(page_lines) == 20, f"expected 20 page lines, got {len(page_lines)}"
    assert "more not shown" in out


async def test_search_no_match_message(tmp_path):
    t = _tool(tmp_path)
    _seed_page(
        tmp_path,
        "people/alice.md",
        _page(type="people", title="Alice", body="payments"),
    )
    out = await t.execute(operation="search", query="zzqqnomatchxx")
    assert isinstance(out, str)
    assert out.strip() != ""
    assert "No matching pages" in out
    assert "Traceback" not in out


async def test_search_does_not_reheat(tmp_path):
    t = _tool(tmp_path)
    cold_file = _seed_page(
        tmp_path,
        ".cold/concepts/frozen.md",
        _page(
            type="concepts",
            title="Frozen",
            status="cold",
            last_touched="2020-01-01",
            body="cold knowledge about widgets",
        ),
    )
    mtime_before = cold_file.stat().st_mtime_ns
    bytes_before = cold_file.read_bytes()

    out = await t.execute(operation="search", query="widgets")
    assert ".cold/concepts/frozen.md" in out

    # Search must NEVER reheat: file byte-identical, mtime untouched, still cold.
    assert cold_file.read_bytes() == bytes_before
    assert cold_file.stat().st_mtime_ns == mtime_before, (
        "search rewrote the cold page (must never reheat — only read does)"
    )
    on_disk = parse_page(cold_file.read_text(encoding="utf-8"))
    assert on_disk.status == "cold"


# --- Task 3: wiki_note marks its vault dirty on create/append success ---


def _tool_for(tmp_path, session_key):
    """``_tool`` parametrized by session key (same wiring as ``_tool``)."""
    t = WikiNoteTool.create(_ctx(tmp_path))
    channel, _, chat_id = session_key.partition(":")
    t.set_context(
        RequestContext(
            channel=channel, chat_id=chat_id, session_key=session_key
        )
    )
    return t


async def test_wiki_note_create_marks_vault_dirty(tmp_path):
    from nanobot.agent.wiki.moc_refresh import take_dirty

    t = _tool_for(tmp_path, "telegram:42")
    slug = vault_slug("telegram:42")
    # Clean precondition (no stale mark from another test).
    assert take_dirty(slug) is False

    out = await t.execute(
        operation="create",
        type="concepts",
        slug="alpha",
        title="Alpha",
        body="b",
    )
    # Same success assertion style the existing create tests use.
    assert out.startswith("Created page"), out
    assert "alpha" in out.lower()

    # Marked exactly once by the successful create.
    assert take_dirty(slug) is True
    assert take_dirty(slug) is False


async def test_wiki_note_failed_create_does_not_mark(tmp_path):
    from nanobot.agent.wiki.moc_refresh import take_dirty

    t = _tool_for(tmp_path, "telegram:42")
    slug = vault_slug("telegram:42")
    assert take_dirty(slug) is False

    out = await t.execute(
        operation="create",
        type="nonsuchtype",
        slug="x",
        title="X",
    )
    assert out.lower().startswith("error:"), out
    assert "unknown type" in out.lower()
    # A refused create (schema admission gate) must NOT mark the vault.
    assert take_dirty(slug) is False


async def test_wiki_note_append_marks_vault_dirty(tmp_path):
    from nanobot.agent.wiki.moc_refresh import take_dirty

    t = _tool_for(tmp_path, "telegram:42")
    slug = vault_slug("telegram:42")
    assert take_dirty(slug) is False

    created = await t.execute(
        operation="create",
        type="concepts",
        slug="beta",
        title="Beta",
        body="start",
    )
    assert created.startswith("Created page"), created
    # Consume the mark left by the successful create so we isolate append.
    assert take_dirty(slug) is True
    assert take_dirty(slug) is False

    out = await t.execute(
        operation="append", path="concepts/beta.md", text="more"
    )
    assert not out.startswith("Error:"), out

    assert take_dirty(slug) is True
    assert take_dirty(slug) is False


async def test_wiki_note_failed_append_does_not_mark(tmp_path):
    from nanobot.agent.wiki.moc_refresh import take_dirty

    t = _tool_for(tmp_path, "telegram:42")
    slug = vault_slug("telegram:42")
    assert take_dirty(slug) is False

    out = await t.execute(
        operation="append", path="concepts/ghost.md", text="x"
    )
    assert out.startswith("Error:"), out
    assert "not found" in out.lower()
    # A failed append must NOT mark the vault.
    assert take_dirty(slug) is False


# --- search tool-line logging (visibility) ----------------------------------


async def test_search_logs_tool_line_hybrid(tmp_path):
    from loguru import logger
    t = _tool(tmp_path)
    _seed_page(tmp_path, "people/alice.md",
               _page(type="people", title="Alice", body="handles the payment flow"))
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        await t.execute(operation="search", query="payment")
    finally:
        logger.remove(sink_id)
    line = next((m for m in captured if "wiki_note search" in m), "")
    assert line, captured
    assert "[hybrid]" in line
    assert "shown" in line and "total" in line


async def test_search_logs_tool_line_tag(tmp_path):
    from loguru import logger
    t = _tool(tmp_path)
    _seed_page(tmp_path, "people/alice.md",
               _page(type="people", title="Alice", body="x", tags=["eng"]))
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        await t.execute(operation="search", query="tag:eng")
    finally:
        logger.remove(sink_id)
    assert any("wiki_note search [tag:eng]" in m for m in captured), captured


async def test_search_logs_tool_line_recent(tmp_path):
    from loguru import logger
    t = _tool(tmp_path)
    _seed_page(tmp_path, "people/alice.md",
               _page(type="people", title="Alice", body="x"))
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        await t.execute(operation="search", query="")
    finally:
        logger.remove(sink_id)
    assert any("wiki_note search [recent]" in m for m in captured), captured


# --- Layer 2: agent self-binding (sender_ids + summary frontmatter) ---------


async def test_bind_sets_summary_and_sender_id(tmp_path):
    t = _tool(tmp_path)
    await t.execute(operation="create", type="people", slug="alice",
                    title="Alice", body="Leads payments.")
    out = await t.execute(operation="bind", path="people/alice.md",
                          sender_id="telegram:1|alice", summary="dev, IT informale")
    assert "alice" in out.lower() and "Traceback" not in out
    page = parse_page(
        (vault_dir(tmp_path, "telegram:1") / "wiki" / "people" / "alice.md")
        .read_text(encoding="utf-8")
    )
    assert page.summary == "dev, IT informale"
    assert "telegram:1|alice" in page.sender_ids


async def test_bind_dedupes_sender_id(tmp_path):
    t = _tool(tmp_path)
    await t.execute(operation="create", type="people", slug="alice", title="Alice")
    await t.execute(operation="bind", path="people/alice.md", sender_id="telegram:1|a")
    await t.execute(operation="bind", path="people/alice.md", sender_id="telegram:1|a")
    page = parse_page(
        (vault_dir(tmp_path, "telegram:1") / "wiki" / "people" / "alice.md")
        .read_text(encoding="utf-8")
    )
    assert page.sender_ids.count("telegram:1|a") == 1


async def test_bind_missing_page_errors_cleanly(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(operation="bind", path="people/ghost.md",
                          sender_id="telegram:1|g")
    assert "not found" in out.lower()
    assert "Traceback" not in out


async def test_bind_requires_some_field(tmp_path):
    t = _tool(tmp_path)
    await t.execute(operation="create", type="people", slug="alice", title="Alice")
    out = await t.execute(operation="bind", path="people/alice.md")
    assert "error" in out.lower()


# --- tags on create: _normalize_tags (pure) + create wiring ---


def test_normalize_tags_none_and_empty_return_empty():
    assert _normalize_tags(None) == []
    assert _normalize_tags([]) == []
    assert _normalize_tags(["", "   ", "\t"]) == []


def test_normalize_tags_slug_style_lowercase_and_dashes():
    assert _normalize_tags(["Project Alpha!"]) == ["project-alpha"]
    assert _normalize_tags(["Machine   Learning"]) == ["machine-learning"]
    # path separators are folded too
    assert _normalize_tags(["a/b c"]) == ["a-b-c"]


def test_normalize_tags_keeps_accented_and_cjk_letters():
    assert _normalize_tags(["Caffè", "记忆"]) == ["caffè", "记忆"]


def test_normalize_tags_strips_symbols_and_emoji():
    assert _normalize_tags(["c++", "hello🎉", "#python"]) == ["c", "hello", "python"]


def test_normalize_tags_dedup_preserves_first_seen_order():
    assert _normalize_tags(["Python", "async", "python", "ASYNC"]) == ["python", "async"]


def test_normalize_tags_caps_to_max_dropping_extras():
    raw = [f"tag{i}" for i in range(_TAGS_MAX + 5)]
    out = _normalize_tags(raw)
    assert out == [f"tag{i}" for i in range(_TAGS_MAX)]
    assert len(out) == _TAGS_MAX


def test_normalize_tags_clamps_length_without_dangling_separator():
    long = "a" * (_TAG_MAX_LEN + 10)
    assert _normalize_tags([long]) == ["a" * _TAG_MAX_LEN]
    # a value long enough to be clamped never comes back ending on a separator
    tag = "ab-" * _TAG_MAX_LEN
    out = _normalize_tags([tag])[0]
    assert len(out) <= _TAG_MAX_LEN
    assert not out.endswith("-")


def test_normalize_tags_accepts_a_bare_string():
    # Models sometimes send a single string instead of a one-element array.
    assert _normalize_tags("Python") == ["python"]


async def test_create_with_tags_normalizes_and_persists(tmp_path):
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="bob",
        title="Bob",
        body="Runs ops.",
        tags=["Project Alpha!", "ops", "Project Alpha!"],
    )
    out = await t.execute(operation="read", path="people/bob.md")
    page = parse_page(out)
    assert page.tags == ["project-alpha", "ops"]


async def test_create_without_tags_is_empty_list(tmp_path):
    t = _tool(tmp_path)
    await t.execute(
        operation="create", type="people", slug="carol", title="Carol", body="x",
    )
    page = parse_page(await t.execute(operation="read", path="people/carol.md"))
    assert page.tags == []


async def test_create_with_garbage_tags_still_succeeds(tmp_path):
    t = _tool(tmp_path)
    out = await t.execute(
        operation="create",
        type="people",
        slug="dave",
        title="Dave",
        body="x",
        tags=["", "   ", "🎉", "!!!"],
    )
    assert "dave" in out.lower()  # create succeeded
    assert "error" not in out.lower()
    page = parse_page(await t.execute(operation="read", path="people/dave.md"))
    assert page.tags == []


async def test_create_caps_tags_in_schema_and_normalizer(tmp_path):
    # The cap constant is honored end-to-end (single _TAGS_MAX source).
    t = _tool(tmp_path)
    await t.execute(
        operation="create",
        type="people",
        slug="erin",
        title="Erin",
        body="x",
        tags=[f"tag{i}" for i in range(_TAGS_MAX + 3)],
    )
    page = parse_page(await t.execute(operation="read", path="people/erin.md"))
    assert len(page.tags) == _TAGS_MAX

    # And the declared JSON schema advertises the same cap.
    props = t.parameters["properties"]
    assert props["tags"]["maxItems"] == _TAGS_MAX
