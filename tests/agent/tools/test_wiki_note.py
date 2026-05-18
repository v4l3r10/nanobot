"""Tests for the wiki_note agent tool (Task 2.1: read/create + session routing;
Task 2.2: SCHEMA admission gate, _index stub, dup refusal, vault lock)."""

import asyncio
from types import SimpleNamespace

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.wiki_note import WikiNoteTool
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
