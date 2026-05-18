"""Tests for the wiki_note agent tool (Task 2.1: read/create + session routing)."""

from types import SimpleNamespace

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.wiki_note import WikiNoteTool


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


def test_session_routing_isolates_vaults(tmp_path):
    """Two sessions must resolve to distinct vault directories (multi-user isolation)."""
    a = WikiNoteTool.create(_ctx(tmp_path))
    a.set_context(RequestContext(channel="telegram", chat_id="1", session_key="telegram:1"))
    b = WikiNoteTool.create(_ctx(tmp_path))
    b.set_context(RequestContext(channel="telegram", chat_id="2", session_key="telegram:2"))
    assert a._vault().root != b._vault().root
