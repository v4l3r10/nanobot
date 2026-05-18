"""``wiki_note`` agent tool: read/write pen into the per-user wiki memory.

This is the agent's interface onto the long-term wiki tree (design §3/§4).
Each request is routed to a per-session *vault* under
``<workspace>/memory/users/<slug>/wiki/`` so concurrent users never share
memory (report P3). Session isolation uses a per-instance
:class:`~contextvars.ContextVar`, mirroring :mod:`nanobot.agent.tools.spawn`.

Task 2.1 implements ``operation=read`` and a minimal ``operation=create``.
Task 2.2 hardens ``create`` with the SCHEMA admission gate (unknown-type +
frontmatter validation, design §3/P7), refuses duplicates, scaffolds the
per-type ``_index.md`` Map-of-Content, and runs the page-write + index-append
as one critical section under the per-vault async lock (design H2/Task 0.3).
``append`` + reheat and ``search`` arrive in Tasks 2.3-2.4; the parameter
schema here is intentionally the full stable set so it does not churn across
tasks.
"""

from __future__ import annotations

import datetime
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.filesystem import _FsTool
from nanobot.agent.tools.path_utils import is_under
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema
from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.agent.wiki.paths import vault_dir, vault_slug
from nanobot.agent.wiki.vault import Vault
from nanobot.utils.atomic import atomic_write_text
from nanobot.utils.helpers import safe_filename
from nanobot.utils.vault_lock import get_vault_lock

_FALLBACK_SESSION_KEY = "unified:default"


def _as_lines(text: str) -> set[str]:
    """Existing ``_index.md`` lines, each re-terminated with a single ``\\n``.

    Used for the idempotent stub check: comparing whole, newline-terminated
    lines (not a raw substring) so ``[[people/al]]`` does not falsely match
    against an existing ``[[people/alice]]`` stub.
    """
    return {f"{line}\n" for line in text.splitlines()}


@tool_parameters(
    tool_parameters_schema(
        operation=StringSchema(
            "What to do: 'read' a page by path, or 'create' a new leaf page."
        ),
        path=StringSchema(
            "For read: page path relative to wiki/, e.g. 'people/alice.md'."
        ),
        type=StringSchema(
            "For create: the page type, e.g. 'people', 'projects', 'concepts'."
        ),
        slug=StringSchema(
            "For create: the filename stem (no extension), e.g. 'alice'."
        ),
        title=StringSchema("For create: the human-readable page title."),
        body=StringSchema("For create: the Markdown body of the page."),
        text=StringSchema("Reserved for the append operation (Task 2.3)."),
        query=StringSchema("Reserved for the search operation (Task 2.4)."),
        required=["operation"],
    )
)
class WikiNoteTool(_FsTool, ContextAware):
    """Read and create pages in the per-user long-term wiki memory."""

    _scopes = {"core"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-instance ContextVar so concurrent sessions sharing one tool
        # instance do not bleed routing into each other (spawn.py pattern).
        self._session_key_var: ContextVar[str] = ContextVar(
            "wiki_note_session_key", default=_FALLBACK_SESSION_KEY
        )

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        # Reuse _FsTool.create verbatim so _workspace / allowed_dir / sandbox
        # are wired exactly like every other filesystem tool. _FsTool.create
        # ends in ``return cls(...)``, so calling its underlying function with
        # this subclass constructs a fully-initialised WikiNoteTool (our
        # __init__ then sets up the session ContextVar).
        return _FsTool.create.__func__(cls, ctx)

    def set_context(self, ctx: RequestContext) -> None:
        self._session_key_var.set(ctx.session_key or f"{ctx.channel}:{ctx.chat_id}")

    def _session_key(self) -> str:
        return self._session_key_var.get()

    def _vault(self) -> Vault:
        root = vault_dir(Path(self._workspace), self._session_key())
        # Ensure the wiki/ subtree exists so writes (and a freshly created
        # vault's reads) work. The schema still falls back to the bundled
        # master — copying SCHEMA.md into the vault is a later task.
        (root / "wiki").mkdir(parents=True, exist_ok=True)
        return Vault(root)

    @property
    def name(self) -> str:
        return "wiki_note"

    @property
    def description(self) -> str:
        return (
            "Manage your long-term wiki memory (durable notes about people, "
            "projects, concepts, decisions). Operations:\n"
            "- read: fetch a page by its path relative to wiki/ "
            "(e.g. path='people/alice.md'). Returns frontmatter + body.\n"
            "- create: add a new leaf page (args: type, slug, title, "
            "optional body). The type is validated against the wiki schema "
            "(unknown types are refused); pages are filed by type and an "
            "index of each type is kept up to date automatically.\n"
            "You may only read and create leaf pages. You cannot move pages "
            "to cold storage, merge pages, or rewrite indexes/MOC files — "
            "those are automatic and Dream-only."
        )

    async def execute(self, operation: str | None = None, **kw: Any) -> str:
        if operation == "read":
            return self._do_read(kw.get("path"))
        if operation == "create":
            return await self._do_create(
                kw.get("type"),
                kw.get("slug"),
                kw.get("title"),
                kw.get("body") or "",
            )
        return f"Error: unknown operation {operation!r}"

    def _do_read(self, path: str | None) -> str:
        if not path:
            return "Error: 'path' is required for read"
        vault = self._vault()
        try:
            vault.read_page(path)
        except FileNotFoundError:
            return f"Page not found: {path}"
        except ValueError as e:
            return f"Error: {e}"
        # Return the raw serialized page text so the model sees frontmatter +
        # body. read_page already enforced wiki-dir containment (vault.py:86);
        # we reuse its resolved path rather than self._resolve so the sandbox
        # check and the vault-traversal guard cannot disagree.
        target = (vault.wiki_dir / path).resolve()
        try:
            return target.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error: {e}"

    async def _do_create(
        self,
        type: str | None,
        slug: str | None,
        title: str | None,
        body: str,
    ) -> str:
        if not type:
            return "Error: 'type' is required for create"
        if not slug:
            return "Error: 'slug' is required for create"
        if not title:
            return "Error: 'title' is required for create"

        vault = self._vault()
        schema = vault.schema

        # (1) Unknown-type admission gate (friendly). Refuse before building
        # anything so nothing is written for an unrecognised type.
        if not schema.is_known_type(type):
            allowed = ", ".join(sorted(schema.types))
            return f"Error: unknown type '{type}'. Allowed: {allowed}"

        today = datetime.date.today().isoformat()
        page = Page(
            type=type,
            title=title,
            status="hot",
            created=today,
            updated=today,
            last_touched=today,
            tags=[],
            links_out=[],
            pinned=None,
            body=body,
        )

        # (2) Frontmatter admission gate (design §3/P7). The tool always
        # constructs a complete Page so this normally passes, but it is the
        # explicit gate the design mandates — keep it, do not skip.
        fm = {
            "type": page.type,
            "title": page.title,
            "status": page.status,
            "created": page.created,
            "updated": page.updated,
            "last_touched": page.last_touched,
            "tags": page.tags,
            "links_out": page.links_out,
        }
        fm_errors = schema.validate_frontmatter(fm)
        if fm_errors:
            return "Error: " + "; ".join(fm_errors)

        safe_slug = safe_filename(slug)
        try:
            target = vault.page_path(type, safe_slug)
        except (KeyError, ValueError) as e:  # pragma: no cover - gated above
            return f"Error: {e}"

        # Enforce vault containment symmetrically with read_page's is_under
        # guard (Task 2.1 security fix — DO NOT remove). Schema.folder(type)
        # is whatever the per-vault SCHEMA.md declares with no single-component
        # validation; a folder of '../...' or an absolute path would let
        # atomic_write_text (parent.mkdir(parents=True)) write OUTSIDE the
        # vault. Resolve the destination and pass the *resolved* path to
        # atomic_write_text so the checked path and the written path are
        # identical (no TOCTOU).
        resolved = target.resolve()
        if not is_under(resolved, vault.wiki_dir):
            return (
                f"Error: refusing to write {type}/{slug} "
                "— resolves outside the vault"
            )

        folder = schema.folder(type)
        index_path = vault.wiki_dir / folder / "_index.md"
        resolved_index = index_path.resolve()
        if not is_under(resolved_index, vault.wiki_dir):
            return (
                f"Error: refusing to write the {type} index "
                "— resolves outside the vault"
            )

        # (5) Page-write + index-append are ONE critical section under the
        # per-vault async lock so a concurrent Dream-Lint pass or another
        # turn cannot interleave a half-written page/index (design H2).
        async with get_vault_lock(vault_slug(self._session_key())):
            # (3) Duplicate refusal: never overwrite an existing page; the
            # existing body must survive byte-for-byte. Checked inside the
            # lock so two concurrent creates of the same slug can't race.
            if resolved.exists():
                return (
                    f"Page already exists: {type}/{safe_slug}. "
                    "Use operation='append' to add to it."
                )

            try:
                atomic_write_text(resolved, serialize_page(page))
            except OSError as e:
                return f"Error: {e}"

            # (4) Append the wikilink stub to the type's _index.md MOC.
            # APPEND-ONLY: read current content, ensure the stub is present
            # exactly once, and write back all existing bytes unchanged plus
            # the stub if missing. Dream-Lint owns rewrites/reordering.
            stub = f"- [[{folder}/{safe_slug}]]\n"
            try:
                if resolved_index.exists():
                    current = resolved_index.read_text(encoding="utf-8")
                    # Idempotent re-run after a partial failure: if the exact
                    # stub line is already present, leave the index untouched.
                    if stub not in _as_lines(current):
                        new_text = current
                        if new_text and not new_text.endswith("\n"):
                            new_text += "\n"
                        atomic_write_text(resolved_index, new_text + stub)
                else:
                    header = f"# {type} index\n\n"
                    atomic_write_text(resolved_index, header + stub)
            except OSError as e:
                # The page is written; surface the index failure honestly so
                # a retry (idempotent above) can repair the MOC.
                return (
                    f"Created page {type}/{safe_slug}.md but failed to update "
                    f"the index: {e}"
                )

        return f"Created page {type}/{safe_slug}.md"
