"""Load and render agent system prompt templates (Jinja2) under nanobot/templates/.

Agent prompts live in ``templates/agent/`` (pass names like ``agent/identity.md``).
Shared copy lives under ``agent/_snippets/`` and is included via
``{% include 'agent/_snippets/....md' %}``.
"""

from contextlib import suppress
from functools import lru_cache
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


def is_bundled_template_content(content: str, template_path: str) -> bool:
    """Whether *content* is the bundled stock template (user never customized it).

    ``template_path`` is relative to ``nanobot/templates/`` (e.g.
    ``"memory/MEMORY.md"``). Returns True iff the trimmed *content* equals the
    trimmed bundled template text. A missing template (or any read error)
    yields False — unknown means "treat as real content", never a false
    positive that would silently discard a user's data.

    Single source of truth for the "this is just the stock template" check:
    :meth:`nanobot.agent.context.ContextBuilder._is_template_content` and the
    Task 7.1 legacy migration both delegate here so the two can never diverge.
    This module is a leaf (stdlib + jinja2 only), so callers in any layer
    (context, the wiki vault migration) can reuse it without an import cycle.
    """
    with suppress(Exception):
        tpl = pkg_files("nanobot") / "templates" / template_path
        if tpl.is_file():
            return content.strip() == tpl.read_text(encoding="utf-8").strip()
    return False


@lru_cache
def _environment() -> Environment:
    # Plain-text prompts: do not HTML-escape variable values.
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """Render ``name`` (e.g. ``agent/identity.md``, ``agent/platform_policy.md``) under ``templates/``.

    Use ``strip=True`` for single-line user-facing strings when the file ends
    with a trailing newline you do not want preserved.
    """
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text
