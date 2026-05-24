"""Wiki page data model with frontmatter parse/serialize.

A wiki page is a Markdown document with a leading YAML frontmatter block
fenced by ``---`` lines, followed by exactly one blank line and the body.

``serialize_page`` and ``parse_page`` are exact inverses for well-formed
pages. Serialization is deterministic (stable key order) so the Lint engine
can regenerate files idempotently. ``parse_page`` rejects malformed
frontmatter, an unknown ``status``, or an empty ``type`` -- this is the
admission-gate validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

__all__ = ["Page", "parse_page", "serialize_page"]

_VALID_STATUS = {"hot", "cold"}

# Stable, deterministic frontmatter key order. ``pinned`` is emitted only
# when it is not None (it is the single optional field).
_KEY_ORDER = (
    "type",
    "title",
    "status",
    "created",
    "updated",
    "last_touched",
    "tags",
    "links_out",
    "pinned",
    "summary",
    "sender_ids",
)


@dataclass
class Page:
    """A single wiki page.

    All frontmatter scalars are kept as strings (dates are stored verbatim
    so they round-trip exactly). ``pinned`` is the only optional field;
    ``None`` means it is absent from the frontmatter entirely.
    """

    type: str
    title: str
    status: str
    created: str
    updated: str
    last_touched: str
    tags: list[str] = field(default_factory=list)
    links_out: list[str] = field(default_factory=list)
    pinned: bool | None = None
    # Optional sender-card fields (Layer 2). ``summary`` is a short one-liner
    # surfaced in the runtime-context tail; ``sender_ids`` are channel-qualified
    # interlocutor ids (e.g. ``"telegram:136150230"``) the agent self-binds.
    # Both are emitted ONLY when present so fieldless pages serialize
    # byte-identically to before. ``None`` / ``[]`` mean absent.
    summary: str | None = None
    sender_ids: list[str] = field(default_factory=list)
    body: str = ""


def serialize_page(page: Page) -> str:
    """Serialize a :class:`Page` to ``---\\n<frontmatter>\\n---\\n\\n<body>``.

    Frontmatter keys are emitted in :data:`_KEY_ORDER`. ``pinned`` is
    omitted when ``None``. The body is appended verbatim after the closing
    fence and exactly one blank line.
    """
    values: dict[str, object] = {
        "type": page.type,
        "title": page.title,
        "status": page.status,
        "created": page.created,
        "updated": page.updated,
        "last_touched": page.last_touched,
        "tags": list(page.tags),
        "links_out": list(page.links_out),
        "pinned": page.pinned,
        "summary": page.summary,
        "sender_ids": list(page.sender_ids),
    }
    # _KEY_ORDER is the single source of truth for which keys are emitted and
    # in what order; building fm by iterating it keeps the two from drifting.
    fm: dict[str, object] = {}
    for key in _KEY_ORDER:
        value = values[key]
        # pinned is the sole optional bool: omitted entirely when None.
        if key == "pinned" and value is None:
            continue
        # Optional sender-card fields: omitted when absent so a page that
        # never set them serializes byte-identically to the pre-Layer-2 format.
        if key == "summary" and value is None:
            continue
        if key == "sender_ids" and not value:
            continue
        fm[key] = value

    # sort_keys=False preserves insertion order, which we built per _KEY_ORDER.
    fm_text = yaml.safe_dump(
        fm,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    # yaml.safe_dump always terminates with a newline.
    return f"---\n{fm_text}---\n\n{page.body}"


def parse_page(text: str) -> Page:
    """Parse ``text`` produced by :func:`serialize_page` back into a :class:`Page`.

    Raises :class:`ValueError` if the frontmatter fence is malformed or
    absent, ``status`` is not in ``{"hot", "cold"}``, or ``type`` is empty.
    A missing ``pinned`` key parses back as ``None``.
    """
    if not text.startswith("---\n"):
        raise ValueError("frontmatter block absent: missing opening '---' fence")

    # Strip the opening fence, then split on the closing fence line.
    rest = text[len("---\n"):]
    end = rest.find("\n---\n")
    if end == -1:
        raise ValueError("frontmatter block malformed: missing closing '---' fence")

    fm_text = rest[:end]
    after = rest[end + len("\n---\n"):]

    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"frontmatter block malformed: {exc}") from exc

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter block malformed: not a mapping")

    type_ = parsed.get("type")
    if not isinstance(type_, str) or not type_.strip():
        raise ValueError("frontmatter 'type' is empty or missing")

    status = parsed.get("status")
    if status not in _VALID_STATUS:
        raise ValueError(
            f"frontmatter 'status' must be one of {sorted(_VALID_STATUS)}, "
            f"got {status!r}"
        )

    def _str(key: str) -> str:
        value = parsed.get(key)
        if value is None:
            raise ValueError(f"frontmatter '{key}' is missing")
        return str(value)

    def _str_list(key: str) -> list[str]:
        value = parsed.get(key)
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"frontmatter '{key}' must be a list")
        return [str(item) for item in value]

    pinned = parsed.get("pinned", None)
    if pinned is not None and not isinstance(pinned, bool):
        raise ValueError("frontmatter 'pinned' must be a boolean if present")

    # Optional sender-card fields (absent -> None / []).
    summary = parsed.get("summary")
    summary = None if summary is None else str(summary)
    sender_ids = _str_list("sender_ids")

    # Body convention: after the closing "---\n" there is exactly one blank
    # line ("\n"), then the body verbatim. serialize_page emits "---\n\n<body>",
    # so `after` here is "\n<body>"; drop that single separator newline.
    if after.startswith("\n"):
        body = after[1:]
    else:
        body = after

    return Page(
        type=type_,
        title=_str("title"),
        status=status,
        created=_str("created"),
        updated=_str("updated"),
        last_touched=_str("last_touched"),
        tags=_str_list("tags"),
        links_out=_str_list("links_out"),
        pinned=pinned,
        summary=summary,
        sender_ids=sender_ids,
        body=body,
    )
