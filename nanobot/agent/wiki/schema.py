"""SCHEMA.md parser + validator for the wiki memory tree.

``SCHEMA.md`` is the declarative layer of the wiki: a human-curated,
machine-parsable Markdown file whose FIRST fenced ``yaml`` block declares

* ``types``: a mapping of type name -> ``{folder, cold_after_days}``, where
  ``cold_after_days: null`` means the type never decays;
* ``required_frontmatter``: the frontmatter keys every page must carry;
* ``moc_max_lines``: the soft cap a Map-of-Content file is linted against.

It drives the admission-gate validation and the hot/cold decay policy. A
single bundled master lives at ``nanobot/templates/memory/wiki/SCHEMA.md``
and is copied into each per-user vault on first use.

PyYAML usage mirrors :mod:`nanobot.agent.wiki.page`: ``yaml.safe_load`` with
``yaml.YAMLError`` re-raised as ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

__all__ = ["Schema", "load_schema"]

# Fallback when SCHEMA.md omits ``moc_max_lines``; matches the bundled master.
_DEFAULT_MOC_MAX_LINES = 120


def _extract_yaml_block(text: str) -> str | None:
    """Return the body of the FIRST ```yaml fenced block in ``text``.

    The opening fence is a line whose stripped content is ```` ```yaml ````;
    the closing fence is a line whose stripped content is ```` ``` ````.
    Returns ``None`` if no such block is present.
    """
    lines = text.splitlines()
    in_block = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_block:
            if stripped == "```yaml":
                in_block = True
            continue
        if stripped == "```":
            return "\n".join(collected)
        collected.append(line)
    return None


@dataclass(frozen=True)
class Schema:
    """Parsed, effectively-immutable view of ``SCHEMA.md``.

    ``types`` maps a type name to its ``{folder, cold_after_days}`` mapping.
    The dataclass is frozen; its dict/list fields are never mutated.
    """

    types: dict[str, dict]
    required_frontmatter: list[str]
    moc_max_lines: int

    def is_known_type(self, type: str) -> bool:
        """Whether ``type`` is declared in the schema's ``types`` mapping."""
        return type in self.types

    def folder(self, type: str) -> str:
        """Return the vault subfolder configured for ``type``.

        Raises :class:`KeyError` for an unknown type.
        """
        return self.types[type]["folder"]

    def cold_after_days(self, type: str) -> int | None:
        """Days before a page of ``type`` is eligible to go cold.

        Returns ``None`` when the type never decays (``cold_after_days:
        null``) or when ``type`` is unknown.
        """
        spec = self.types.get(type)
        if spec is None:
            return None
        return spec.get("cold_after_days")

    def validate_frontmatter(self, fm: dict) -> list[str]:
        """Validate page frontmatter against the schema.

        Returns a list of human-readable error strings: one per missing
        ``required_frontmatter`` key, plus one if ``fm['type']`` is not a
        known type. An empty list means the frontmatter is valid.
        """
        errors: list[str] = []
        for key in self.required_frontmatter:
            if key not in fm:
                errors.append(f"missing required frontmatter field: {key!r}")
        type_ = fm.get("type")
        if type_ is not None and not self.is_known_type(type_):
            errors.append(f"unknown type: {type_!r}")
        return errors


def load_schema(text: str) -> Schema:
    """Parse ``SCHEMA.md`` ``text`` into a :class:`Schema`.

    Raises :class:`ValueError` if there is no ```yaml block, the block is
    not valid YAML, the parsed document is not a mapping, or it lacks a
    non-empty ``types`` mapping. ``required_frontmatter`` defaults to ``[]``
    and ``moc_max_lines`` to :data:`_DEFAULT_MOC_MAX_LINES` when absent.
    """
    block = _extract_yaml_block(text)
    if block is None:
        raise ValueError("SCHEMA.md malformed: no ```yaml fenced block found")

    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"SCHEMA.md malformed: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("SCHEMA.md malformed: yaml block is not a mapping")

    types = parsed.get("types")
    if not isinstance(types, dict) or not types:
        raise ValueError(
            "SCHEMA.md malformed: a non-empty 'types' mapping is required"
        )

    required = parsed.get("required_frontmatter")
    if required is None:
        required = []
    elif not isinstance(required, list):
        raise ValueError(
            "SCHEMA.md malformed: 'required_frontmatter' must be a list"
        )

    moc_max_lines = parsed.get("moc_max_lines", _DEFAULT_MOC_MAX_LINES)

    return Schema(
        types=types,
        required_frontmatter=[str(k) for k in required],
        moc_max_lines=int(moc_max_lines),
    )
