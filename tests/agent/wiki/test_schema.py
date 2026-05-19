import pytest
from nanobot.agent.wiki.schema import load_schema, Schema

SAMPLE = '''# Wiki Schema
```yaml
types:
  people:    { folder: people,    cold_after_days: 180 }
  projects:  { folder: projects,  cold_after_days: 90 }
  concepts:  { folder: concepts,  cold_after_days: 365 }
  decisions: { folder: decisions, cold_after_days: null }
required_frontmatter: [type, title, status, created, updated, last_touched]
moc_max_lines: 120
```
'''

def test_parses_types_and_decay():
    s = load_schema(SAMPLE)
    assert s.cold_after_days("people") == 180
    assert s.cold_after_days("decisions") is None  # never decays
    assert s.folder("projects") == "projects"

def test_unknown_type_is_invalid():
    s = load_schema(SAMPLE)
    assert not s.is_known_type("aliens")
    assert s.is_known_type("people")

def test_validate_page_frontmatter_missing_field():
    s = load_schema(SAMPLE)
    errs = s.validate_frontmatter({"type": "people", "title": "A"})
    assert any("status" in e for e in errs)

def test_rejects_schema_without_types():
    with pytest.raises(ValueError):
        load_schema("```yaml\nmoc_max_lines: 10\n```")
