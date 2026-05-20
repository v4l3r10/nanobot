# Wiki Schema
```yaml
types:
  people:    { folder: people,    cold_after_days: 180 }
  projects:  { folder: projects,  cold_after_days: 90 }
  concepts:  { folder: concepts,  cold_after_days: 365 }
  decisions: { folder: decisions, cold_after_days: null }
  inbox:     { folder: inbox,     cold_after_days: 30 }
required_frontmatter: [type, title, status, created, updated, last_touched]
moc_max_lines: 120
```
