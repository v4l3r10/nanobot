from nanobot.utils.prompt_templates import render_template


def _render() -> str:
    return render_template("agent/memory_skill_wiki.md")


def test_template_renders_nonempty_and_static():
    out = _render()
    assert out.strip()
    assert _render() == out  # deterministic, no required kwargs


def test_directs_proactive_wiki_note_capture_and_recall():
    out = _render().lower()
    # capture mechanism
    assert "wiki_note" in out
    assert "search" in out and "create" in out and "append" in out
    # proactive + recall posture
    assert "proactив" not in out  # guard against accidental cyrillic
    assert "don't wait to be asked" in out or "do not wait to be asked" in out
    assert "search the wiki before" in out or "before answering" in out


def test_does_not_repeat_stale_legacy_memory_claims():
    """Under wiki, MEMORY.md is NOT the store and history.jsonl is not the
    knowledge surface — the wiki-on guidance must not tell the model that."""
    out = _render()
    assert "history.jsonl" not in out
    assert "memory/MEMORY.md" not in out
    # must not instruct grepping history as the memory mechanism
    assert "grep" not in out.lower()


def test_mentions_automatic_curation_contract():
    out = _render().lower()
    assert "cannot" in out and ("merge" in out or "move" in out or "delete" in out)
