from nanobot.agent.wiki.moc_refresh import mark_vault_dirty, take_dirty


def test_take_returns_false_when_unmarked():
    assert take_dirty("never-marked-slug") is False


def test_mark_then_take_is_true_once_then_false():
    mark_vault_dirty("vault-a")
    assert take_dirty("vault-a") is True
    assert take_dirty("vault-a") is False  # cleared by the take


def test_marks_are_per_slug_independent():
    mark_vault_dirty("vault-x")
    assert take_dirty("vault-y") is False
    assert take_dirty("vault-x") is True
