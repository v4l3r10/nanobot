from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.agent.wiki.retrieval import bm25_ranking, rrf_fuse, tokenize


def test_tokenize_lowercases_splits_drops_stopwords():
    assert tokenize("The Logistica E-commerce Plan") == ["logistica", "commerce", "plan"]
    assert tokenize("il piano di logistica") == ["piano", "logistica"]


def test_tokenize_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("...,;") == []


def test_bm25_ranks_exact_term_first():
    corpus = {
        "projects/logistica.md": "piano logistica ecommerce magazzino",
        "people/alice.md": "alice gestisce il marketing",
        "concepts/seo.md": "ottimizzazione motori ricerca seo",
    }
    ranking = bm25_ranking(corpus, tokenize("logistica magazzino"))
    # returns [(relpath, rank), ...] rank 1-based, only score>0, best first
    assert ranking[0][0] == "projects/logistica.md"
    assert ranking[0][1] == 1
    assert all(r[0] != "people/alice.md" for r in ranking)  # no shared terms → excluded


def test_bm25_empty_query_returns_empty():
    assert bm25_ranking({"a.md": "x"}, []) == []


def test_bm25_ties_break_by_relpath_ascending():
    # Identical bodies + identical query term → identical scores; the spec
    # requires a deterministic relpath-ascending tiebreak.
    corpus = {"b/p.md": "logistica", "a/p.md": "logistica", "c/p.md": "logistica"}
    ranking = bm25_ranking(corpus, tokenize("logistica"))
    assert [rel for rel, _ in ranking] == ["a/p.md", "b/p.md", "c/p.md"]
    assert [rank for _, rank in ranking] == [1, 2, 3]


def test_rrf_fuses_two_rankings():
    bm25 = [("a.md", 1), ("b.md", 2), ("c.md", 3)]
    dense = [("c.md", 1), ("a.md", 2), ("d.md", 3)]
    fused = rrf_fuse([bm25, dense], k_rrf=60)
    # a.md appears high in both → should win; result is ordered list of relpaths
    assert fused[0] == "a.md"
    assert set(fused) == {"a.md", "b.md", "c.md", "d.md"}


def test_rrf_degenerates_to_single_ranker():
    bm25 = [("a.md", 1), ("b.md", 2)]
    assert rrf_fuse([bm25, []], k_rrf=60) == ["a.md", "b.md"]
    assert rrf_fuse([[]], k_rrf=60) == []


def test_rrf_caps_each_input_before_fusing():
    big = [(f"{i}.md", i + 1) for i in range(100)]
    fused = rrf_fuse([big], k_rrf=60, per_ranker_cap=50)
    assert len(fused) == 50


# --- search() orchestrator (Task 5) ----------------------------------------


def _page(type, title, body):
    # Mirrors the page-creation idiom in tests/agent/wiki/test_vault.py::_page:
    # build a Page dataclass and serialize_page() it onto disk.
    return Page(type=type, title=title, status="hot",
                created="2026-05-23", updated="2026-05-23", last_touched="2026-05-23",
                tags=[], links_out=[], pinned=None, body=body)


def _write_page(vault, rel, page):
    target = vault.wiki_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_page(page), encoding="utf-8")


def _seed_vault(vault):
    _write_page(vault, "projects/logistica.md",
                _page("projects", "Logistica",
                      "piano logistica magazzino ecommerce spedizioni"))
    _write_page(vault, "people/alice.md",
                _page("people", "Alice", "alice gestisce il marketing"))


def test_search_bm25_only_when_dense_unavailable(tmp_path, monkeypatch):
    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    # Force the dense tier off regardless of environment so this is deterministic.
    monkeypatch.setattr(retrieval, "_load_dense_ranker", lambda *a, **k: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _seed_vault(vault)
    results = retrieval.search(vault, "logistica magazzino", k=20, model=None)
    assert results, "expected at least one hit"
    assert results[0][0].endswith("logistica.md")
    # results are (relpath, Page) tuples
    assert hasattr(results[0][1], "body")


def test_search_empty_query_returns_empty(tmp_path):
    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    assert retrieval.search(vault, "   ", k=20, model=None) == []


def test_search_empty_vault_returns_empty(tmp_path):
    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    assert retrieval.search(vault, "logistica", k=20, model=None) == []


def test_search_fuses_dense_when_available(tmp_path, monkeypatch):
    # Inject a fake dense ranker to prove fusion path runs and influences order.
    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _seed_vault(vault)

    class FakeDense:
        def rank(self, query):
            # rank alice first regardless, to prove dense contributes to RRF
            return [("people/alice.md", 1), ("projects/logistica.md", 2)]

    monkeypatch.setattr(retrieval, "_load_dense_ranker", lambda vault, model: FakeDense())
    results = retrieval.search(vault, "logistica magazzino", k=20, model="fake-model")
    rels = [r for r, _ in results]
    assert "people/alice.md" in rels and "projects/logistica.md" in rels
