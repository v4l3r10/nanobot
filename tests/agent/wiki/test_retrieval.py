from nanobot.agent.wiki.retrieval import bm25_ranking, tokenize


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
