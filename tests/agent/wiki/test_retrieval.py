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
