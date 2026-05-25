from nanobot.agent.wiki.page import Page, serialize_page
from nanobot.agent.wiki.retrieval import (
    _graph_ranking,
    bm25_ranking,
    rrf_fuse,
    tokenize,
)


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


# --- _graph_ranking ---

def test_graph_ranking_counts_then_seed_rank_then_relpath():
    adjacency = {
        "s0": {"a", "b"},
        "s1": {"a"},
        "a": {"s0", "s1"},
        "b": {"s0"},
    }
    # "a" is reached by both seeds (count 2), "b" by one (count 1) -> a first.
    assert _graph_ranking(adjacency, ["s0", "s1"]) == [("a", 1), ("b", 2)]


def test_graph_ranking_excludes_seeds():
    adjacency = {"s0": {"s1", "x"}, "s1": {"s0"}, "x": {"s0"}}
    # s1 is itself a seed -> excluded; only the non-seed neighbor x remains.
    assert _graph_ranking(adjacency, ["s0", "s1"]) == [("x", 1)]


def test_graph_ranking_tiebreak_best_seed_then_relpath():
    # equal count (1 each): "m" via s0 (rank 0) beats "z" via s1 (rank 1).
    adj1 = {"s0": {"m"}, "s1": {"z"}, "m": {"s0"}, "z": {"s1"}}
    assert _graph_ranking(adj1, ["s0", "s1"]) == [("m", 1), ("z", 2)]
    # equal count AND equal best-seed-rank -> relpath ascending.
    adj2 = {"s0": {"beta", "alpha"}, "alpha": {"s0"}, "beta": {"s0"}}
    assert _graph_ranking(adj2, ["s0"]) == [("alpha", 1), ("beta", 2)]


def test_graph_ranking_empty_when_no_neighbors():
    assert _graph_ranking({}, ["s0", "s1"]) == []


# --- search() orchestrator (Task 5) ----------------------------------------


def _page(type, title, body, links_out=None):
    # Mirrors the page-creation idiom in tests/agent/wiki/test_vault.py::_page:
    # build a Page dataclass and serialize_page() it onto disk.
    return Page(type=type, title=title, status="hot",
                created="2026-05-23", updated="2026-05-23", last_touched="2026-05-23",
                tags=[], links_out=list(links_out or []), pinned=None, body=body)


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


def test_search_graph_boost_surfaces_linked_neighbor(tmp_path, monkeypatch):
    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(retrieval, "_load_dense_ranker", lambda *a, **k: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    # payment-svc's body matches "gateway billing"; alice's body does NOT,
    # but alice links to payment-svc -> she is surfaced ONLY via the graph.
    _write_page(vault, "projects/payment-svc.md",
                _page("projects", "Payment Service", "payment gateway billing stripe"))
    _write_page(vault, "people/alice.md",
                _page("people", "Alice", "alice runs the team",
                      links_out=["projects/payment-svc"]))
    rels = [r for r, _ in retrieval.search(vault, "gateway billing", k=20, model=None)]
    assert "projects/payment-svc.md" in rels   # direct lexical hit (the seed)
    assert "people/alice.md" in rels           # pulled in by the link graph


def test_search_no_links_identical_to_base_ordering(tmp_path, monkeypatch):
    # Non-degradation: with no links, graph ranker is empty and search() output
    # order equals the pre-graph base fusion exactly.
    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(retrieval, "_load_dense_ranker", lambda *a, **k: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _seed_vault(vault)  # pages have empty links_out
    q = "logistica magazzino"
    got = [r for r, _ in retrieval.search(vault, q, k=20, model=None)]
    pages = {rel: p for rel, p in vault.iter_pages(include_cold=True)}
    corpus = {
        rel: "\n".join((p.title, " ".join(p.tags), p.body)) for rel, p in pages.items()
    }
    base = rrf_fuse([bm25_ranking(corpus, tokenize(q))], k_rrf=60)
    assert got == base


def test_search_logs_graph_segment(tmp_path, monkeypatch):
    from loguru import logger

    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(retrieval, "_load_dense_ranker", lambda *a, **k: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _seed_vault(vault)
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        retrieval.search(vault, "logistica magazzino", k=20, model=None)
    finally:
        logger.remove(sink_id)
    line = next((m for m in captured if "wiki search" in m), "")
    assert "graph=" in line


# --- search layer-breakdown logging (visibility) ----------------------------


def test_search_logs_layer_breakdown_with_dense(tmp_path, monkeypatch):
    from loguru import logger

    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _seed_vault(vault)

    class FakeDense:
        def rank(self, query):
            return [("people/alice.md", 1), ("projects/logistica.md", 2)]

    monkeypatch.setattr(retrieval, "_load_dense_ranker",
                        lambda vault, model: FakeDense())
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        retrieval.search(vault, "logistica magazzino", k=20, model="fake-model")
    finally:
        logger.remove(sink_id)
    line = next((m for m in captured if "wiki search" in m), "")
    assert line, captured
    assert "bm25=" in line
    assert "dense=2" in line          # FakeDense returned 2 candidates
    assert "fused" in line
    assert "top=" in line
    assert "ms" in line               # per-layer latency present


def test_search_logs_dense_inactive(tmp_path, monkeypatch):
    from loguru import logger

    from nanobot.agent.wiki import retrieval
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(retrieval, "_load_dense_ranker", lambda *a, **k: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _seed_vault(vault)
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        retrieval.search(vault, "logistica magazzino", k=20, model=None)
    finally:
        logger.remove(sink_id)
    line = next((m for m in captured if "wiki search" in m), "")
    assert line, captured
    assert "dense=inactive" in line
    assert "bm25=" in line


def test_importing_wiki_note_does_not_pull_numpy_or_fastembed():
    # Architectural invariant: the always-on lexical path must never import the
    # optional dense deps (numpy/fastembed). Run in a fresh interpreter so the
    # assertion is not contaminated by modules other tests already imported.
    import subprocess
    import sys

    code = (
        "import sys, nanobot.agent.tools.wiki_note; "
        "assert 'numpy' not in sys.modules, 'numpy leaked into the lexical path'; "
        "assert 'fastembed' not in sys.modules, 'fastembed leaked into the lexical path'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
