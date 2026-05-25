"""Hybrid wiki search: always-on lexical core (tokenizer + Okapi BM25 + RRF).

Pure stdlib by design — this module is the guaranteed fallback and MUST NOT
import numpy/fastembed. The optional dense tier lives in
``nanobot.agent.wiki.embeddings`` and is imported lazily by ``search``.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

from loguru import logger

# Minimal IT/EN stopwords — just the highest-frequency function words that add
# noise to BM25. Deliberately small (YAGNI): a full list is overkill for a
# personal vault and risks dropping meaningful tokens.
_STOPWORDS = frozenset(
    "a an and the of to in on for is are be it this that with as at by "
    "il lo la i gli le un uno una di a da in con su per tra fra e o ma "
    "che non si come".split()
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)  # runs of word-chars, drops _ and punct

_K1 = 1.5
_B = 0.75

# Graph-boost: how many top pre-fused hits seed the 1-hop link expansion.
_GRAPH_SEEDS = 5


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, drop stopwords and 1-char tokens."""
    return [
        t for t in (m.group(0).lower() for m in _TOKEN.finditer(text))
        if len(t) > 1 and t not in _STOPWORDS
    ]


def bm25_ranking(corpus: dict[str, str], query_tokens: list[str]) -> list[tuple[str, int]]:
    """Okapi BM25 ranking. ``corpus`` maps relpath -> raw text (caller-defined;
    ``search()`` uses title + tags + body).

    Returns ``[(relpath, rank), ...]`` 1-based, score>0 only, best first,
    ties broken by relpath asc for determinism.
    """
    if not query_tokens or not corpus:
        return []
    docs = {rel: tokenize(text) for rel, text in corpus.items()}
    n = len(docs)
    avgdl = sum(len(toks) for toks in docs.values()) / n if n else 0.0
    df: Counter[str] = Counter()
    for toks in docs.values():
        for term in set(toks):
            df[term] += 1
    idf = {
        term: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5))
        for term, dfi in df.items()
    }
    scores: dict[str, float] = {}
    q_terms = set(query_tokens)
    for rel, toks in docs.items():
        if not toks:
            continue
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + _K1 * (1 - _B + _B * dl / avgdl)
            s += idf[term] * (f * (_K1 + 1)) / denom
        if s > 0:
            scores[rel] = s
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(rel, i + 1) for i, (rel, _) in enumerate(ordered)]


def rrf_fuse(
    rankings: list[list[tuple[str, int]]],
    k_rrf: int = 60,
    per_ranker_cap: int = 50,
) -> list[str]:
    """Reciprocal Rank Fusion over rank lists. Returns relpaths, best first.

    Fuses on ranks (not raw scores) so an unbounded BM25 score and a bounded
    cosine combine without normalization. Empty rankings contribute nothing,
    so the same code degrades to single-ranker (or zero-ranker) cleanly.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rel, rank in ranking[:per_ranker_cap]:
            scores[rel] = scores.get(rel, 0.0) + 1.0 / (k_rrf + rank)
    # Tie-break by relpath asc for determinism.
    return [rel for rel, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _graph_ranking(
    adjacency: dict[str, set[str]], seeds: list[str]
) -> list[tuple[str, int]]:
    """Rank the 1-hop neighbors of ``seeds`` for RRF (best first, 1-based).

    ``adjacency`` is the undirected relpath graph (``links.build_adjacency``);
    ``seeds`` are the top pre-fused hit relpaths in rank order. Neighbors that
    are themselves seeds are excluded (already ranked by bm25/dense). Order:
    seed-adjacency count DESC, then best (lowest) seed rank ASC, then relpath
    ASC — fully deterministic. Empty when no neighbors.
    """
    seed_rank = {rel: i for i, rel in enumerate(seeds)}
    seed_set = set(seeds)
    count: Counter[str] = Counter()
    best: dict[str, int] = {}
    for s in seeds:
        for nb in adjacency.get(s, ()):
            if nb in seed_set:
                continue
            count[nb] += 1
            r = seed_rank[s]
            if nb not in best or r < best[nb]:
                best[nb] = r
    ordered = sorted(count, key=lambda nb: (-count[nb], best[nb], nb))
    return [(nb, i + 1) for i, nb in enumerate(ordered)]


def _load_dense_ranker(vault, model=None):
    """Lazily load the optional dense ranker; None when unavailable.

    Import is INSIDE this function so importing this module never pulls
    numpy/fastembed. ``model=None`` → auto-detect from the persisted manifest.
    """
    try:
        from nanobot.agent.wiki.embeddings import load_dense_ranker
    except Exception:
        return None
    try:
        return load_dense_ranker(vault.wiki_dir, model)
    except Exception:
        return None


def search(vault, query, k=None, model=None):
    """Hybrid ranked search → ordered ``list[(relpath, Page)]`` (best first).

    ``model`` None / dense unavailable → pure BM25 via the same RRF path; the
    dense tier auto-detects its model from the persisted manifest. ``k`` None
    returns ALL fused candidates (the caller applies its own cap + overflow);
    an explicit ``k`` truncates. The caller (``_do_search``) owns the
    empty-query and ``tag:`` branches; ``search`` is only the keyword branch
    and returns ``[]`` for an empty query or an empty vault.
    """
    q = (query or "").strip()
    if not q:
        return []
    pages = {rel: page for rel, page in vault.iter_pages(include_cold=True)}
    if not pages:
        return []
    # BM25 corpus = title + tags + body, so a keyword query can still surface a
    # page via its tags (the pre-BM25 substring ranker scored tag matches too).
    corpus = {
        rel: "\n".join((p.title, " ".join(p.tags), p.body)) for rel, p in pages.items()
    }
    t0 = time.perf_counter()
    bm25 = bm25_ranking(corpus, tokenize(q))
    t_bm25 = time.perf_counter() - t0
    rankings = [bm25]

    # dense_n is None while the tier is inactive (fastembed missing / no
    # embeddings built / query-embed failure) — logged as "inactive" so a
    # silent BM25-only degrade is visible.
    dense_n: int | None = None
    t1 = time.perf_counter()
    dense = _load_dense_ranker(vault, model)
    if dense is not None:
        try:
            dense_rank = dense.rank(q)
            rankings.append(dense_rank)
            dense_n = len(dense_rank)
        except Exception:
            pass  # best-effort: a query-embed failure degrades to BM25 for this call
    t_dense = time.perf_counter() - t1

    t2 = time.perf_counter()
    fused = rrf_fuse(rankings, k_rrf=60)
    t_fuse = time.perf_counter() - t2

    hits = [(rel, pages[rel]) for rel in fused if rel in pages]

    def _ms(s: float) -> str:
        return f"{s * 1000:.0f}ms"

    dense_part = (
        f" + dense={dense_n} ({_ms(t_dense)})" if dense_n is not None
        else ", dense=inactive"
    )
    msg = (
        f"wiki search {q[:80]!r} : bm25={len(bm25)} ({_ms(t_bm25)})"
        f"{dense_part} → fused {len(fused)} ({_ms(t_fuse)}), "
        f"top={fused[0] if fused else 'none'}"
    )
    logger.info("{}", msg)

    return hits if k is None else hits[:k]
