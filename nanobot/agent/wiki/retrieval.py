"""Hybrid wiki search: always-on lexical core (tokenizer + Okapi BM25 + RRF).

Pure stdlib by design — this module is the guaranteed fallback and MUST NOT
import numpy/fastembed. The optional dense tier lives in
``nanobot.agent.wiki.embeddings`` and is imported lazily by ``search``.
"""
from __future__ import annotations

import math
import re
from collections import Counter

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


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, drop stopwords and 1-char tokens."""
    return [
        t for t in (m.group(0).lower() for m in _TOKEN.finditer(text))
        if len(t) > 1 and t not in _STOPWORDS
    ]


def bm25_ranking(corpus: dict[str, str], query_tokens: list[str]) -> list[tuple[str, int]]:
    """Okapi BM25 ranking. ``corpus`` maps relpath -> raw text (title+body).

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
    corpus = {rel: f"{p.title}\n{p.body}" for rel, p in pages.items()}
    rankings = [bm25_ranking(corpus, tokenize(q))]
    dense = _load_dense_ranker(vault, model)
    if dense is not None:
        try:
            rankings.append(dense.rank(q))
        except Exception:
            pass  # best-effort: a query-embed failure degrades to BM25 for this call
    fused = rrf_fuse(rankings, k_rrf=60)
    hits = [(rel, pages[rel]) for rel in fused if rel in pages]
    return hits if k is None else hits[:k]
