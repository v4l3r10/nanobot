"""Optional dense tier for wiki search: per-vault embedding persistence + ranker.

Imports numpy and (in a later task) fastembed — both supplied by the
``nanobot[wiki-search]`` extra. This module is imported LAZILY by
``retrieval.search`` so the always-on lexical core never pulls these deps.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from nanobot.utils.atomic import atomic_write_text

_MANIFEST = "manifest.json"
_VECTORS = "vectors.npy"


def body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingStore:
    """Persists [N, dim] float32 doc vectors + a content-hash manifest."""

    def __init__(self, wiki_dir: Path, model: str, dim: int) -> None:
        self.dir = Path(wiki_dir) / ".embeddings"
        self.model = model
        self.dim = dim

    def load(self) -> tuple[dict, "np.ndarray | None"]:
        mpath, vpath = self.dir / _MANIFEST, self.dir / _VECTORS
        if not mpath.is_file() or not vpath.is_file():
            return {}, None
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}, None
        # Model/dim identity guard: stale vectors of the wrong shape are useless.
        if manifest.get("model") != self.model or manifest.get("dim") != self.dim:
            return {}, None
        try:
            vectors = np.load(vpath)
        except (ValueError, OSError):
            return {}, None
        # Shape guard: row count must match the manifest, and the column count
        # must match the configured dim. A partial overwrite that left a
        # wrong-shaped array on disk is treated as corrupt → full rebuild.
        if vectors.ndim != 2 or vectors.shape[0] != len(manifest.get("entries", [])):
            return {}, None
        if vectors.shape[1] != self.dim:
            return {}, None
        return manifest, vectors

    def delta(self, current: dict[str, str], manifest: dict) -> tuple[list[str], list[str]]:
        prev = {e["slug"]: e["sha256"] for e in manifest.get("entries", [])}
        new = [rel for rel, h in current.items() if prev.get(rel) != h]
        deleted = [rel for rel in prev if rel not in current]
        return new, deleted

    def save(self, entries: list[tuple[str, str]], vectors: "np.ndarray") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model": self.model,
            "dim": self.dim,
            "entries": [{"slug": rel, "sha256": h, "row": i}
                        for i, (rel, h) in enumerate(entries)],
        }
        # np.save appends ".npy" to any path not already ending in ".npy"
        # (true even when the path has a different extension like ".tmp"),
        # so use a ".npy"-suffixed temp name it will write verbatim.
        tmp = self.dir / (_VECTORS + ".tmp.npy")
        np.save(tmp, vectors.astype("float32"))
        os.replace(tmp, self.dir / _VECTORS)
        atomic_write_text(self.dir / _MANIFEST,
                          json.dumps(manifest, ensure_ascii=False, indent=2))


def _import_text_embedding():
    """Return fastembed's TextEmbedding class, or None if unavailable.

    Isolated + monkeypatchable so the dense tier degrades gracefully (callers
    fall back to BM25-only) when the optional ``nanobot[wiki-search]`` extra
    is not installed.
    """
    try:
        from fastembed import TextEmbedding
    except Exception:
        return None
    return TextEmbedding


@lru_cache(maxsize=2)
def _get_model(model_name: str):
    cls = _import_text_embedding()
    if cls is None:
        raise RuntimeError("fastembed is not installed")
    return cls(model_name=model_name)


def embed_texts(texts, model_name: str) -> "np.ndarray":
    """Embed a list of strings → float32 [N, dim] ndarray. Requires fastembed."""
    model = _get_model(model_name)
    return np.array(list(model.embed(list(texts))), dtype="float32")


def cosine_ranking(query_vec, doc_vectors, rels) -> list[tuple[str, int]]:
    """Rank docs by cosine similarity to the query. Returns [(rel, rank), ...]
    1-based, best first, deterministic relpath-asc tiebreak. Zero-norm vectors
    are handled without dividing by zero."""
    if doc_vectors is None or len(rels) == 0:
        return []
    q = query_vec.astype("float32")
    qn = q / (float(np.linalg.norm(q)) or 1.0)
    docs = doc_vectors.astype("float32")
    norms = np.linalg.norm(docs, axis=1)
    norms[norms == 0] = 1.0
    sims = (docs / norms[:, None]) @ qn
    order = sorted(range(len(rels)), key=lambda i: (-float(sims[i]), rels[i]))
    return [(rels[i], rank + 1) for rank, i in enumerate(order)]


class DenseRanker:
    """Holds persisted doc vectors + the model name; embeds queries on demand."""

    def __init__(self, rels: list[str], doc_vectors: "np.ndarray", model: str) -> None:
        self.rels = rels
        self.doc_vectors = doc_vectors
        self.model = model

    def rank(self, query: str) -> list[tuple[str, int]]:
        qv = embed_texts([query], self.model)[0]
        return cosine_ranking(qv, self.doc_vectors, self.rels)


def load_dense_ranker(wiki_dir: Path, model: str) -> "DenseRanker | None":
    """Build a DenseRanker from persisted vectors, or None to degrade to BM25.

    Returns None when: fastembed is unavailable, no persisted vectors exist,
    the manifest is unreadable, or the persisted vectors were built for a
    DIFFERENT model (stale until the next Dream refresh re-embeds).
    """
    if _import_text_embedding() is None:
        return None
    mpath = Path(wiki_dir) / ".embeddings" / _MANIFEST
    if not mpath.is_file():
        return None
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if manifest.get("model") != model:
        return None
    store = EmbeddingStore(Path(wiki_dir), model=model, dim=manifest.get("dim"))
    loaded, vectors = store.load()
    if vectors is None:
        return None
    rels = [e["slug"] for e in loaded.get("entries", [])]
    return DenseRanker(rels, vectors, model)
