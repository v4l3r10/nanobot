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
from loguru import logger

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


# Granite R2 multilingual embedding models are NOT in fastembed's bundled
# registry, but they publish ONNX weights, so we register them as custom models
# on first use. Each uses CLS pooling + L2 normalization; map id -> embedding
# dim (the only per-model parameter fastembed needs for registration).
_GRANITE_CUSTOM_MODELS = {
    "ibm-granite/granite-embedding-97m-multilingual-r2": 384,
    "ibm-granite/granite-embedding-311m-multilingual-r2": 768,
}


def _ensure_custom_model_registered(cls, model_name: str) -> None:
    """Register a known Granite R2 model with fastembed if it isn't built in.

    Idempotent + best-effort: skips models already in fastembed's registry, and
    swallows a re-register race or an ``add_custom_model`` API mismatch on the
    installed fastembed version — the normal load path then decides (and raises
    → graceful BM25-only) rather than crashing here.
    """
    dim = _GRANITE_CUSTOM_MODELS.get(model_name)
    if dim is None:
        return
    try:
        already = {m.get("model") for m in cls.list_supported_models()}
    except Exception:
        already = set()
    if model_name in already:
        return
    try:
        from fastembed.common.model_description import ModelSource, PoolingType

        cls.add_custom_model(
            model=model_name,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=model_name),
            dim=dim,
            model_file="onnx/model.onnx",
        )
    except Exception:
        logger.debug("could not register custom embedding model {}", model_name)


# Granite R2 is a ModernBERT (max_position_embeddings=32768), so a single long
# page would tokenize into thousands of tokens and the global-attention layers'
# O(seq^2) activations spike RAM (a 5k-token page alone OOMs a 2 GB cgroup).
# Instead of TRUNCATING (which drops the tail of long pages and loses recall),
# we split a page into <=_CHUNK_CHARS windows, embed each, and mean-pool the
# chunk vectors into one per page — every word still contributes. _MAX_SEQ_TOKENS
# is a hard backstop on the tokenizer so a pathologically dense chunk can never
# blow up attention memory; _CHUNK_CHARS is sized to stay well under it for
# multilingual text (~0.3 tok/char here) so the backstop rarely bites.
_MAX_SEQ_TOKENS = 512
_CHUNK_CHARS = 1200
_EMBED_BATCH = 8


@lru_cache(maxsize=2)
def _get_model(model_name: str):
    cls = _import_text_embedding()
    if cls is None:
        raise RuntimeError("fastembed is not installed")
    _ensure_custom_model_registered(cls, model_name)
    model = cls(model_name=model_name)
    # Cap sequence length at the source: the tokenizer ships with max_length=
    # 32768, so without this a long chunk would still produce a huge sequence.
    # Best-effort (internal handle) — degrades to fastembed's own default.
    try:
        model.model.tokenizer.enable_truncation(max_length=_MAX_SEQ_TOKENS)
    except Exception:
        logger.debug("could not pin tokenizer truncation on {}", model_name)
    return model


def _chunk_text(text: str) -> list[str]:
    """Split into <=_CHUNK_CHARS windows. Always returns >=1 non-empty chunk
    (fastembed errors on an empty string)."""
    text = text or " "
    if len(text) <= _CHUNK_CHARS:
        return [text]
    return [text[i:i + _CHUNK_CHARS] for i in range(0, len(text), _CHUNK_CHARS)]


def embed_texts(texts, model_name: str) -> "np.ndarray":
    """Embed a list of strings → float32 [N, dim] ndarray. Requires fastembed.

    Long inputs are split into <=_CHUNK_CHARS windows, embedded in small
    batches, and mean-pooled (then L2-normalized) into one vector per input, so
    no content is dropped and per-pass attention memory stays bounded. A
    single-chunk input is unchanged (mean of one already-unit vector = itself).
    """
    model = _get_model(model_name)
    texts = list(texts)
    # Flatten every page's chunks into one stream so fastembed batches across
    # pages, then fold back per page via the recorded chunk counts.
    flat_chunks: list[str] = []
    counts: list[int] = []
    for t in texts:
        cs = _chunk_text(t)
        counts.append(len(cs))
        flat_chunks.extend(cs)
    vecs = np.array(list(model.embed(flat_chunks, batch_size=_EMBED_BATCH)), dtype="float32")
    out = np.empty((len(texts), vecs.shape[1]), dtype="float32")
    pos = 0
    for i, c in enumerate(counts):
        pooled = vecs[pos:pos + c].mean(axis=0)
        pos += c
        # Granite emits L2-normalized vectors; the mean of unit vectors is NOT
        # unit-length, so renormalize to keep cosine == dot product downstream.
        norm = float(np.linalg.norm(pooled)) or 1.0
        out[i] = pooled / norm
    return out


def warm_embedding_model(model_name: str) -> bool:
    """Download + initialize the embedding model into the local cache.

    Called at image-build (or boot) time so the Dream cycle never downloads the
    model at runtime. Goes through the same path as a real embed
    (``_get_model`` → custom-model registration + fastembed download + one
    inference), so whatever cache the runtime uses is populated identically.

    Returns True when the model loaded and produced a vector; False when
    fastembed is unavailable or the load/download failed. Never raises — the
    caller (Dockerfile/entrypoint) decides whether a miss is fatal.
    """
    if _import_text_embedding() is None:
        logger.warning("fastembed not installed; cannot warm embedding model {}", model_name)
        return False
    try:
        vec = embed_texts(["warmup"], model_name)
        ok = getattr(vec, "shape", (0,))[0] == 1
        if ok:
            logger.info("warmed embedding model {} (dim={})", model_name, vec.shape[1])
        return ok
    except Exception:
        logger.warning("failed to warm embedding model {}", model_name)
        return False


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


def load_dense_ranker(wiki_dir: Path, model: str | None = None) -> "DenseRanker | None":
    """Build a DenseRanker from persisted vectors, or None to degrade to BM25.

    ``model`` None → use whatever model the persisted manifest was built with
    (guarantees the query embeds with the same model as the docs). An EXPLICIT
    model that does not match the manifest → None (stale until next refresh).
    Also None when: fastembed unavailable, no/unreadable manifest, no vectors.
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
    manifest_model = manifest.get("model")
    if model is not None and manifest_model != model:
        return None
    use_model = model or manifest_model
    if not use_model:
        return None
    store = EmbeddingStore(Path(wiki_dir), model=use_model, dim=manifest.get("dim"))
    loaded, vectors = store.load()
    if vectors is None:
        return None
    rels = [e["slug"] for e in loaded.get("entries", [])]
    return DenseRanker(rels, vectors, use_model)


def _doc_text(page) -> str:
    # Tags are intentionally excluded here: the dense tier embeds the page's
    # semantic content (title + body). BM25 in retrieval.search() DOES include
    # tags so a keyword query can still match a page by its tags — a different,
    # complementary signal. Keep the two corpora deliberately distinct.
    return f"{page.title}\n{page.body}"


def refresh_embeddings(vault, model: str) -> None:
    """Re-embed new/changed wiki pages, drop deleted ones, persist. Best-effort.

    Caller holds the per-vault Dream lock (serialized with ingest/lint and
    wiki_note writes). No-op when fastembed is unavailable. Never raises — a
    failure here must never break the Dream cycle.
    """
    if _import_text_embedding() is None:
        return
    try:
        pages = {rel: page for rel, page in vault.iter_pages(include_cold=True)}
        current = {rel: body_hash(_doc_text(page)) for rel, page in pages.items()}

        # Peek the existing manifest (no dim needed) to compute the delta and
        # decide whether any embedding work is required at all — so an
        # up-to-date vault triggers neither a model load nor a write.
        mpath = Path(vault.wiki_dir) / ".embeddings" / _MANIFEST
        prev_manifest: dict = {}
        if mpath.is_file():
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
                if m.get("model") == model:
                    prev_manifest = m
            except (ValueError, OSError):
                prev_manifest = {}
        prev = {e["slug"]: e["sha256"] for e in prev_manifest.get("entries", [])}
        new = [rel for rel in current if prev.get(rel) != current[rel]]
        deleted = [rel for rel in prev if rel not in current]
        if not new and not deleted:
            return

        dim = prev_manifest.get("dim") or int(embed_texts(["x"], model).shape[1])
        store = EmbeddingStore(vault.wiki_dir, model=model, dim=dim)
        _, old_vectors = store.load()
        if old_vectors is None:
            old_rows: dict = {}
            to_embed = sorted(current)            # corrupt/missing → full rebuild
        else:
            old_rows = {e["slug"]: e["row"] for e in prev_manifest.get("entries", [])}
            to_embed = [rel for rel in sorted(current) if prev.get(rel) != current[rel]]

        embedded: dict = {}
        if to_embed:
            mat = embed_texts([_doc_text(pages[rel]) for rel in to_embed], model)
            embedded = {rel: mat[i] for i, rel in enumerate(to_embed)}

        rows = []
        entries = []
        for rel in sorted(current):
            vec = embedded[rel] if rel in embedded else old_vectors[old_rows[rel]]
            rows.append(vec)
            entries.append((rel, current[rel]))
        matrix = np.vstack(rows).astype("float32") if rows else np.zeros((0, dim), "float32")
        store.save(entries=entries, vectors=matrix)
    except Exception:
        logger.exception("wiki embedding refresh failed for {} (non-fatal)", vault.wiki_dir)
