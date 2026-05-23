"""Optional dense tier for wiki search: per-vault embedding persistence + ranker.

Imports numpy and (in a later task) fastembed — both supplied by the
``nanobot[wiki-search]`` extra. This module is imported LAZILY by
``retrieval.search`` so the always-on lexical core never pulls these deps.
"""
from __future__ import annotations

import hashlib
import json
import os
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
        if vectors.shape[0] != len(manifest.get("entries", [])):
            return {}, None  # misaligned → rebuild
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
