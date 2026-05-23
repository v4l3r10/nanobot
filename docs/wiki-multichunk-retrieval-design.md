# Wiki Search — Multichunk (Multi-Vector, Chunk-Level) Dense Retrieval — Design

**Status:** design validated (brainstorming closed, 2026-05-23). NOT scheduled
for implementation — "design on paper" only; implement/skip to be decided later
against the success metric below.

**Goal:** Evolve the dense tier from one mean-pooled vector per page to N vectors
per page (one per chunk), matching queries against the **best chunk** (max-pool)
at rank time, so long, topically heterogeneous pages surface on the *section*
that answers the query instead of on a diluted page average.

**Date:** 2026-05-23
**Predecessor:** the hybrid BM25 + Granite-R2 dense + RRF wiki search (architecture
note [`.agent/wiki-search.md`](../.agent/wiki-search.md)) and its OOM follow-up
`b0b41281` (*chunk + mean-pool long pages to cap dense-embed memory*).
**Branch (if implemented):** TBD, base `personal/bronzo-v0.2.0`.

---

## Context — what already exists

The dense tier lives entirely in `nanobot/agent/wiki/embeddings.py`; the lexical
core + fusion live in `nanobot/agent/wiki/retrieval.py`. Today:

- **Storage** (`EmbeddingStore`, `embeddings.py:28-81`): `manifest.json` with
  `entries: [{slug, sha256, row}]` + `vectors.npy` shape `[N_pages, dim]` — exactly
  **one row per page**, row `i` ↔ entry `i` ↔ page.
- **Embedding** (`embed_texts`, `embeddings.py:180-208`): the OOM fix splits each
  page into `_CHUNK_CHARS=1200` windows, embeds the chunks, then **mean-pools +
  re-normalizes** the chunk vectors into one vector per page. Per-pass attention
  memory stays bounded; storage/retrieval see one vector per page.
- **Ranking** (`cosine_ranking` + `DenseRanker.rank`, `embeddings.py:237-263`):
  cosine of the query vector against `doc_vectors`, returns `[(relpath, rank)]`.
- **Refresh** (`refresh_embeddings`, `embeddings.py:305-360`): delta on page
  `sha256`; re-embed changed pages, reuse unchanged rows, drop deleted; rebuild
  the full matrix in `sorted(current)` order on every save.
- **Fusion** (`retrieval.py:75-140`): `bm25_ranking` (page-level) + the dense
  ranking are combined by `rrf_fuse` (rank-based RRF, `k=60`, `per_ranker_cap=50`);
  `search` returns `list[(relpath, Page)]`.

### Why this matters only for the long tail

Corpus stats: median **355** tokens, p90 **2321**, max **5184**. With
`_CHUNK_CHARS=1200` (~0.3 tok/char ⇒ ~360 tok/chunk), the **median page is a
single chunk**, where mean-pool and multichunk produce the *identical* vector.
Only the ~10% long tail (p90+) splits into multiple chunks and is the sole
segment where multichunk changes any result. The gain is real but concentrated.

---

## Decisions (validated)

| Decision | Choice | Rationale |
|---|---|---|
| Per-page representation | N chunk vectors (no pooling) | Best-chunk match beats diluted average on long pages |
| Rank-time aggregation | **Max-pool per page** (best chunk wins) | Standard multi-vector retrieval; preserves page identity |
| Manifest layout | **Page-grouped** (`entries:[{slug, sha256, chunks}]`, `layout:"chunk-v1"`) | Delta-unit (page) and manifest entry stay 1:1; only the vector array fans out |
| Row layout in `vectors.npy` | Contiguous per page, offsets via prefix-sum over `chunks` | Refresh already rebuilds the full matrix in sorted order ⇒ contiguity holds; no explicit indices to store |
| Chunk cap per page | **None** (documented knob `_MAX_CHUNKS_PER_PAGE`, default off) | Max page ≈14 chunks — irrelevant; YAGNI |
| Matching-chunk snippet | **Out of v1** (documented future extension) | Touches the `(relpath, Page)` return contract + caller `_do_search` |
| Runtime memory | Unchanged | We already embed short chunks; the OOM fix (`b0b41281`) is untouched |
| Migration | None (manifest `layout` guard ⇒ full rebuild on next Dream) | Same mechanism as the existing model/dim guard |

---

## Architecture (one line)

Keep the page as the unit of identity and the BM25 + RRF fusion contract
**unchanged**; fan out *only* the dense vectors to chunk granularity and collapse
back to a page-level `[(relpath, rank)]` via max-pool before fusion. The blast
radius is `embeddings.py` + its tests — `retrieval.py`, `rrf_fuse`, `bm25_ranking`,
the `search` signature, and the caller `_do_search` are not touched.

```
refresh (Dream)                         search
───────────────                         ──────
page → _chunk_text → embed (no pool)     query → embed (1 vec; mean-pool if long)
     → [n_chunks, dim] block                   → cosine vs ALL chunk vectors
     → append per page in sorted order         → max-pool per page (segment max)
     → vectors.npy [N_chunks, dim]              → [(relpath, rank)]  ◄── same shape
     → manifest entries {slug,sha256,chunks}          │                as today
                                                        ▼
                                          rrf_fuse([bm25_rank, dense_rank])  [UNCHANGED]
```

---

## 1. Storage / manifest

`vectors.npy` becomes `[N_chunks, dim]`. The manifest stays **one entry per page**
and carries that page's chunk count:

```json
{
  "model": "ibm-granite/granite-embedding-97m-multilingual-r2",
  "dim": 384,
  "layout": "chunk-v1",
  "entries": [
    {"slug": "concepts/foo.md", "sha256": "…", "chunks": 4},
    {"slug": "people/bar.md",   "sha256": "…", "chunks": 1}
  ]
}
```

A page's chunk rows are **contiguous** because `refresh_embeddings` rebuilds the
whole matrix in `sorted(current)` order on every save. So offsets are a prefix-sum
over `chunks` in manifest order — no explicit row indices needed:

```
offset[0] = 0;  offset[i] = offset[i-1] + entries[i-1].chunks
page i owns vectors[offset[i] : offset[i] + entries[i].chunks]
```

**Load-time guards** (`EmbeddingStore.load`):
- `manifest.layout != "chunk-v1"` (incl. the old page-level manifest with no
  `layout` key) ⇒ treat as stale ⇒ return `({}, None)` ⇒ full rebuild next Dream.
- `vectors.shape[0] != sum(e["chunks"] for e in entries)` ⇒ corrupt ⇒ rebuild.
- `vectors.shape[1] != dim` and `model`/`dim` mismatch ⇒ as today.

No migration script: the `layout` guard plus the existing model/dim guard force a
clean rebuild on the next Dream cycle.

**Size:** still tiny — the long tail adds a few rows each; a 1500-page vault with
a 10% multi-chunk tail averaging ~5 chunks is ≈ `1500*0.9 + 1500*0.1*5 ≈ 2100`
rows × 384 × 4 B ≈ 3.2 MB. Full atomic rewrite per delta stays trivial.

---

## 2. Embedding API

Add a sibling to `embed_texts` (which stays for mean-pool / back-compat):

```python
def embed_texts_chunked(texts, model_name) -> list[np.ndarray]:
    """Per input text, its [n_chunks, dim] L2-normalized chunk matrix."""
```

It reuses the existing `_chunk_text` + flatten-batch machinery but **does not
mean-pool**: it returns the chunk vectors as Granite emits them (already
unit-norm), folded back per input via the recorded chunk counts. `embed_texts`
(the mean-pool one) can be reimplemented on top of it
(`normalize(mean(chunk_matrix))`) so there is one chunking path.

**Chunk cap:** none by default. A documented knob `_MAX_CHUNKS_PER_PAGE` (e.g. 32)
could merge the tail for a pathological page; not implemented (max real page
≈14 chunks). YAGNI.

---

## 3. Retrieval (max-pool per page)

`DenseRanker` holds `chunk_vectors [N_chunks, dim]` + `chunk_slugs [N_chunks]`
(the per-row slug, built by expanding `entries` via the chunk counts) instead of
`doc_vectors` + `rels`. `rank(query)`:

1. Embed the query → **one** vector (queries are short = 1 chunk; a long query is
   mean-pooled on the query side, exactly as `embed_texts` does today).
2. Cosine of the query vector vs **all** chunk vectors → `sims [N_chunks]`.
3. **Max-pool per page**: because chunk rows are contiguous per page, segment-max
   over the offsets, e.g. `np.maximum.reduceat(sims, offsets)`, → `page_sim` per
   page.
4. Order pages by `page_sim` desc, tie-break `relpath` asc → `[(slug, rank)]`.

The output shape is **identical** to today's `cosine_ranking`, so `rrf_fuse` and
`search` are untouched. Implement as a new `chunk_max_ranking(query_vec,
chunk_vectors, offsets, slugs)`; keep `cosine_ranking` (still useful / tested).

The matching chunk is the per-page argmax — free to expose as a snippet, but that
crosses into the `(relpath, Page)` contract and `_do_search`; **deferred** (see
YAGNI).

---

## 4. Refresh

`refresh_embeddings` keeps the page-level delta (page `sha256`):

- **Changed pages** → `embed_texts_chunked` → each page's `[n_chunks, dim]` block.
- **Reused pages** → copy their old chunk block from `old_vectors`. The old
  manifest's per-page `chunks` counts (in manifest order) give the old offsets;
  slice `old_vectors[old_offset : old_offset + n]` per slug.
- **Rebuild** → for each `slug` in `sorted(current)`, append its block (new or
  reused) and record `{slug, sha256, chunks: block.shape[0]}`; `vstack` →
  `[N_chunks, dim]`; atomic-save manifest (`layout:"chunk-v1"`) + vectors.

"Re-embed a changed page = re-chunk and replace all its rows" falls out naturally
because the matrix is rebuilt block-by-block in slug order every save. Still
best-effort + under the per-vault Dream lock; never raises.

---

## 5. Testing

**Unit**
- `embed_texts_chunked`: per-page matrices aligned to inputs; every row unit-norm;
  a single-chunk input yields the same vector the mean-pool path would.
- `chunk_max_ranking`: a page whose 2nd chunk matches the query ranks **above** a
  page that would lose under page-averaging (the core behavioral win); offset/
  segment-max correctness for non-uniform chunk counts.
- `EmbeddingStore` round-trip: save chunked manifest + vectors, reload, offsets
  reconstruct the right per-page blocks; guards reject a wrong total row count and
  an old (no-`layout`) manifest ⇒ rebuild.
- `refresh_embeddings` delta (with a mocked embed): change one page ⇒ only its
  block re-embedded, the others' rows byte-identical; delete ⇒ block dropped;
  add ⇒ block appended; all rows stay contiguous + in sorted order.

**Integration / e2e**
- Gated real-dense (`pytest.importorskip("fastembed")`, like the existing hybrid
  round-trip): a long two-topic page; a query on topic B retrieves it and ranks it
  **at least as well as** the mean-pool baseline (ideally higher).
- Regression: the existing hybrid e2e and the BM25-only degradation path stay
  green; full suite against the documented Windows-host baseline (see
  `regression-gate-baseline` memory) — any other failure is a real regression.

---

## Success metric (decide implement vs skip)

Before committing the bookkeeping, measure on the real vault: build a small
`query → expected page` set biased toward long pages, score retrieval (e.g.
MRR / recall@k) for the **current mean-pool baseline** vs a multichunk prototype.
Implement only if the long-tail gain is material; otherwise keep mean-pool. The
median (single-chunk) pages are unaffected either way, so any delta comes entirely
from the long tail.

---

## Deferred / out of scope (YAGNI)

- **Matching-chunk snippet** in search output — argmax chunk is available, but it
  changes the `(relpath, Page)` contract + `_do_search`. Future extension.
- **Chunk cap per page** — knob documented, not implemented (corpus max ≈14
  chunks).
- **Explicit per-chunk row indices in the manifest** — unnecessary while refresh
  rebuilds contiguously in sorted order; revisit only if incremental in-place
  vector updates are ever introduced.
- **ColBERT-style late interaction / token-level multi-vector** — far beyond a
  personal vault's scale.
- **Changing BM25 / RRF / `search` signature** — explicitly preserved; multichunk
  is a dense-tier-internal change.
