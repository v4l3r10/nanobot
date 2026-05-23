# Wiki Search — Hybrid BM25 + Dense + RRF

## Why

`wiki_note operation="search"` used to rank by field-presence substring scoring
(title 3 / tag 2 / body 1) and matched the **whole query string** as one needle
— so a multi-word query like `"logistica magazzino"` only hit a page containing
that exact phrase. The model couldn't find a page by topic when the wording
differed. This replaces the substring ranker with a hybrid lexical + semantic
ranker so pages surface by relevance, not exact substring.

## Architecture

Two modules with a strict dependency boundary:

```
   wiki_note._do_search(q)                      (live tool, query time)
        │  empty→recency / "tag:"→filter UNCHANGED
        │  keyword → retrieval.search(vault, q)
        ▼
   retrieval.py  (ALWAYS ON, pure stdlib — NO numpy/fastembed)
     tokenize · bm25_ranking (Okapi) · rrf_fuse (k=60) · search()
        │   corpus = title + tags + body
        └── _load_dense_ranker(vault)  ── lazy import ──┐
                                                         ▼
   embeddings.py  (OPTIONAL — numpy + fastembed, nanobot[wiki-search] extra)
     EmbeddingStore · DenseRanker · load_dense_ranker · refresh_embeddings
```

`retrieval.py` MUST stay import-light: it imports `embeddings` only *inside*
`_load_dense_ranker` (guarded), so importing `wiki_note`/`retrieval` never pulls
numpy or fastembed. A subprocess test
(`tests/agent/wiki/test_retrieval.py::test_importing_wiki_note_does_not_pull_numpy_or_fastembed`)
locks this invariant.

## Two data paths (must agree)

- **Write (Dream)** — `memory.py`, inside `get_vault_lock(slug)` after
  `run_lint`, calls `refresh_embeddings(vault, model)` (gated by
  `self.wiki_embeddings AND self.wiki_embedding_model`). Incremental: content
  hash over `title\nbody`; re-embeds only new/changed pages, drops deleted,
  full-rebuilds on a corrupt store. Persists `<vault>/wiki/.embeddings/`
  (`vectors.npy` via `os.replace` + `manifest.json` via `atomic_write_text`).
- **Read (query)** — `_do_search` → `retrieval.search` → BM25 always; dense
  ranker auto-detects its model **from the persisted manifest** (so the query
  embeds with the SAME model as the docs — no config threading to the tool).
  Fused via RRF; `search(k=None)` returns all candidates so `_do_search` keeps
  its `_SEARCH_CAP` + "more not shown" overflow.

Identity key everywhere is the POSIX relpath (e.g. `people/alice.md`).

## Corpus split (intentional)

- BM25 corpus = `title + tags + body` (tags kept so keyword search still
  surfaces a page by its tags, as the old substring ranker did).
- Dense embed text (`_doc_text`) = `title + body` only (tags are short labels;
  the dense tier captures semantic content). Different, complementary signals.

## Graceful degradation

fastembed absent / flag off / model unset / model load fails / corrupt vectors
→ every dense entry point returns `None`/no-op and RRF degenerates to BM25-only
through the same code path. A stock install (flag off) never imports numpy in
the Dream path. BM25 (in-house Okapi, no `rank_bm25` dep) is the guaranteed
always-available tier.

## Embedding model (Granite R2) + fastembed

Default `wiki_embedding_model = ibm-granite/granite-embedding-97m-multilingual-r2`
(384-dim; 311M variant is 768-dim). Granite R2 needs **no** query/passage
instruction prefix — query and document text are embedded symmetrically (unlike
E5). CLS pooling, L2-normalized (cosine normalizes internally anyway).

fastembed does **not** ship Granite in its built-in registry, so `_get_model`
registers the known multilingual R2 variants as custom ONNX models on first use
(`add_custom_model`: `pooling=CLS`, `normalization=True`, `dim`,
`sources=ModelSource(hf=...)`, `model_file="onnx/model.onnx"`). Best-effort: a
re-register race or an `add_custom_model` API change on the installed fastembed
version falls through to the normal load → graceful BM25-only.

## Config

`DreamConfig.wiki_embeddings: bool=False`, `wiki_embedding_model: str=<granite>`
(`nanobot/config/schema.py`). Threaded onto the `Dream` object post-construction
in `cli/commands.py` (same attribute-assignment pattern as `wiki_enabled`).
Optional dependency: `pip install nanobot[wiki-search]` (fastembed + numpy).

## Deployment — model pre-bake

To stop fastembed from downloading the model *during the Dream cycle* (runtime
latency + network dependency), the Docker image **pre-bakes** it: the Dockerfile
installs `.[wiki-search]` and runs `embeddings.warm_embedding_model(<id>)` at
build. fastembed caches models under `FASTEMBED_CACHE_PATH` (it IGNORES
`HF_HOME`; default `/tmp/fastembed_cache`), so the Dockerfile pins
`FASTEMBED_CACHE_PATH=/home/nanobot/.cache/fastembed` — under `$HOME/.cache`,
NOT the `~/.nanobot` volume, so the baked layer (~397 MB for 97M) is what the
runtime reads and the volume mount can't shadow it. Validated: a
`--network none` container embeds with no download (cache hit). Pinned to the
DreamConfig default; override with `--build-arg WIKI_EMBEDDING_MODEL=<id>` (must
match config). An HF token is optional (Granite is public) — pass it as a
BuildKit secret, never an ARG, so it isn't recorded in `docker history`:
`docker build --secret id=hf_token,env=HF_TOKEN ...`. FAIL-FAST: the build
exits non-zero if the model can't be baked, so an image that would download at
runtime is never shipped (retry on transient HF outage). The 1 CPU / 1 GB
container favours 97M (384-dim).
