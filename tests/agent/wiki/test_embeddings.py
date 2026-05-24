import pytest

np = pytest.importorskip("numpy")
from nanobot.agent.wiki.embeddings import EmbeddingStore, body_hash  # noqa: E402


def _store(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    return EmbeddingStore(wiki, model="m", dim=3)


def test_load_missing_returns_empty(tmp_path):
    manifest, vectors = _store(tmp_path).load()
    assert manifest == {} and vectors is None


def test_save_then_load_roundtrip(tmp_path):
    # chunk-v1: each entry carries its chunk count; here every page is 1 chunk.
    st = _store(tmp_path)
    vecs = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype="float32")
    st.save(entries=[("a.md", "h1", 1), ("b.md", "h2", 1)], vectors=vecs)
    manifest, loaded = st.load()
    assert manifest["model"] == "m" and manifest["dim"] == 3
    assert manifest["layout"] == "chunk-v1"
    assert [e["slug"] for e in manifest["entries"]] == ["a.md", "b.md"]
    assert [e["chunks"] for e in manifest["entries"]] == [1, 1]
    assert np.allclose(loaded, vecs)


def test_save_load_multichunk_roundtrip(tmp_path):
    # A 3-chunk page followed by a 1-chunk page → vectors.npy is [4, dim] and
    # the per-page chunk rows stay contiguous in manifest order.
    st = _store(tmp_path)
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype="float32")
    st.save(entries=[("long.md", "h1", 3), ("short.md", "h2", 1)], vectors=vecs)
    manifest, loaded = st.load()
    assert [e["chunks"] for e in manifest["entries"]] == [3, 1]
    assert loaded.shape == (4, 3)
    assert np.allclose(loaded, vecs)


def test_delta_detects_new_changed_deleted(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1), ("b.md", "h2", 1)],
            vectors=np.zeros((2, 3), dtype="float32"))
    manifest, _ = st.load()
    new, deleted = st.delta({"a.md": "h1", "c.md": "h9"}, manifest)  # b removed, c added, a same
    assert set(new) == {"c.md"} and set(deleted) == {"b.md"}


def test_changed_hash_is_in_new(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    manifest, _ = st.load()
    new, deleted = st.delta({"a.md": "h_CHANGED"}, manifest)
    assert new == ["a.md"] and deleted == []


def test_model_or_dim_mismatch_forces_full_rebuild(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    other = EmbeddingStore(tmp_path / "wiki", model="OTHER", dim=3)
    manifest, vectors = other.load()
    assert manifest == {} and vectors is None  # identity mismatch → treated as empty

    other_dim = EmbeddingStore(tmp_path / "wiki", model="m", dim=999)
    assert other_dim.load() == ({}, None)


def test_old_layout_manifest_is_stale(tmp_path):
    # A pre-multichunk manifest (one row per page, "row" field, NO "layout") must
    # be treated as stale → ({}, None) → clean rebuild on the next Dream. This is
    # the whole migration story: a layout guard, no migration script.
    st = _store(tmp_path)
    st.dir.mkdir(parents=True, exist_ok=True)
    import json as _j
    old = {"model": "m", "dim": 3,
           "entries": [{"slug": "a.md", "sha256": "h1", "row": 0}]}
    (st.dir / "manifest.json").write_text(_j.dumps(old), encoding="utf-8")
    np.save(st.dir / "vectors.npy", np.zeros((1, 3), dtype="float32"))
    assert st.load() == ({}, None)


def test_corrupt_manifest_returns_empty(tmp_path):
    st = _store(tmp_path)
    st.dir.mkdir(parents=True, exist_ok=True)
    (st.dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert st.load() == ({}, None)


def test_corrupt_vectors_returns_empty(tmp_path):
    # Manifest is valid but vectors.npy is binary garbage → np.load raises,
    # caught, full rebuild.
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    (st.dir / "vectors.npy").write_bytes(b"not a numpy file")
    assert st.load() == ({}, None)


def test_wrong_column_count_returns_empty(tmp_path):
    # Vectors with the wrong dim (cols) vs the configured dim → rebuild.
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    wrong = EmbeddingStore(tmp_path / "wiki", model="m", dim=3)
    # overwrite vectors with a 1x5 array but keep the dim-3 manifest/identity
    np.save(st.dir / "vectors.npy", np.zeros((1, 5), dtype="float32"))
    assert wrong.load() == ({}, None)


def test_total_chunk_rows_mismatch_returns_empty(tmp_path):
    # manifest chunk counts sum to 3 but vectors.npy has 1 row → corrupt → rebuild
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    manifest, _ = st.load()
    import json as _j
    bad = dict(manifest)
    bad["entries"] = [{"slug": "a.md", "sha256": "h1", "chunks": 3}]  # claims 3 rows
    (st.dir / "manifest.json").write_text(_j.dumps(bad), encoding="utf-8")
    assert st.load() == ({}, None)


def test_body_hash_stable_and_sensitive():
    assert body_hash("abc") == body_hash("abc")
    assert body_hash("abc") != body_hash("abd")


def test_save_leaves_no_stray_tmp_files(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    assert set(p.name for p in st.dir.iterdir()) == {"manifest.json", "vectors.npy"}


def test_cosine_ranking_orders_by_similarity():
    from nanobot.agent.wiki.embeddings import cosine_ranking
    docs = np.array([[1, 0, 0], [0, 1, 0], [0.7, 0.7, 0]], dtype="float32")
    rels = ["a.md", "b.md", "c.md"]
    qv = np.array([1, 0, 0], dtype="float32")
    ranking = cosine_ranking(qv, docs, rels)
    assert ranking[0] == ("a.md", 1)
    assert ranking[1][0] == "c.md"  # 45° closer than b (90°)
    assert [r for r, _ in ranking] == ["a.md", "c.md", "b.md"]


def test_cosine_ranking_empty():
    from nanobot.agent.wiki.embeddings import cosine_ranking
    assert cosine_ranking(np.array([1.0, 0, 0], dtype="float32"),
                          np.zeros((0, 3), dtype="float32"), []) == []


def test_cosine_ranking_handles_zero_vector_doc():
    # a zero doc vector must not raise (div-by-zero) and ranks last
    from nanobot.agent.wiki.embeddings import cosine_ranking
    docs = np.array([[1, 0, 0], [0, 0, 0]], dtype="float32")
    ranking = cosine_ranking(np.array([1, 0, 0], dtype="float32"), docs, ["a.md", "z.md"])
    assert ranking[0][0] == "a.md"


def test_load_dense_ranker_none_when_fastembed_missing(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: None)
    (tmp_path / "wiki").mkdir()
    assert emb.load_dense_ranker(tmp_path / "wiki", "m") is None


def test_load_dense_ranker_none_when_no_vectors(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    # Pretend fastembed is available so we exercise the no-vectors path.
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    (tmp_path / "wiki").mkdir()
    assert emb.load_dense_ranker(tmp_path / "wiki", "m") is None


def test_load_dense_ranker_none_when_model_mismatch(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    st = emb.EmbeddingStore(tmp_path / "wiki", model="OLD", dim=3)
    (tmp_path / "wiki").mkdir()
    st.save(entries=[("a.md", "h1", 1)], vectors=np.zeros((1, 3), dtype="float32"))
    # requesting a different model → vectors are stale → None (BM25-only until refresh)
    assert emb.load_dense_ranker(tmp_path / "wiki", "NEW") is None


def test_load_dense_ranker_builds_from_persisted(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    (tmp_path / "wiki").mkdir()
    st = emb.EmbeddingStore(tmp_path / "wiki", model="m", dim=3)
    st.save(entries=[("a.md", "h1", 1), ("b.md", "h2", 1)],
            vectors=np.array([[1, 0, 0], [0, 1, 0]], dtype="float32"))
    ranker = emb.load_dense_ranker(tmp_path / "wiki", "m")
    assert ranker is not None
    assert ranker.slugs == ["a.md", "b.md"]
    assert ranker.offsets == [0, 1]  # prefix-sum over per-page chunk counts
    # rank() needs query embedding; stub embed_texts to avoid fastembed
    monkeypatch.setattr(emb, "embed_texts",
                        lambda texts, model: np.array([[1, 0, 0]], dtype="float32"))
    assert ranker.rank("anything")[0][0] == "a.md"


def test_load_dense_ranker_multichunk_max_pools(tmp_path, monkeypatch):
    # A 2-chunk page whose SECOND chunk matches the query must win over a
    # single-chunk page that only weakly matches — end-to-end through the store,
    # offset expansion, and DenseRanker.rank (the headline multichunk behavior).
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    (tmp_path / "wiki").mkdir()
    st = emb.EmbeddingStore(tmp_path / "wiki", model="m", dim=2)
    st.save(entries=[("long.md", "h1", 2), ("short.md", "h2", 1)],
            vectors=np.array([[0.0, 1.0],        # long chunk0: orthogonal
                              [1.0, 0.0],        # long chunk1: exact match
                              [0.7, 0.714143]],  # short: weak match
                             dtype="float32"))
    ranker = emb.load_dense_ranker(tmp_path / "wiki", "m")
    assert ranker.slugs == ["long.md", "short.md"]
    assert ranker.offsets == [0, 2]
    monkeypatch.setattr(emb, "embed_texts",
                        lambda texts, model: np.array([[1.0, 0.0]], dtype="float32"))
    assert ranker.rank("q") == [("long.md", 1), ("short.md", 2)]


def test_load_dense_ranker_auto_detects_model_from_manifest(tmp_path, monkeypatch):
    # model=None → use whatever model the manifest was built with, so the query
    # is guaranteed to embed with the same model as the persisted docs.
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    (tmp_path / "wiki").mkdir()
    st = emb.EmbeddingStore(tmp_path / "wiki", model="built-with-this", dim=3)
    st.save(entries=[("a.md", "h1", 1)], vectors=np.array([[1, 0, 0]], dtype="float32"))
    ranker = emb.load_dense_ranker(tmp_path / "wiki")  # no model arg
    assert ranker is not None
    assert ranker.model == "built-with-this"
    assert ranker.slugs == ["a.md"]


@pytest.mark.slow
def test_real_fastembed_embeds(tmp_path):
    pytest.importorskip("fastembed")
    from nanobot.agent.wiki.embeddings import embed_texts
    vecs = embed_texts(["ciao mondo", "logistica"], "BAAI/bge-small-en-v1.5")
    assert vecs.shape[0] == 2 and vecs.shape[1] > 0


# --- refresh_embeddings (Task 7) --------------------------------------------
# Page-creation idiom copied from tests/agent/wiki/test_retrieval.py
# (which copied _page/serialize_page from tests/agent/wiki/test_vault.py).
from nanobot.agent.wiki.page import Page, serialize_page  # noqa: E402


def _page(type, title, body):
    return Page(type=type, title=title, status="hot",
                created="2026-05-23", updated="2026-05-23", last_touched="2026-05-23",
                tags=[], links_out=[], pinned=None, body=body)


def _write_page(vault, rel, page):
    target = vault.wiki_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_page(page), encoding="utf-8")


def _fake_chunked_factory(calls, chunks_for=None):
    """Fake ``embed_texts_chunked``: records each batch, returns one
    ``[n_chunks, 4]`` block per text. ``chunks_for(text) -> int`` controls the
    chunk count (default 1 chunk/page, so each page is one row)."""
    chunks_for = chunks_for or (lambda t: 1)

    def fake(texts, model):
        texts = list(texts)
        calls.append(texts)
        out = []
        for t in texts:
            n = chunks_for(t)
            out.append(np.array([[float(len(t)), 1.0, float(i), 0.0] for i in range(n)],
                                 dtype="float32"))
        return out
    return fake


def test_refresh_noop_without_fastembed(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    emb.refresh_embeddings(vault, "m")  # must not raise
    assert not (vault.wiki_dir / ".embeddings").exists()


def test_refresh_embeds_then_incremental(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    calls = []
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory(calls))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    # seed 2 pages (use the test_retrieval idiom): projects/a.md, projects/b.md
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "beta body"))
    emb.refresh_embeddings(vault, "m")
    manifest, vectors = emb.EmbeddingStore(vault.wiki_dir, "m", 4).load()
    assert vectors is not None and vectors.shape == (2, 4)
    assert manifest["layout"] == "chunk-v1"
    assert {e["slug"] for e in manifest["entries"]} == {"projects/a.md", "projects/b.md"}

    # second run, NO changes → no page re-embedded (early return, no embed call)
    calls.clear()
    emb.refresh_embeddings(vault, "m")
    assert calls == []  # nothing embedded; manifest unchanged
    manifest2, vectors2 = emb.EmbeddingStore(vault.wiki_dir, "m", 4).load()
    assert vectors2.shape == (2, 4)


def test_refresh_reembeds_only_changed(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    calls = []
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory(calls))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "beta body"))
    emb.refresh_embeddings(vault, "m")
    calls.clear()
    # modify projects/a.md body so its hash changes
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body CHANGED"))
    emb.refresh_embeddings(vault, "m")
    # exactly one batch, containing only the changed page's text
    assert len(calls) == 1 and len(calls[0]) == 1


def test_real_dense_refresh_then_search_roundtrip(tmp_path):
    """GATED real-dense round-trip (Task 9). SKIPS without the
    ``nanobot[wiki-search]`` extra (fastembed). Documents + locks the real
    dense path for anyone who installs the extra: refresh persists vectors, a
    second refresh is a no-op (delta empty), and ``retrieval.search`` fuses
    BM25 + dense (model auto-detected from the manifest) and surfaces the
    semantically-relevant page. Reuses the ``_page``/``_write_page`` seeding
    idiom defined above in this file (copied from test_retrieval/test_vault)."""
    pytest.importorskip("fastembed")
    from nanobot.agent.wiki import embeddings, retrieval
    from nanobot.agent.wiki.vault import Vault

    model = "BAAI/bge-small-en-v1.5"
    vault = Vault(tmp_path)
    vault.ensure_initialized()

    # Seed 3 pages with clearly distinct topics.
    _write_page(vault, "concepts/payments.md", _page(
        "concepts", "Payment Processing",
        "Credit card transactions, idempotency keys, refunds and chargebacks."))
    _write_page(vault, "people/gardener.md", _page(
        "people", "The Gardener",
        "Grows tomatoes, prunes roses, and waters the vegetable beds daily."))
    _write_page(vault, "projects/telescope.md", _page(
        "projects", "Telescope",
        "Astronomy software for tracking stars, planets and distant galaxies."))

    vectors_path = vault.wiki_dir / ".embeddings" / "vectors.npy"
    manifest_path = vault.wiki_dir / ".embeddings" / "manifest.json"

    embeddings.refresh_embeddings(vault, model)
    assert vectors_path.exists()
    assert manifest_path.exists()
    import json as _json
    count1 = len(_json.loads(manifest_path.read_text(encoding="utf-8"))["entries"])
    assert count1 == 3
    mtime1 = vectors_path.stat().st_mtime_ns

    # Second refresh: delta is empty → no-op (must not raise; stable manifest).
    embeddings.refresh_embeddings(vault, model)
    count2 = len(_json.loads(manifest_path.read_text(encoding="utf-8"))["entries"])
    assert count2 == count1
    # The early-return-on-empty-delta means vectors.npy is not rewritten.
    assert vectors_path.stat().st_mtime_ns == mtime1

    # search now fuses BM25 + dense (model auto-detected from the manifest).
    # A semantic query that shares NO surface tokens with the target page's
    # text still surfaces the gardening page via the dense tier.
    results = retrieval.search(vault, "horticulture and planting flowers")
    assert results, "real-dense fused search returned no hits"
    rels = [rel for rel, _ in results]
    assert "people/gardener.md" in rels, (
        f"the semantically-closest page was absent from the fused results: {rels}"
    )


def test_refresh_drops_deleted(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory([]))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "beta body"))
    emb.refresh_embeddings(vault, "m")
    # delete the projects/b.md file from disk
    (vault.wiki_dir / "projects" / "b.md").unlink()
    emb.refresh_embeddings(vault, "m")
    manifest, vectors = emb.EmbeddingStore(vault.wiki_dir, "m", 4).load()
    assert vectors.shape == (1, 4)
    assert {e["slug"] for e in manifest["entries"]} == {"projects/a.md"}


def test_refresh_multichunk_blocks_contiguous_and_reused(tmp_path, monkeypatch):
    # A page that chunks into 3 + a page that chunks into 1 → vectors.npy has 4
    # rows, manifest records chunks=[3,1] in sorted-slug order. Changing only the
    # 1-chunk page must REUSE the 3-chunk page's rows byte-for-byte (block reuse).
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    calls = []
    # projects/a.md body is long → 3 chunks; projects/b.md → 1 chunk.
    chunks_for = lambda t: 3 if "LONG" in t else 1  # noqa: E731
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory(calls, chunks_for))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "LONG body here"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "short"))
    emb.refresh_embeddings(vault, "m")
    manifest, vectors = emb.EmbeddingStore(vault.wiki_dir, "m", 4).load()
    by_slug = {e["slug"]: e["chunks"] for e in manifest["entries"]}
    assert by_slug == {"projects/a.md": 3, "projects/b.md": 1}
    assert vectors.shape == (4, 4)
    # a.md is sorted before b.md → its 3 rows are the first 3
    a_rows_before = vectors[:3].copy()

    calls.clear()
    _write_page(vault, "projects/b.md", _page("projects", "B", "short CHANGED"))
    emb.refresh_embeddings(vault, "m")
    # only b.md re-embedded (one batch, one page)
    assert len(calls) == 1 and len(calls[0]) == 1
    manifest2, vectors2 = emb.EmbeddingStore(vault.wiki_dir, "m", 4).load()
    assert {e["slug"]: e["chunks"] for e in manifest2["entries"]} == {
        "projects/a.md": 3, "projects/b.md": 1}
    assert vectors2.shape == (4, 4)
    # a.md's 3 rows were reused unchanged (block reuse, not re-embedded)
    assert np.array_equal(vectors2[:3], a_rows_before)


def test_refresh_rebuilds_from_old_layout_manifest(tmp_path, monkeypatch):
    # Migration: a pre-multichunk manifest (no "layout") on disk must trigger a
    # full rebuild into chunk-v1 on the next refresh, even though page hashes are
    # unchanged — the layout guard, not a hash delta, drives the rebuild.
    import json as _j

    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    calls = []
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory(calls))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    # Hand-write a stale old-layout manifest whose sha256 MATCHES the current page,
    # so only the layout guard can force the rebuild.
    h = emb.body_hash(emb._doc_text(
        next(p for rel, p in vault.iter_pages(include_cold=True) if rel == "projects/a.md")))
    embdir = vault.wiki_dir / ".embeddings"
    embdir.mkdir(parents=True, exist_ok=True)
    (embdir / "manifest.json").write_text(_j.dumps(
        {"model": "m", "dim": 4,
         "entries": [{"slug": "projects/a.md", "sha256": h, "row": 0}]}),
        encoding="utf-8")
    np.save(embdir / "vectors.npy", np.zeros((1, 4), dtype="float32"))

    emb.refresh_embeddings(vault, "m")
    # rebuilt: the page WAS re-embedded despite the matching hash (without the
    # layout guard the delta would be empty → early return → no embed call), and
    # the new manifest is chunk-v1.
    assert calls, "old-layout manifest did not force a rebuild"
    manifest, vectors = emb.EmbeddingStore(vault.wiki_dir, "m", 4).load()
    assert manifest != {} and manifest["layout"] == "chunk-v1"
    assert {e["slug"] for e in manifest["entries"]} == {"projects/a.md"}


# --- refresh_embeddings RETURN REPORT (dream visibility) --------------------
# refresh_embeddings now returns an EmbedRefreshReport so the Dream loop can log
# what it did per vault. These lock the report fields for every branch.


def test_refresh_report_fastembed_unavailable(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: None)
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    report = emb.refresh_embeddings(vault, "m")
    assert report.available is False
    assert report.changed is False
    assert report.failed is False


def test_refresh_report_on_rebuild(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory([]))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "beta body"))
    report = emb.refresh_embeddings(vault, "m")
    assert report.available is True
    assert report.changed is True
    assert report.pages == 2
    assert report.vectors == 2
    assert report.reembedded == 2
    assert report.deleted == 0


def test_refresh_report_up_to_date(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory([]))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "beta body"))
    emb.refresh_embeddings(vault, "m")
    report = emb.refresh_embeddings(vault, "m")  # second run, no changes
    assert report.available is True
    assert report.changed is False
    assert report.pages == 2
    assert report.vectors == 2
    assert report.reembedded == 0
    assert report.deleted == 0


def test_refresh_report_counts_deleted(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    monkeypatch.setattr(emb, "embed_texts_chunked", _fake_chunked_factory([]))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "alpha body"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "beta body"))
    emb.refresh_embeddings(vault, "m")
    (vault.wiki_dir / "projects" / "b.md").unlink()
    report = emb.refresh_embeddings(vault, "m")
    assert report.changed is True
    assert report.deleted == 1
    assert report.reembedded == 0
    assert report.pages == 1
    assert report.vectors == 1


def test_refresh_report_counts_multichunk_vectors(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    from nanobot.agent.wiki.vault import Vault
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    chunks_for = lambda t: 3 if "LONG" in t else 1  # noqa: E731
    monkeypatch.setattr(emb, "embed_texts_chunked",
                        _fake_chunked_factory([], chunks_for))
    vault = Vault(tmp_path)
    vault.ensure_initialized()
    _write_page(vault, "projects/a.md", _page("projects", "A", "LONG body here"))
    _write_page(vault, "projects/b.md", _page("projects", "B", "short"))
    report = emb.refresh_embeddings(vault, "m")
    assert report.pages == 2       # two pages
    assert report.vectors == 4     # 3 chunks + 1 chunk
    assert report.reembedded == 2


def _fake_model_description_module():
    import types
    mod = types.ModuleType("fastembed.common.model_description")

    class _PoolingType:
        CLS = "cls"
        MEAN = "mean"

    class _ModelSource:
        def __init__(self, hf=None):
            self.hf = hf

    mod.PoolingType = _PoolingType
    mod.ModelSource = _ModelSource
    return mod


def _install_fake_fastembed(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "fastembed", types.ModuleType("fastembed"))
    monkeypatch.setitem(sys.modules, "fastembed.common", types.ModuleType("fastembed.common"))
    monkeypatch.setitem(
        sys.modules, "fastembed.common.model_description", _fake_model_description_module()
    )


def test_ensure_registers_granite_custom_model(monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    _install_fake_fastembed(monkeypatch)
    calls = []

    class FakeCls:
        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5"}]

        @staticmethod
        def add_custom_model(**kwargs):
            calls.append(kwargs)

    emb._ensure_custom_model_registered(
        FakeCls, "ibm-granite/granite-embedding-97m-multilingual-r2"
    )
    assert len(calls) == 1
    assert calls[0]["model"] == "ibm-granite/granite-embedding-97m-multilingual-r2"
    assert calls[0]["dim"] == 384
    assert calls[0]["normalization"] is True
    assert calls[0]["model_file"] == "onnx/model.onnx"


def test_ensure_skips_when_already_supported(monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    _install_fake_fastembed(monkeypatch)
    calls = []

    class FakeCls:
        @staticmethod
        def list_supported_models():
            return [{"model": "ibm-granite/granite-embedding-311m-multilingual-r2"}]

        @staticmethod
        def add_custom_model(**kwargs):
            calls.append(kwargs)

    emb._ensure_custom_model_registered(
        FakeCls, "ibm-granite/granite-embedding-311m-multilingual-r2"
    )
    assert calls == []  # already registered → no-op


def test_ensure_noop_for_non_granite_model(monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    _install_fake_fastembed(monkeypatch)
    calls = []

    class FakeCls:
        @staticmethod
        def list_supported_models():
            raise AssertionError("should not be consulted for a built-in model")

        @staticmethod
        def add_custom_model(**kwargs):
            calls.append(kwargs)

    # Not a Granite custom id → returns immediately, registers nothing.
    emb._ensure_custom_model_registered(FakeCls, "BAAI/bge-small-en-v1.5")
    assert calls == []


def test_warm_embedding_model_false_without_fastembed(monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: None)
    assert emb.warm_embedding_model("any/model") is False


def test_warm_embedding_model_downloads_and_reports_ok(monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)
    calls = []

    def fake_embed(texts, model):
        calls.append((list(texts), model))
        return np.zeros((1, 4), dtype="float32")

    monkeypatch.setattr(emb, "embed_texts", fake_embed)
    assert emb.warm_embedding_model("some/model") is True
    assert calls == [(["warmup"], "some/model")]


def test_warm_embedding_model_false_on_failure(monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_import_text_embedding", lambda: object)

    def boom(texts, model):
        raise RuntimeError("download failed")

    monkeypatch.setattr(emb, "embed_texts", boom)
    assert emb.warm_embedding_model("some/model") is False


# --- Multichunk dense retrieval (multi-vector, chunk-level) ------------------
# Design: docs/wiki-multichunk-retrieval-design.md. The dense tier fans out to
# one vector per chunk (no mean-pool) and max-pools per page at rank time.


class _OneHotModel:
    """Fake fastembed model: each chunk -> a deterministic UNIT vector.

    One-hot at ``len(chunk) % dim`` so rows are unit-norm (like Granite) and a
    single-chunk input's chunk vector equals what the mean-pool path produces.
    """

    def __init__(self, dim=4):
        self.dim = dim

    def embed(self, chunks, batch_size=8):
        for c in chunks:
            v = np.zeros(self.dim, dtype="float32")
            v[len(c) % self.dim] = 1.0
            yield v


def test_embed_texts_chunked_returns_one_block_per_input(tmp_path, monkeypatch):
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_get_model", lambda name: _OneHotModel())
    short = "abcde"                       # 5 chars  -> 1 chunk
    long = "a" * 3000                     # 3000 chars -> 1200/1200/600 -> 3 chunks
    blocks = emb.embed_texts_chunked([short, long], "m")
    assert len(blocks) == 2
    assert blocks[0].shape == (1, 4)
    assert blocks[1].shape == (3, 4)


def test_embed_texts_chunked_does_not_pool(tmp_path, monkeypatch):
    # All rows are returned as the model emits them (no averaging): the 3-chunk
    # block keeps 3 distinct rows.
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_get_model", lambda name: _OneHotModel())
    [block] = emb.embed_texts_chunked(["a" * 3000], "m")
    assert block.shape[0] == 3
    assert np.allclose(np.linalg.norm(block, axis=1), 1.0)  # every row unit-norm


def test_single_chunk_block_equals_mean_pool_vector(monkeypatch):
    # A single-chunk input must yield the same vector under both paths, so the
    # median (single-chunk) page is unaffected by the multichunk change.
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_get_model", lambda name: _OneHotModel())
    [block] = emb.embed_texts_chunked(["abcde"], "m")
    pooled = emb.embed_texts(["abcde"], "m")
    assert np.allclose(block[0], pooled[0])


def test_embed_texts_is_mean_pool_of_chunked(monkeypatch):
    # embed_texts == L2-normalized mean of the chunked block (one chunking path).
    import nanobot.agent.wiki.embeddings as emb
    monkeypatch.setattr(emb, "_get_model", lambda name: _OneHotModel())
    [block] = emb.embed_texts_chunked(["a" * 3000], "m")
    expected = block.mean(axis=0)
    expected = expected / (float(np.linalg.norm(expected)) or 1.0)
    pooled = emb.embed_texts(["a" * 3000], "m")
    assert pooled.shape == (1, 4)
    assert np.allclose(pooled[0], expected)


def test_chunk_max_ranking_best_chunk_wins_over_average():
    # THE core behavioral win: page A's 2nd chunk matches the query exactly while
    # its 1st chunk is orthogonal. Page B is a single chunk that beats A's *average*
    # but not A's *best* chunk. Max-pool must rank A above B.
    from nanobot.agent.wiki.embeddings import chunk_max_ranking
    q = np.array([1.0, 0.0], dtype="float32")
    chunks = np.array([[0.0, 1.0],            # A chunk0: orthogonal (sim 0)
                       [1.0, 0.0],            # A chunk1: exact match (sim 1)
                       [0.7, 0.714143]],      # B chunk0: sim ~0.7
                      dtype="float32")
    offsets = [0, 2]
    slugs = ["a.md", "b.md"]
    ranking = chunk_max_ranking(q, chunks, offsets, slugs)
    assert ranking == [("a.md", 1), ("b.md", 2)]


def test_chunk_max_ranking_non_uniform_chunk_counts_and_tiebreak():
    # Two pages with equal best similarity → deterministic slug-asc tiebreak.
    # z.md has 3 chunks, a.md has 1 — segment offsets must group correctly.
    from nanobot.agent.wiki.embeddings import chunk_max_ranking
    q = np.array([1.0, 0.0], dtype="float32")
    chunks = np.array([[1.0, 0.0],            # a.md chunk0: sim 1
                       [0.0, 1.0],            # z.md chunk0: sim 0
                       [1.0, 0.0],            # z.md chunk1: sim 1
                       [0.0, 1.0]],           # z.md chunk2: sim 0
                      dtype="float32")
    offsets = [0, 1]
    slugs = ["a.md", "z.md"]
    ranking = chunk_max_ranking(q, chunks, offsets, slugs)
    assert ranking == [("a.md", 1), ("z.md", 2)]  # tie on sim=1 → a before z


def test_chunk_max_ranking_empty():
    from nanobot.agent.wiki.embeddings import chunk_max_ranking
    assert chunk_max_ranking(np.array([1.0, 0.0], dtype="float32"),
                             np.zeros((0, 2), dtype="float32"), [], []) == []


def test_chunk_max_ranking_handles_zero_chunk_vector():
    from nanobot.agent.wiki.embeddings import chunk_max_ranking
    q = np.array([1.0, 0.0], dtype="float32")
    chunks = np.array([[1.0, 0.0], [0.0, 0.0]], dtype="float32")  # 2nd is zero
    ranking = chunk_max_ranking(q, chunks, [0, 1], ["a.md", "z.md"])
    assert ranking[0][0] == "a.md"  # must not raise (div-by-zero), zero ranks last
