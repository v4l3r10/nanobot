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
    st = _store(tmp_path)
    vecs = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype="float32")
    st.save(entries=[("a.md", "h1"), ("b.md", "h2")], vectors=vecs)
    manifest, loaded = st.load()
    assert manifest["model"] == "m" and manifest["dim"] == 3
    assert [e["slug"] for e in manifest["entries"]] == ["a.md", "b.md"]
    assert np.allclose(loaded, vecs)


def test_delta_detects_new_changed_deleted(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1"), ("b.md", "h2")],
            vectors=np.zeros((2, 3), dtype="float32"))
    manifest, _ = st.load()
    new, deleted = st.delta({"a.md": "h1", "c.md": "h9"}, manifest)  # b removed, c added, a same
    assert set(new) == {"c.md"} and set(deleted) == {"b.md"}


def test_changed_hash_is_in_new(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1")], vectors=np.zeros((1, 3), dtype="float32"))
    manifest, _ = st.load()
    new, deleted = st.delta({"a.md": "h_CHANGED"}, manifest)
    assert new == ["a.md"] and deleted == []


def test_model_or_dim_mismatch_forces_full_rebuild(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1")], vectors=np.zeros((1, 3), dtype="float32"))
    other = EmbeddingStore(tmp_path / "wiki", model="OTHER", dim=3)
    manifest, vectors = other.load()
    assert manifest == {} and vectors is None  # identity mismatch → treated as empty

    other_dim = EmbeddingStore(tmp_path / "wiki", model="m", dim=999)
    assert other_dim.load() == ({}, None)


def test_corrupt_manifest_returns_empty(tmp_path):
    st = _store(tmp_path)
    st.dir.mkdir(parents=True, exist_ok=True)
    (st.dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert st.load() == ({}, None)


def test_corrupt_vectors_returns_empty(tmp_path):
    # Manifest is valid but vectors.npy is binary garbage → np.load raises,
    # caught, full rebuild.
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1")], vectors=np.zeros((1, 3), dtype="float32"))
    (st.dir / "vectors.npy").write_bytes(b"not a numpy file")
    assert st.load() == ({}, None)


def test_wrong_column_count_returns_empty(tmp_path):
    # Vectors with the wrong dim (cols) vs the configured dim → rebuild.
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1")], vectors=np.zeros((1, 3), dtype="float32"))
    wrong = EmbeddingStore(tmp_path / "wiki", model="m", dim=3)
    # overwrite vectors with a 1x5 array but keep the dim-3 manifest/identity
    np.save(st.dir / "vectors.npy", np.zeros((1, 5), dtype="float32"))
    assert wrong.load() == ({}, None)


def test_misaligned_rows_returns_empty(tmp_path):
    # manifest says 2 entries but vectors.npy has 1 row → rebuild
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1")], vectors=np.zeros((1, 3), dtype="float32"))
    manifest, _ = st.load()
    # tamper: rewrite manifest to claim 2 entries
    import json as _j
    bad = dict(manifest)
    bad["entries"] = manifest["entries"] + [{"slug": "b.md", "sha256": "h2", "row": 1}]
    (st.dir / "manifest.json").write_text(_j.dumps(bad), encoding="utf-8")
    assert st.load() == ({}, None)


def test_body_hash_stable_and_sensitive():
    assert body_hash("abc") == body_hash("abc")
    assert body_hash("abc") != body_hash("abd")


def test_save_leaves_no_stray_tmp_files(tmp_path):
    st = _store(tmp_path)
    st.save(entries=[("a.md", "h1")], vectors=np.zeros((1, 3), dtype="float32"))
    assert set(p.name for p in st.dir.iterdir()) == {"manifest.json", "vectors.npy"}
