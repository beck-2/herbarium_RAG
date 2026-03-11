"""
Tests for src/index/: build (FAISS IVF-PQ), bundle (packing utilities).
Phase 0: import stubs.
Phase 5: functional tests on synthetic embeddings — no real images required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import / API surface (Phase 0)
# ---------------------------------------------------------------------------

def test_build_module_importable():
    from src.index import build
    assert hasattr(build, "encode_specimens")
    assert hasattr(build, "build_ivfpq_index")
    assert hasattr(build, "build_family_subindexes")
    assert hasattr(build, "verify_recall")
    assert hasattr(build, "save_index")
    assert hasattr(build, "load_index")


def test_bundle_module_importable():
    from src.index import bundle
    assert hasattr(bundle, "pack_bundle")
    assert hasattr(bundle, "generate_thumbnails")
    assert hasattr(bundle, "create_specimens_db")
    assert hasattr(bundle, "check_bundle_size")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embeddings(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    """Random float32 embeddings with norm < 1 (mimics Poincaré ball output)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms * 0.9  # inside unit ball


# ---------------------------------------------------------------------------
# build_ivfpq_index
# ---------------------------------------------------------------------------

def test_build_ivfpq_index_trains_and_searches():
    # IVF-PQ with bits_per_code=8 needs >= 256 training vectors (2^8 PQ centroids).
    from src.index.build import build_ivfpq_index
    emb = _random_embeddings(320, d=32)
    index = build_ivfpq_index(emb, n_clusters=8, n_subquantizers=4, bits_per_code=8, nprobe=4)
    D, I = index.search(emb[:10].copy(), k=5)
    assert D.shape == (10, 5)
    assert I.shape == (10, 5)
    assert index.ntotal == 320


def test_build_ivfpq_index_nprobe_set():
    from src.index.build import build_ivfpq_index
    emb = _random_embeddings(320, d=32)
    index = build_ivfpq_index(emb, n_clusters=8, n_subquantizers=4, bits_per_code=8, nprobe=6)
    assert index.nprobe == 6


def test_build_ivfpq_index_accepts_float32():
    """float64 input should not crash — function must cast internally."""
    from src.index.build import build_ivfpq_index
    emb_f64 = _random_embeddings(320, d=16).astype(np.float64)
    index = build_ivfpq_index(emb_f64, n_clusters=4, n_subquantizers=4, bits_per_code=8, nprobe=2)
    assert index.ntotal == 320


# ---------------------------------------------------------------------------
# build_family_subindexes
# ---------------------------------------------------------------------------

def test_build_family_subindexes_one_per_family():
    from src.index.build import build_family_subindexes
    emb = _random_embeddings(60, d=16)
    labels = ["Asteraceae"] * 20 + ["Rosaceae"] * 20 + ["Poaceae"] * 20
    sub = build_family_subindexes(emb, labels, n_clusters=4)
    assert set(sub.keys()) == {"Asteraceae", "Rosaceae", "Poaceae"}


def test_build_family_subindexes_small_family_uses_flat():
    """Families with fewer vectors than n_clusters must use exact IndexFlatL2."""
    import faiss
    from src.index.build import build_family_subindexes
    emb = _random_embeddings(12, d=16)
    # "Rare" has only 2 vectors, well below n_clusters=8
    labels = ["Common"] * 10 + ["Rare"] * 2
    sub = build_family_subindexes(emb, labels, n_clusters=8)
    assert isinstance(sub["Rare"], faiss.IndexFlatL2), (
        f"Expected IndexFlatL2 for tiny family, got {type(sub['Rare'])}"
    )


def test_build_family_subindexes_all_searchable():
    from src.index.build import build_family_subindexes
    emb = _random_embeddings(60, d=16)
    labels = ["A"] * 30 + ["B"] * 30
    sub = build_family_subindexes(emb, labels, n_clusters=4)
    for name, idx in sub.items():
        D, I = idx.search(emb[:2].copy(), k=1)
        assert D.shape == (2, 1), f"Search on {name} sub-index failed"


# ---------------------------------------------------------------------------
# verify_recall
# ---------------------------------------------------------------------------

def test_verify_recall_self_retrieval():
    """Probes identical to indexed vectors — recall@10 must be 1.0."""
    from src.index.build import build_ivfpq_index, verify_recall
    emb = _random_embeddings(320, d=16)
    # nprobe=n_clusters forces exhaustive IVF search → exact recall
    index = build_ivfpq_index(emb, n_clusters=8, n_subquantizers=4, bits_per_code=8, nprobe=8)
    ids = [f"spec_{i}" for i in range(320)]
    recall = verify_recall(index, emb, ids, k=10)
    assert recall == 1.0, f"Expected recall@10=1.0, got {recall}"


def test_verify_recall_returns_float():
    from src.index.build import build_ivfpq_index, verify_recall
    emb = _random_embeddings(320, d=16)
    index = build_ivfpq_index(emb, n_clusters=4, n_subquantizers=4, bits_per_code=8, nprobe=4)
    ids = [f"s{i}" for i in range(320)]
    r = verify_recall(index, emb, ids, k=5)
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# save_index / load_index
# ---------------------------------------------------------------------------

def test_save_load_index_roundtrip(tmp_path):
    from src.index.build import build_ivfpq_index, save_index, load_index
    emb = _random_embeddings(320, d=16)
    index = build_ivfpq_index(emb, n_clusters=4, n_subquantizers=4, bits_per_code=8, nprobe=4)

    path = str(tmp_path / "test.faiss")
    save_index(index, path)
    loaded = load_index(path)

    query = emb[:5].copy()
    D_orig, I_orig = index.search(query, k=3)
    D_load, I_load = loaded.search(query, k=3)
    assert np.array_equal(I_orig, I_load), "Loaded index returns different neighbor indices"


# ---------------------------------------------------------------------------
# encode_specimens (mock model)
# ---------------------------------------------------------------------------

class _MockModel:
    """Fake model that returns deterministic 512-d Poincaré-ball-like vectors."""
    def encode(self, images) -> np.ndarray:
        n = len(images)
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, 512)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs * 0.9


def test_encode_specimens_shape(tmp_path):
    from PIL import Image as PILImage
    from src.index.build import encode_specimens

    # Create 5 dummy images in tmp_path
    ids = [f"spec_{i}" for i in range(5)]
    for sid in ids:
        img = PILImage.new("RGB", (64, 64), color=(100, 150, 200))
        img.save(tmp_path / f"{sid}.jpg")

    model = _MockModel()
    embeddings, returned_ids = encode_specimens(ids, str(tmp_path), model, batch_size=3)

    assert embeddings.shape == (5, 512), f"Expected (5, 512), got {embeddings.shape}"
    assert embeddings.dtype == np.float32
    assert returned_ids == ids


def test_encode_specimens_skips_missing(tmp_path):
    """Missing image files should be skipped gracefully."""
    from PIL import Image as PILImage
    from src.index.build import encode_specimens

    # Only create 3 of 5 images
    all_ids = [f"spec_{i}" for i in range(5)]
    present_ids = all_ids[:3]
    for sid in present_ids:
        img = PILImage.new("RGB", (32, 32))
        img.save(tmp_path / f"{sid}.jpg")

    model = _MockModel()
    embeddings, returned_ids = encode_specimens(all_ids, str(tmp_path), model, batch_size=4)

    assert embeddings.shape[0] == 3
    assert returned_ids == present_ids


# ---------------------------------------------------------------------------
# create_specimens_db
# ---------------------------------------------------------------------------

def _make_specimen_df():
    import pandas as pd
    return pd.DataFrame({
        "occurrence_id":   ["occ1", "occ2", "occ3"],
        "scientific_name": ["Clarkia gracilis", "Rosa californica", "Bromus hordeaceus"],
        "ott_id":          [123, 456, 789],
        "family":          ["Onagraceae", "Rosaceae", "Poaceae"],
        "genus":           ["Clarkia", "Rosa", "Bromus"],
        "species":         ["gracilis", "californica", "hordeaceus"],
        "latitude":        [37.5, 38.0, 36.5],
        "longitude":       [-122.0, -122.5, -121.0],
        "state_province":  ["California", "California", "California"],
        "event_date":      ["2021-05-01", "2020-06-15", "2019-04-10"],
        "institution":     ["UC Berkeley", "Calflora", "Jepson"],
        "image_url":       ["img1.jpg", "img2.jpg", "img3.jpg"],
    })


def test_create_specimens_db_columns(tmp_path):
    from src.index.bundle import create_specimens_db
    df = _make_specimen_df()
    db_path = str(tmp_path / "specimens.db")
    create_specimens_db(df, db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM specimens").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM specimens").description]
    conn.close()

    assert len(rows) == 3
    assert "occurrence_id" in cols
    assert "scientific_name" in cols
    assert "family" in cols


def test_create_specimens_db_occurrence_id_index(tmp_path):
    from src.index.bundle import create_specimens_db
    df = _make_specimen_df()
    db_path = str(tmp_path / "specimens.db")
    create_specimens_db(df, db_path)

    conn = sqlite3.connect(db_path)
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='specimens'"
    ).fetchall()
    conn.close()

    index_names = [r[0] for r in indexes]
    assert any("occurrence_id" in n for n in index_names), (
        f"No index on occurrence_id found. Indexes: {index_names}"
    )


# ---------------------------------------------------------------------------
# check_bundle_size
# ---------------------------------------------------------------------------

def test_check_bundle_size_empty_dir(tmp_path):
    from src.index.bundle import check_bundle_size
    size = check_bundle_size(str(tmp_path), warn_threshold_mb=400.0)
    assert size == 0.0


def test_check_bundle_size_with_files(tmp_path):
    from src.index.bundle import check_bundle_size
    (tmp_path / "a.bin").write_bytes(b"\x00" * 1024)        # 1 KB
    (tmp_path / "b.bin").write_bytes(b"\x00" * 2048)        # 2 KB
    size = check_bundle_size(str(tmp_path), warn_threshold_mb=400.0)
    expected_mb = 3 * 1024 / (1024 * 1024)
    assert abs(size - expected_mb) < 1e-5


def test_check_bundle_size_warns_when_over_threshold(tmp_path):
    import warnings
    from src.index.bundle import check_bundle_size
    # Write 1MB + 1 byte (over 1MB threshold)
    (tmp_path / "big.bin").write_bytes(b"\x00" * (1024 * 1024 + 1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_bundle_size(str(tmp_path), warn_threshold_mb=1.0)
    assert len(w) == 1
    assert "1.0" in str(w[0].message) or "threshold" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# generate_thumbnails
# ---------------------------------------------------------------------------

def test_generate_thumbnails_creates_files(tmp_path):
    from PIL import Image as PILImage
    from src.index.bundle import generate_thumbnails

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()

    # Create a 200×300 test image
    img = PILImage.new("RGB", (200, 300), color=(80, 120, 160))
    img.save(image_dir / "spec1.jpg")

    generate_thumbnails(str(image_dir), ["spec1"], str(thumb_dir), size=(128, 128))

    thumb_path = thumb_dir / "spec1.jpg"
    assert thumb_path.exists(), "Thumbnail not created"
    loaded = PILImage.open(thumb_path)
    assert loaded.size == (128, 128), f"Expected 128×128, got {loaded.size}"


def test_generate_thumbnails_skips_missing(tmp_path):
    """Missing source images must not raise, just be skipped."""
    from src.index.bundle import generate_thumbnails

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir()

    # No image files created — all IDs are "missing"
    generate_thumbnails(str(image_dir), ["missing1", "missing2"], str(thumb_dir))
    assert list(thumb_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# pack_bundle
# ---------------------------------------------------------------------------

def test_pack_bundle_creates_manifest(tmp_path):
    """Smoke test: pack_bundle with no real indexes/images writes manifest.json."""
    import json
    from src.index.bundle import pack_bundle

    # Create minimal directory structure
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "best.pt").write_bytes(b"fake")

    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "faiss_global.bin").write_bytes(b"fake")

    # Write a tiny parquet from the spec df
    import pandas as pd
    df = _make_specimen_df()
    parquet_path = str(tmp_path / "specimens.parquet")
    df.to_parquet(parquet_path, index=False)

    subtree_path = tmp_path / "subtree.json"
    subtree_path.write_text('{"newick": "(A,B);", "ott_ids": [1,2]}')

    out_dir = str(tmp_path / "bundle")

    manifest = pack_bundle(
        region="california",
        checkpoint_dir=str(ckpt_dir),
        index_dir=str(index_dir),
        specimens_parquet=parquet_path,
        image_dir=None,           # thumbnails skipped when None
        opentree_subtree_json=str(subtree_path),
        output_dir=out_dir,
    )

    manifest_path = Path(out_dir) / "manifest.json"
    assert manifest_path.exists(), "manifest.json not written"

    with open(manifest_path) as f:
        saved = json.load(f)

    for key in ("version", "region", "creation_date", "n_specimens"):
        assert key in saved, f"Missing key '{key}' in manifest"

    assert saved["region"] == "california"
    assert saved["n_specimens"] == 3
