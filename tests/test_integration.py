"""
Phase 8 integration tests — full pipeline on real CCH2 data.

Uses the SPIF collection (smallest CCH2 collection, ~494 rows) so the test
runs fast.  Skipped automatically if the data directory is absent.

Flow:
    parse_dwca(SPIF) → filter_by_region(california) → cap_per_taxon
    → synthetic embeddings → build_ivfpq_index / build_family_subindexes
    → pack_bundle → load_bundle → retrieve → eval metrics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SPIF_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "symbiota" / "cch2" / "SPIF"
OPENTREE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "opentree_mini_subtree.json"
REGIONS_CONFIG = Path(__file__).resolve().parent.parent / "config" / "regions.yaml"

pytestmark = pytest.mark.skipif(
    not SPIF_DIR.exists(),
    reason="CCH2 SPIF collection not downloaded — run scripts/download_data.sh",
)

DIM = 32   # small embedding dim for fast tests
SEED = 42


# ---------------------------------------------------------------------------
# Shared fixture: parse + filter + cap → real California specimens from SPIF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def california_spif_df():
    """Parse SPIF, filter to California bbox, cap at 50/taxon."""
    import yaml
    from src.data.parse import parse_dwca
    from src.data.filter import filter_by_region
    from src.data.balance import cap_per_taxon

    df = parse_dwca(str(SPIF_DIR), "cch2")

    with open(REGIONS_CONFIG) as f:
        regions_config = yaml.safe_load(f)
    df_ca = filter_by_region(df, "california", regions_config)

    # Keep rows with scientific_name (don't require image_url — no images downloaded)
    df_ca = df_ca[
        df_ca["scientific_name"].notna()
        & (df_ca["scientific_name"].astype(str).str.strip() != "")
    ].copy()

    df_ca = cap_per_taxon(df_ca, max_images=50)
    return df_ca


@pytest.fixture(scope="module")
def mini_bundle(california_spif_df, tmp_path_factory):
    """Build a complete mini bundle from SPIF California specimens."""
    import faiss
    import yaml
    from src.data.parse import save_parquet
    from src.index.build import build_family_subindexes, save_index
    from src.index.bundle import pack_bundle

    df = california_spif_df
    if len(df) < 5:
        pytest.skip(f"Too few California specimens in SPIF ({len(df)})")

    tmpdir = tmp_path_factory.mktemp("bundle")
    index_dir = tmpdir / "index"
    index_dir.mkdir()
    (index_dir / "faiss_families").mkdir()

    # Synthetic Poincaré-ball embeddings (random, normalized to norm < 1)
    rng = np.random.default_rng(SEED)
    n = len(df)
    emb = rng.standard_normal((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-8) * 0.9

    specimen_ids = df["occurrence_id"].tolist()
    family_labels = df["family"].fillna("Unknown").tolist()

    # Global index (FlatL2 for small n; IVF-PQ needs ≥256 vectors)
    global_index = faiss.IndexFlatL2(DIM)
    global_index.add(emb)
    save_index(global_index, str(index_dir / "faiss_global.bin"))

    # Family sub-indexes
    family_indexes = build_family_subindexes(emb, family_labels)
    for family, idx in family_indexes.items():
        safe = family.replace("/", "_").replace(" ", "_")
        save_index(idx, str(index_dir / "faiss_families" / f"{safe}.bin"))

    # Save specimen IDs alongside index (needed by load_bundle for ordering)
    import json
    (index_dir / "specimen_ids.json").write_text(
        json.dumps(specimen_ids), encoding="utf-8"
    )

    # Specimens parquet
    specimens_parquet = tmpdir / "specimens.parquet"
    save_parquet(df, str(specimens_parquet))

    # pack_bundle (checkpoint_dir empty — no model files, skipped gracefully)
    ckpt_dir = tmpdir / "ckpt"
    ckpt_dir.mkdir()
    bundle_dir = tmpdir / "bundle_out"
    pack_bundle(
        region="california",
        checkpoint_dir=str(ckpt_dir),
        index_dir=str(index_dir),
        specimens_parquet=str(specimens_parquet),
        image_dir=None,
        opentree_subtree_json=str(OPENTREE_FIXTURE),
        output_dir=str(bundle_dir),
    )

    return bundle_dir, emb, specimen_ids, family_labels


# ---------------------------------------------------------------------------
# parse_dwca tests
# ---------------------------------------------------------------------------

def test_parse_dwca_spif_row_count(california_spif_df):
    """At least some California specimens in SPIF."""
    assert len(california_spif_df) > 0, "No California specimens parsed from SPIF"


def test_parse_dwca_spif_family_populated(california_spif_df):
    assert california_spif_df["family"].notna().any()


def test_parse_dwca_spif_genus_populated(california_spif_df):
    assert california_spif_df["genus"].notna().any()


def test_parse_dwca_spif_coordinates_are_floats(california_spif_df):
    lat = california_spif_df["latitude"]
    assert lat.notna().any()
    assert lat.dropna().dtype.kind == "f"


def test_parse_dwca_spif_source_label(california_spif_df):
    assert (california_spif_df["source"] == "cch2").all()


# ---------------------------------------------------------------------------
# load_bundle tests
# ---------------------------------------------------------------------------

def test_load_bundle_returns_bundle_dataclass(mini_bundle):
    from src.index.bundle import load_bundle
    from src.retrieval.search import Bundle
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    assert isinstance(bundle, Bundle)


def test_load_bundle_global_index_has_vectors(mini_bundle, california_spif_df):
    from src.index.bundle import load_bundle
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    assert bundle.global_index.ntotal == len(california_spif_df)


def test_load_bundle_specimen_ids_match(mini_bundle, california_spif_df):
    from src.index.bundle import load_bundle
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    assert len(bundle.specimen_ids) == len(california_spif_df)


def test_load_bundle_specimens_metadata_has_required_keys(mini_bundle):
    from src.index.bundle import load_bundle
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    sample = next(iter(bundle.specimens_metadata.values()))
    assert "family" in sample
    assert "genus" in sample
    assert "taxon" in sample


def test_load_bundle_family_indexes_present(mini_bundle):
    from src.index.bundle import load_bundle
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    assert len(bundle.family_indexes) > 0


def test_load_bundle_family_specimen_ids_consistent(mini_bundle):
    """Every family in family_indexes has a matching entry in family_specimen_ids."""
    from src.index.bundle import load_bundle
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    for family in bundle.family_indexes:
        assert family in bundle.family_specimen_ids, f"Missing family_specimen_ids for {family}"


# ---------------------------------------------------------------------------
# retrieve tests on loaded bundle
# ---------------------------------------------------------------------------

def test_retrieve_returns_results(mini_bundle):
    import torch
    from src.index.bundle import load_bundle
    from src.retrieval.search import retrieve
    bundle_dir, emb, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))

    rng = np.random.default_rng(SEED + 1)
    query_np = rng.standard_normal((1, DIM)).astype(np.float32)
    query_np = query_np / (np.linalg.norm(query_np) + 1e-8) * 0.5
    n_fam = max(1, len(bundle.family_indexes))
    query = {
        "poincare": torch.tensor(query_np),
        "family_probs": torch.ones(1, n_fam) / n_fam,
    }
    results = retrieve(query, bundle, output_top_k=5)
    assert len(results) > 0


def test_retrieve_result_has_required_fields(mini_bundle):
    import torch
    from src.index.bundle import load_bundle
    from src.retrieval.search import retrieve
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))

    rng = np.random.default_rng(SEED + 2)
    query_np = rng.standard_normal((1, DIM)).astype(np.float32)
    query_np = query_np / (np.linalg.norm(query_np) + 1e-8) * 0.5
    n_fam = max(1, len(bundle.family_indexes))
    query = {
        "poincare": torch.tensor(query_np),
        "family_probs": torch.ones(1, n_fam) / n_fam,
    }
    results = retrieve(query, bundle, output_top_k=5)
    for r in results:
        assert "id" in r
        assert "distance" in r
        assert "score" in r
        assert "family" in r
        assert "taxon" in r


def test_retrieve_ids_in_specimen_ids(mini_bundle):
    """All retrieved IDs must be in the bundle's specimen_ids."""
    import torch
    from src.index.bundle import load_bundle
    from src.retrieval.search import retrieve
    bundle_dir, *_ = mini_bundle
    bundle = load_bundle(str(bundle_dir))
    sid_set = set(bundle.specimen_ids)

    rng = np.random.default_rng(SEED + 3)
    query_np = rng.standard_normal((1, DIM)).astype(np.float32)
    query_np = query_np / (np.linalg.norm(query_np) + 1e-8) * 0.5
    n_fam = max(1, len(bundle.family_indexes))
    query = {
        "poincare": torch.tensor(query_np),
        "family_probs": torch.ones(1, n_fam) / n_fam,
    }
    results = retrieve(query, bundle, output_top_k=10)
    for r in results:
        assert r["id"] in sid_set, f"Retrieved ID {r['id']} not in specimen_ids"


# ---------------------------------------------------------------------------
# eval metrics on pipeline output
# ---------------------------------------------------------------------------

def test_eval_precision_at_k_on_pipeline(mini_bundle):
    """Run precision@k on pipeline output — no crash, sensible range."""
    import torch
    from src.index.bundle import load_bundle
    from src.retrieval.search import retrieve
    from src.eval.metrics import precision_at_k
    bundle_dir, emb, specimen_ids, _ = mini_bundle
    bundle = load_bundle(str(bundle_dir))

    # Use first 5 specimens as queries; true_id = the specimen itself (should be retrieved)
    n_queries = min(5, len(specimen_ids))
    retrieved_ids = []
    true_ids = []
    for i in range(n_queries):
        query_np = emb[i : i + 1].copy()
        n_fam = max(1, len(bundle.family_indexes))
        query = {
            "poincare": torch.tensor(query_np),
            "family_probs": torch.ones(1, n_fam) / n_fam,
        }
        results = retrieve(query, bundle, output_top_k=10)
        retrieved_ids.append([r["id"] for r in results])
        true_ids.append(specimen_ids[i])

    p1 = precision_at_k(retrieved_ids, true_ids, k=1)
    p5 = precision_at_k(retrieved_ids, true_ids, k=5)
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p5 <= 1.0
    # FlatL2 index — querying with the exact embedding should find itself
    assert p1 > 0.0, "Expected at least one exact match with FlatL2 index"


def test_eval_hierarchical_accuracy_on_pipeline(mini_bundle):
    """Run hierarchical_accuracy on pipeline output — no crash, sensible range."""
    import torch
    from src.index.bundle import load_bundle
    from src.retrieval.search import retrieve
    from src.eval.metrics import hierarchical_accuracy
    bundle_dir, emb, specimen_ids, _ = mini_bundle
    bundle = load_bundle(str(bundle_dir))

    n_queries = min(5, len(specimen_ids))
    predictions = []
    ground_truth = []
    for i in range(n_queries):
        query_np = emb[i : i + 1].copy()
        n_fam = max(1, len(bundle.family_indexes))
        query = {
            "poincare": torch.tensor(query_np),
            "family_probs": torch.ones(1, n_fam) / n_fam,
        }
        results = retrieve(query, bundle, output_top_k=1)
        if results:
            top = results[0]
            predictions.append({
                "family": top.get("family", ""),
                "genus": top.get("genus", ""),
                "species": top.get("taxon", ""),
            })
            true_meta = bundle.specimens_metadata.get(specimen_ids[i], {})
            ground_truth.append({
                "family": true_meta.get("family", ""),
                "genus": true_meta.get("genus", ""),
                "species": true_meta.get("taxon", ""),
            })

    if predictions:
        result = hierarchical_accuracy(predictions, ground_truth)
        assert 0.0 <= result["family"] <= 1.0
        assert 0.0 <= result["genus"] <= 1.0
        assert 0.0 <= result["species"] <= 1.0
