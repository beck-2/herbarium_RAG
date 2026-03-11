"""
Tests for src/retrieval/: encode, graph, search.
Phase 0: import stubs.
Phase 6: functional tests — mock model + synthetic data, no BioCLIP-2 download.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Import / API surface (Phase 0)
# ---------------------------------------------------------------------------

def test_encode_module_importable():
    from src.retrieval import encode
    assert hasattr(encode, "encode_query")


def test_search_module_importable():
    from src.retrieval import search
    assert hasattr(search, "global_faiss_search")
    assert hasattr(search, "family_index_search")
    assert hasattr(search, "merge_and_dedup")
    assert hasattr(search, "retrieve")
    assert hasattr(search, "FAMILY_CONFIDENCE_THRESHOLD")
    assert hasattr(search, "GENUS_CONFIDENCE_THRESHOLD")


def test_graph_module_importable():
    from src.retrieval import graph
    assert hasattr(graph, "build_retrieval_graph")
    assert hasattr(graph, "aggregate_scores")


def test_confidence_thresholds_in_range():
    from src.retrieval.search import FAMILY_CONFIDENCE_THRESHOLD, GENUS_CONFIDENCE_THRESHOLD
    assert 0.0 < FAMILY_CONFIDENCE_THRESHOLD < 1.0
    assert 0.0 < GENUS_CONFIDENCE_THRESHOLD < 1.0
    assert GENUS_CONFIDENCE_THRESHOLD < FAMILY_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers — mock model and synthetic data
# ---------------------------------------------------------------------------

_EMBED_DIM = 64
_HYPER_DIM = 32
_N_FAM = 5
_N_GEN = 15
_N_SPE = 30


class _MockModel:
    """Minimal mock of the assembled model used by encode_query."""

    def __init__(self, embed_dim=_EMBED_DIM, hyper_dim=_HYPER_DIM,
                 n_families=_N_FAM, n_genera=_N_GEN, n_species=_N_SPE):
        from src.model.hyperbolic import HyperbolicProjection
        from src.model.heads import HierarchicalHeads

        self.embed_dim = embed_dim
        self.hyper_dim = hyper_dim

        # preprocess: converts PIL → (C, H, W) tensor, batch dim added by encode_query
        import torchvision.transforms as T
        self.preprocess = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
        ])

        # backbone: tiny linear that maps flattened image → embed_dim
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, embed_dim),
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.hyperbolic_proj = HyperbolicProjection(
            in_dim=embed_dim, out_dim=hyper_dim
        )
        self.hyperbolic_proj.eval()

        self.heads = HierarchicalHeads(
            in_dim=embed_dim,
            n_families=n_families,
            n_genera=n_genera,
            n_species=n_species,
        )
        self.heads.eval()

        # simple text encoder: bag-of-chars → embed_dim
        self.text_encoder = nn.Sequential(
            nn.Linear(256, embed_dim),
        )
        self.text_encoder.eval()

    def encode_text(self, text: str) -> torch.Tensor:
        """Returns (1, embed_dim) text embedding from raw string."""
        # crude fixed-size encoding: char codes → average
        codes = [ord(c) % 256 for c in text[:256]]
        codes += [0] * (256 - len(codes))
        x = torch.tensor(codes, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.text_encoder(x)


def _make_pil_image(w=64, h=64):
    from PIL import Image as PILImage
    return PILImage.new("RGB", (w, h), color=(80, 120, 160))


def _make_candidates(n=10, n_families=3, seed=0):
    """Synthetic candidate dicts with family/genus/taxon metadata + distances."""
    rng = np.random.default_rng(seed)
    families = [f"Fam{i % n_families}" for i in range(n)]
    genera = [f"Gen{i % (n_families * 2)}" for i in range(n)]
    return [
        {
            "id": f"spec_{i}",
            "distance": float(rng.uniform(0.1, 2.0)),
            "family": families[i],
            "genus": genera[i],
            "taxon": f"{genera[i]} species{i}",
            "poincare": rng.standard_normal(_HYPER_DIM).astype(np.float32) * 0.1,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------

def test_encode_query_returns_required_keys():
    from src.retrieval.encode import encode_query
    model = _MockModel()
    image = _make_pil_image()
    result = encode_query(image, model)
    for key in ("euclidean", "poincare", "family_probs", "genus_probs", "species_probs"):
        assert key in result, f"Missing key '{key}'"


def test_encode_query_shapes():
    from src.retrieval.encode import encode_query
    model = _MockModel()
    image = _make_pil_image()
    result = encode_query(image, model)
    assert result["euclidean"].shape == (1, _EMBED_DIM), result["euclidean"].shape
    assert result["poincare"].shape == (1, _HYPER_DIM), result["poincare"].shape
    assert result["family_probs"].shape == (1, _N_FAM)
    assert result["genus_probs"].shape == (1, _N_GEN)
    assert result["species_probs"].shape == (1, _N_SPE)


def test_encode_query_poincare_norm_lt1():
    from src.retrieval.encode import encode_query
    model = _MockModel()
    image = _make_pil_image()
    result = encode_query(image, model)
    norm = result["poincare"].norm(dim=-1).item()
    assert norm < 1.0, f"Poincaré norm {norm} >= 1.0"


def test_encode_query_probs_sum_to_1():
    from src.retrieval.encode import encode_query
    model = _MockModel()
    image = _make_pil_image()
    result = encode_query(image, model)
    for key in ("family_probs", "genus_probs", "species_probs"):
        s = result[key].sum(dim=-1).item()
        assert abs(s - 1.0) < 1e-4, f"{key} sums to {s}, expected 1.0"


def test_encode_query_text_fusion_changes_embedding():
    """Habitat text should produce a different euclidean embedding than image-only."""
    from src.retrieval.encode import encode_query
    torch.manual_seed(0)
    model = _MockModel()
    image = _make_pil_image()
    result_no_text = encode_query(image, model)
    result_with_text = encode_query(image, model, habitat_text="serpentine soil chaparral")
    diff = (result_no_text["euclidean"] - result_with_text["euclidean"]).abs().max().item()
    assert diff > 1e-4, "Text fusion had no effect on euclidean embedding"


def test_encode_query_no_text_encoder_ok():
    """Model without text_encoder should still work when no habitat_text provided."""
    from src.retrieval.encode import encode_query
    model = _MockModel()
    del model.text_encoder  # simulate no text encoder
    image = _make_pil_image()
    result = encode_query(image, model)
    assert "euclidean" in result


# ---------------------------------------------------------------------------
# build_retrieval_graph
# ---------------------------------------------------------------------------

def test_build_graph_same_genus_weight_1():
    from src.retrieval.graph import build_retrieval_graph
    candidates = [
        {"id": "a", "family": "Rosaceae", "genus": "Rosa",    "taxon": "Rosa canina"},
        {"id": "b", "family": "Rosaceae", "genus": "Rosa",    "taxon": "Rosa gallica"},
        {"id": "c", "family": "Rosaceae", "genus": "Prunus",  "taxon": "Prunus serrulata"},
    ]
    G = build_retrieval_graph(candidates, opentree_subtree={})
    assert G.has_edge(0, 1), "Same-genus pair should have an edge"
    assert abs(G[0][1]["weight"] - 1.0) < 1e-6


def test_build_graph_same_family_diff_genus_weight_03():
    from src.retrieval.graph import build_retrieval_graph
    candidates = [
        {"id": "a", "family": "Rosaceae", "genus": "Rosa",    "taxon": "Rosa canina"},
        {"id": "b", "family": "Rosaceae", "genus": "Prunus",  "taxon": "Prunus serrulata"},
    ]
    G = build_retrieval_graph(candidates, opentree_subtree={})
    assert G.has_edge(0, 1), "Same-family pair should have an edge"
    assert abs(G[0][1]["weight"] - 0.3) < 1e-6


def test_build_graph_diff_family_no_edge():
    from src.retrieval.graph import build_retrieval_graph
    candidates = [
        {"id": "a", "family": "Rosaceae",   "genus": "Rosa",    "taxon": "Rosa canina"},
        {"id": "b", "family": "Asteraceae", "genus": "Senecio", "taxon": "Senecio sp."},
    ]
    G = build_retrieval_graph(candidates, opentree_subtree={})
    assert not G.has_edge(0, 1), "Different-family pair should have no edge"


def test_build_graph_returns_networkx_graph():
    import networkx as nx
    from src.retrieval.graph import build_retrieval_graph
    candidates = _make_candidates(6)
    G = build_retrieval_graph(candidates, opentree_subtree={})
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 6


def test_build_graph_genus_edge_overrides_family():
    """When genus matches, edge weight should be 1.0, not 0.3."""
    from src.retrieval.graph import build_retrieval_graph
    candidates = [
        {"id": "a", "family": "Rosaceae", "genus": "Rosa", "taxon": "Rosa canina"},
        {"id": "b", "family": "Rosaceae", "genus": "Rosa", "taxon": "Rosa gallica"},
    ]
    G = build_retrieval_graph(candidates, opentree_subtree={})
    assert abs(G[0][1]["weight"] - 1.0) < 1e-6, (
        f"Same-genus edge should be weight=1.0, got {G[0][1]['weight']}"
    )


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------

def test_aggregate_scores_shape():
    import networkx as nx
    from src.retrieval.graph import aggregate_scores
    candidates = _make_candidates(8)
    G = nx.Graph()
    G.add_nodes_from(range(8))
    query = np.zeros(_HYPER_DIM, dtype=np.float32)
    scores = aggregate_scores(query, candidates, G, n_rounds=2)
    assert scores.shape == (8,), f"Expected (8,), got {scores.shape}"


def test_aggregate_scores_zero_rounds_unchanged():
    """With n_rounds=0, scores should equal the initial distance-based scores."""
    import networkx as nx
    from src.retrieval.graph import aggregate_scores, _initial_scores
    candidates = _make_candidates(5)
    G = nx.Graph()
    G.add_nodes_from(range(5))
    query = np.zeros(_HYPER_DIM, dtype=np.float32)
    scores_0 = aggregate_scores(query, candidates, G, n_rounds=0)
    initial = _initial_scores(candidates)
    np.testing.assert_allclose(scores_0, initial, rtol=1e-5)


def test_aggregate_scores_neighbor_boosts_score():
    """A low-score node connected to a high-score node should increase after aggregation."""
    import networkx as nx
    from src.retrieval.graph import aggregate_scores
    # Two candidates: node 0 (far, low score), node 1 (near, high score), connected by genus edge
    candidates = [
        {"id": "far",  "distance": 10.0, "family": "Rosaceae", "genus": "Rosa", "taxon": "Rosa x",
         "poincare": np.zeros(_HYPER_DIM, dtype=np.float32)},
        {"id": "near", "distance": 0.01, "family": "Rosaceae", "genus": "Rosa", "taxon": "Rosa y",
         "poincare": np.zeros(_HYPER_DIM, dtype=np.float32)},
    ]
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=1.0)
    query = np.zeros(_HYPER_DIM, dtype=np.float32)
    scores_0 = aggregate_scores(query, candidates, G, n_rounds=0)
    scores_2 = aggregate_scores(query, candidates, G, n_rounds=2)
    assert scores_2[0] > scores_0[0], (
        f"Low-score node should increase after aggregation: {scores_0[0]:.4f} → {scores_2[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# global_faiss_search
# ---------------------------------------------------------------------------

def _build_test_index(n=200, d=32, seed=0):
    """Build a tiny FAISS index for search tests."""
    import faiss
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb *= 0.9
    index = faiss.IndexFlatL2(d)
    index.add(emb)
    ids = [f"spec_{i}" for i in range(n)]
    return index, emb, ids


def test_global_search_returns_k_candidates():
    from src.retrieval.search import global_faiss_search
    index, emb, ids = _build_test_index(200, _HYPER_DIM)
    results = global_faiss_search(emb[0:1], index, ids, k=10)
    assert len(results) == 10


def test_global_search_candidate_structure():
    from src.retrieval.search import global_faiss_search
    index, emb, ids = _build_test_index(200, _HYPER_DIM)
    results = global_faiss_search(emb[0:1], index, ids, k=5)
    for r in results:
        assert "id" in r
        assert "distance" in r
        assert r["id"] in ids
        assert r["distance"] >= 0.0


def test_global_search_sorted_by_distance():
    from src.retrieval.search import global_faiss_search
    index, emb, ids = _build_test_index(200, _HYPER_DIM)
    results = global_faiss_search(emb[0:1], index, ids, k=20)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances), "Results should be sorted by distance ascending"


def test_global_search_k_larger_than_index():
    """k > ntotal should return ntotal results, not crash."""
    from src.retrieval.search import global_faiss_search
    import faiss
    small_emb = np.random.randn(5, _HYPER_DIM).astype(np.float32)
    index = faiss.IndexFlatL2(_HYPER_DIM)
    index.add(small_emb)
    ids = [f"s{i}" for i in range(5)]
    results = global_faiss_search(small_emb[0:1], index, ids, k=50)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# family_index_search
# ---------------------------------------------------------------------------

def test_family_search_returns_candidates():
    import faiss
    from src.retrieval.search import family_index_search
    rng = np.random.default_rng(1)
    fam_emb = rng.standard_normal((20, _HYPER_DIM)).astype(np.float32)
    index = faiss.IndexFlatL2(_HYPER_DIM)
    index.add(fam_emb)
    fam_ids = [f"fam_spec_{i}" for i in range(20)]
    family_indexes = {"Rosaceae": index}
    family_specimen_ids = {"Rosaceae": fam_ids}
    query = fam_emb[0:1]
    results = family_index_search(query, "Rosaceae", family_indexes, family_specimen_ids, k=5)
    assert len(results) <= 5
    for r in results:
        assert r["id"] in fam_ids


def test_family_search_unknown_family_returns_empty():
    from src.retrieval.search import family_index_search
    results = family_index_search(
        np.zeros((1, _HYPER_DIM), dtype=np.float32),
        "UnknownFamily",
        family_indexes={},
        family_specimen_ids={},
        k=10,
    )
    assert results == []


# ---------------------------------------------------------------------------
# merge_and_dedup
# ---------------------------------------------------------------------------

def test_merge_no_duplicates():
    from src.retrieval.search import merge_and_dedup
    a = [{"id": "s1", "distance": 0.5}, {"id": "s2", "distance": 0.8}]
    b = [{"id": "s3", "distance": 0.3}, {"id": "s4", "distance": 1.0}]
    merged = merge_and_dedup(a, b, k=10)
    ids = [r["id"] for r in merged]
    assert len(ids) == len(set(ids)), "Merged result contains duplicate IDs"


def test_merge_keeps_lower_distance():
    from src.retrieval.search import merge_and_dedup
    a = [{"id": "s1", "distance": 0.5}]
    b = [{"id": "s1", "distance": 0.2}]  # same ID, better distance
    merged = merge_and_dedup(a, b, k=10)
    assert len(merged) == 1
    assert abs(merged[0]["distance"] - 0.2) < 1e-6, "Should keep lower distance"


def test_merge_truncates_to_k():
    from src.retrieval.search import merge_and_dedup
    a = [{"id": f"a{i}", "distance": float(i)} for i in range(30)]
    b = [{"id": f"b{i}", "distance": float(i) + 0.5} for i in range(30)]
    merged = merge_and_dedup(a, b, k=20)
    assert len(merged) == 20


def test_merge_sorted_by_distance():
    from src.retrieval.search import merge_and_dedup
    a = [{"id": "s1", "distance": 1.0}, {"id": "s2", "distance": 0.5}]
    b = [{"id": "s3", "distance": 0.1}]
    merged = merge_and_dedup(a, b, k=10)
    distances = [r["distance"] for r in merged]
    assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# retrieve — full pipeline smoke test
# ---------------------------------------------------------------------------

def _make_synthetic_bundle(n=100, d=_HYPER_DIM, n_families=3, seed=0):
    """Build a minimal Bundle for smoke-testing retrieve()."""
    import faiss
    from src.retrieval.search import Bundle

    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb *= 0.9

    families = [f"Fam{i % n_families}" for i in range(n)]
    genera = [f"Gen{i % (n_families * 2)}" for i in range(n)]
    specimen_ids = [f"spec_{i}" for i in range(n)]

    # Global flat index
    global_index = faiss.IndexFlatL2(d)
    global_index.add(emb)

    # Family sub-indexes + family_specimen_ids
    family_indexes = {}
    family_specimen_ids = {}
    for fam in set(families):
        mask = np.array([f == fam for f in families])
        fam_emb = emb[mask]
        fam_ids = [specimen_ids[i] for i, m in enumerate(mask) if m]
        idx = faiss.IndexFlatL2(d)
        idx.add(fam_emb)
        family_indexes[fam] = idx
        family_specimen_ids[fam] = fam_ids

    specimens_metadata = {
        sid: {"family": families[i], "genus": genera[i], "taxon": f"{genera[i]} sp{i}"}
        for i, sid in enumerate(specimen_ids)
    }

    return Bundle(
        global_index=global_index,
        family_indexes=family_indexes,
        specimen_ids=specimen_ids,
        family_specimen_ids=family_specimen_ids,
        opentree_subtree={},
        specimens_metadata=specimens_metadata,
    )


def _make_synthetic_query(n_families=_N_FAM, n_genera=_N_GEN, n_species=_N_SPE, d=_HYPER_DIM,
                          high_family_conf=False):
    """Synthetic query dict mimicking encode_query output."""
    rng = np.random.default_rng(99)
    euclidean = torch.tensor(rng.standard_normal((1, _EMBED_DIM)), dtype=torch.float32)
    poincare_np = rng.standard_normal(d).astype(np.float32)
    poincare_np /= np.linalg.norm(poincare_np)
    poincare_np *= 0.5
    poincare = torch.tensor(poincare_np).unsqueeze(0)

    if high_family_conf:
        # Family 0 gets 0.9 probability
        fam_probs = torch.zeros(1, n_families)
        fam_probs[0, 0] = 0.9
    else:
        fam_probs = torch.softmax(torch.randn(1, n_families) * 0.1, dim=-1)

    return {
        "euclidean": euclidean,
        "poincare": poincare,
        "family_probs": fam_probs,
        "genus_probs": torch.softmax(torch.randn(1, n_genera), dim=-1),
        "species_probs": torch.softmax(torch.randn(1, n_species), dim=-1),
    }


def test_retrieve_returns_top_k():
    from src.retrieval.search import retrieve
    bundle = _make_synthetic_bundle()
    query = _make_synthetic_query()
    results = retrieve(query, bundle, n_reretrieval_rounds=1, output_top_k=10)
    assert len(results) == 10
    for r in results:
        assert "id" in r
        assert "score" in r


def test_retrieve_results_sorted_by_score():
    from src.retrieval.search import retrieve
    bundle = _make_synthetic_bundle()
    query = _make_synthetic_query()
    results = retrieve(query, bundle, output_top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted score descending"


def test_retrieve_reretrieval_triggered_on_high_family_conf():
    """High family confidence should trigger family sub-index search."""
    from src.retrieval.search import retrieve
    bundle = _make_synthetic_bundle(n=100, n_families=3)
    # family_probs[0] = 0.9 >> FAMILY_CONFIDENCE_THRESHOLD (0.65)
    query = _make_synthetic_query(high_family_conf=True)
    # Should not raise; re-retrieval internally queries family sub-index
    results = retrieve(query, bundle, n_reretrieval_rounds=2, output_top_k=5)
    assert len(results) == 5


def test_retrieve_zero_rounds_still_works():
    """n_reretrieval_rounds=0 skips re-retrieval but still returns results."""
    from src.retrieval.search import retrieve
    bundle = _make_synthetic_bundle()
    query = _make_synthetic_query(high_family_conf=True)
    results = retrieve(query, bundle, n_reretrieval_rounds=0, output_top_k=5)
    assert len(results) == 5
