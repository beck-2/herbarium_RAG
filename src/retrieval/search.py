"""
Iterative clade-conditioned retrieval pipeline.

Implements the two-round retrieval logic from SPEC §8.2:

Round 1: Global FAISS IVF-PQ search, top-50 candidates.
Round 2: If family_confidence > 0.65, re-query family-specific subindex (top-30).
         Merge and dedup with Round 1 results.
(Optional Round 3: genus-level, if genus_confidence > 0.55.)

After retrieval rounds, phylogenetic graph aggregation (retrieval.graph) adjusts scores.
Returns top-10 final candidates.

DECISION-9: n_reretrieval_rounds = 2 (default). Config key: retrieval.n_reretrieval_rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FAMILY_CONFIDENCE_THRESHOLD = 0.65
GENUS_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class Bundle:
    """All runtime artifacts needed for one regional retrieval session.

    Attributes:
        global_index:        FAISS index over all specimens (IVF-PQ for production,
                             FlatL2 acceptable for smoke tests).
        family_indexes:      Dict mapping family name → FAISS sub-index.
        specimen_ids:        Ordered list of specimen IDs corresponding to
                             global_index rows (position i → specimen_ids[i]).
        family_specimen_ids: Dict mapping family name → ordered list of specimen
                             IDs in the corresponding family sub-index.
        opentree_subtree:    Dict loaded from bundle's opentree_subtree.json.
        specimens_metadata:  Dict mapping specimen_id → {family, genus, taxon}.
    """
    global_index: Any
    family_indexes: dict[str, Any]
    specimen_ids: list[str]
    family_specimen_ids: dict[str, list[str]]
    opentree_subtree: dict
    specimens_metadata: dict[str, dict]


def global_faiss_search(
    query_euclidean: np.ndarray,
    index,
    specimen_ids: list[str],
    k: int = 50,
) -> list[dict]:
    """Search the global FAISS index, return up to k candidate dicts.

    Args:
        query_euclidean: Query vector of shape (1, D) or (D,), float32.
        index: FAISS index with ntotal entries.
        specimen_ids: Ordered list of IDs (position i → specimen_ids[i]).
        k: Number of nearest neighbours to retrieve.

    Returns:
        List of dicts [{id, distance}, ...] sorted by distance ascending.
        Length may be less than k if ntotal < k.
    """
    query = np.ascontiguousarray(query_euclidean, dtype=np.float32).reshape(1, -1)
    actual_k = min(k, index.ntotal)
    D, I = index.search(query, actual_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:  # FAISS padding sentinel
            continue
        results.append({"id": specimen_ids[idx], "distance": float(dist)})
    return results


def family_index_search(
    query_euclidean: np.ndarray,
    family_name: str,
    family_indexes: dict[str, Any],
    family_specimen_ids: dict[str, list[str]],
    k: int = 30,
) -> list[dict]:
    """Search a family-specific sub-index for clade-conditioned re-retrieval.

    Args:
        query_euclidean: Query vector (1, D) or (D,), float32.
        family_name: Name of the family to search.
        family_indexes: Dict mapping family name → FAISS index.
        family_specimen_ids: Dict mapping family name → ordered specimen ID list.
        k: Number of nearest neighbours.

    Returns:
        List of candidate dicts [{id, distance}, ...], or [] if family unknown.
    """
    if family_name not in family_indexes:
        return []
    index = family_indexes[family_name]
    ids = family_specimen_ids[family_name]
    return global_faiss_search(query_euclidean, index, ids, k=k)


def merge_and_dedup(
    candidates_a: list[dict],
    candidates_b: list[dict],
    k: int = 50,
) -> list[dict]:
    """Merge two candidate lists, dedup by ID (keep lower distance), return top-k.

    Args:
        candidates_a: First candidate list.
        candidates_b: Second candidate list.
        k: Maximum number of results to return.

    Returns:
        Merged, deduped, distance-sorted list of up to k candidates.
    """
    best: dict[str, dict] = {}
    for c in candidates_a + candidates_b:
        cid = c["id"]
        if cid not in best or c["distance"] < best[cid]["distance"]:
            best[cid] = c
    return sorted(best.values(), key=lambda x: x["distance"])[:k]


def retrieve(
    query: dict,
    bundle: Bundle,
    n_reretrieval_rounds: int = 2,
    global_top_k: int = 50,
    family_top_k: int = 30,
    output_top_k: int = 10,
) -> list[dict]:
    """Full iterative clade-conditioned retrieval pipeline.

    Args:
        query: Output of retrieval.encode.encode_query — must contain
               'euclidean', 'poincare', and 'family_probs' tensors.
        bundle: Loaded regional Bundle.
        n_reretrieval_rounds: Clade-conditioned re-retrieval rounds (DECISION-9).
        global_top_k: Top-k from global search (Round 1).
        family_top_k: Top-k from family sub-index (Round 2).
        output_top_k: Final number of ranked results to return.

    Returns:
        List of up to output_top_k candidate dicts with keys:
            id, distance, family, genus, taxon, score (graph-aggregated).
        Sorted by score descending.
    """
    from src.retrieval.graph import build_retrieval_graph, aggregate_scores

    # --- Round 1: global search ---
    # Index is built on Poincaré-projected vectors (hyperbolic_dim), not Euclidean.
    query_np = query["poincare"].detach().cpu().numpy()
    candidates = global_faiss_search(
        query_np, bundle.global_index, bundle.specimen_ids, k=global_top_k
    )

    # --- Round 2: family-targeted re-retrieval (if confident) ---
    if n_reretrieval_rounds >= 1:
        family_probs = query["family_probs"]  # (1, n_families)
        top_family_conf = float(family_probs.max())
        if top_family_conf > FAMILY_CONFIDENCE_THRESHOLD:
            # Map top family probability index to a family name via candidate metadata
            family_name = _top_family_name(candidates, bundle.specimens_metadata)
            if family_name:
                family_candidates = family_index_search(
                    query_np, family_name,
                    bundle.family_indexes, bundle.family_specimen_ids,
                    k=family_top_k,
                )
                candidates = merge_and_dedup(candidates, family_candidates, k=global_top_k)

    # --- Enrich candidates with taxonomy metadata ---
    for c in candidates:
        meta = bundle.specimens_metadata.get(c["id"], {})
        c.setdefault("family", meta.get("family", ""))
        c.setdefault("genus",  meta.get("genus",  ""))
        c.setdefault("taxon",  meta.get("taxon",  ""))

    # --- Phylogenetic graph aggregation ---
    query_poincare = query["poincare"].detach().cpu().numpy().reshape(-1)
    graph = build_retrieval_graph(candidates, bundle.opentree_subtree)
    scores = aggregate_scores(query_poincare, candidates, graph)

    # --- Rank and return top-k ---
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:output_top_k]

    return [{**c, "score": float(s)} for c, s in ranked]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _top_family_name(candidates: list[dict], specimens_metadata: dict) -> str | None:
    """Return the most frequent family name among the current candidates."""
    from collections import Counter
    counts: Counter = Counter()
    for c in candidates:
        meta = specimens_metadata.get(c["id"], {})
        fam = meta.get("family", "")
        if fam:
            counts[fam] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
