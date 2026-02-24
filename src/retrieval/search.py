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

import numpy as np

# TODO(phase6): implement global_faiss_search
# TODO(phase6): implement family_index_search
# TODO(phase6): implement merge_and_dedup
# TODO(phase6): implement full retrieve() pipeline


FAMILY_CONFIDENCE_THRESHOLD = 0.65
GENUS_CONFIDENCE_THRESHOLD = 0.55


def global_faiss_search(
    query_euclidean: np.ndarray,
    index,
    specimen_ids: list[str],
    k: int = 50,
) -> list[dict]:
    """Search the global IVF-PQ index.

    Returns:
        List of candidate dicts: {id, embedding, distance, taxon, family, genus}.
    """
    raise NotImplementedError


def family_index_search(
    query_euclidean: np.ndarray,
    family_name: str,
    family_indexes: dict[str, object],
    specimen_ids: list[str],
    k: int = 30,
) -> list[dict]:
    """Search a family-specific sub-index for re-retrieval."""
    raise NotImplementedError


def merge_and_dedup(
    candidates_a: list[dict],
    candidates_b: list[dict],
    k: int = 50,
) -> list[dict]:
    """Merge two candidate lists, dedup by ID, keep top-k by distance."""
    raise NotImplementedError


def retrieve(
    query: dict,
    bundle,
    n_reretrieval_rounds: int = 2,
    global_top_k: int = 50,
    family_top_k: int = 30,
    output_top_k: int = 10,
) -> list[dict]:
    """Full iterative retrieval pipeline.

    Args:
        query: Output of retrieval.encode.encode_query.
        bundle: Loaded regional bundle (indexes, specimens.db, opentree_subtree).
        n_reretrieval_rounds: Number of clade-conditioned re-retrieval rounds (DECISION-9).
        global_top_k: Top-k from global search.
        family_top_k: Top-k from family sub-index search.
        output_top_k: Final number of ranked results.

    Returns:
        List of top output_top_k candidate dicts with updated graph-aggregated scores.
    """
    raise NotImplementedError
