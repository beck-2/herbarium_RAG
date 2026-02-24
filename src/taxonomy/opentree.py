"""
OpenTree of Life taxonomy utilities.

Responsibilities:
- Fetch subtree for a set of OTT IDs (regional subset) from the OpenTree API
- Compute pairwise patristic distances between taxa from the synoptic tree
- Cache distances in opentree_distances.db (SQLite) — all API results cached
- Export relevant subtree as JSON for bundle (opentree_subtree.json)
- Provide LCA (lowest common ancestor) rank lookups for graph aggregation

CRITICAL: Cache all API results. OpenTree API is rate-limited and
          must be available offline during inference.
"""

from __future__ import annotations

# TODO(phase2): implement subtree fetch via opentree.OT.induced_subtree
# TODO(phase2): implement patristic distance computation from Newick/nexml tree
# TODO(phase2): implement SQLite distance cache (opentree_distances.db)
# TODO(phase2): implement LCA rank lookup


def fetch_induced_subtree(ott_ids: list[int], cache_path: str) -> dict:
    """Fetch the induced subtree for a set of OTT IDs.

    Returns the tree as a dict (parsed from OpenTree API response).
    Results are cached to avoid repeated API calls.

    Args:
        ott_ids: List of OpenTree OTT IDs for the taxa of interest.
        cache_path: Path to opentree_distances.db SQLite file.
    """
    raise NotImplementedError


def compute_patristic_distances(
    ott_ids: list[int],
    subtree: dict,
    cache_db_path: str,
) -> dict[tuple[int, int], float]:
    """Compute pairwise patristic distances from a subtree.

    Args:
        ott_ids: Taxa to compute distances for.
        subtree: Tree dict from fetch_induced_subtree.
        cache_db_path: Path to SQLite cache for storing distances.

    Returns:
        Dict mapping (ott_id_a, ott_id_b) pairs to float distances.
    """
    raise NotImplementedError


def get_lca_rank(ott_id_a: int, ott_id_b: int, subtree: dict) -> str:
    """Return the taxonomic rank of the LCA of two taxa.

    Returns one of: 'species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom'.
    Used during inference-time graph construction.
    """
    raise NotImplementedError


def export_subtree_json(subtree: dict, output_path: str) -> None:
    """Serialize the subtree dict to JSON for inclusion in a regional bundle."""
    raise NotImplementedError
