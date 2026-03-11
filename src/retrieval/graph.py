"""
Inference-time phylogenetic graph aggregation.

Builds a small graph over retrieved candidates where edges connect specimens
whose taxa share a genus or family ancestor. Runs 1–2 rounds of message
passing to smooth similarity scores across phylogenetically related candidates.

DECISION-10: graph_edge_mode = 'genus_family' (default).
             Alternative: 'full_patristic' (expensive).

DECISION-9:  graph_n_rounds = 2 (default). Reduce to 1 if latency > 3s.

From SPEC §4.6:
    def build_retrieval_graph(retrieved_specimens, opentree_distances):
        G = nx.Graph()
        for s_i, s_j in pairs:
            lca_rank = get_lca_rank(s_i.taxon, s_j.taxon)
            if lca_rank in ('species', 'genus'):
                G.add_edge(i, j, weight=1.0 if lca_rank=='genus' else 0.3)
        return G

    def aggregate_scores(query_embed, retrieved, graph, n_rounds=2):
        scores = cosine_similarity(query_embed, [r.embed for r in retrieved])
        for _ in range(n_rounds):
            new_scores = scores.copy()
            for node in graph.nodes:
                neighbors = graph[node]
                neighbor_contribution = sum(
                    graph[node][nb]['weight'] * scores[nb] for nb in neighbors
                ) / (len(neighbors) + 1e-6)
                new_scores[node] = 0.7 * scores[node] + 0.3 * neighbor_contribution
            scores = new_scores
        return scores
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def build_retrieval_graph(
    candidates: list[dict],
    opentree_subtree: dict,
    edge_mode: str = "genus_family",
) -> nx.Graph:
    """Build a phylogenetic graph over retrieved candidates.

    Args:
        candidates: List of candidate dicts (each has 'family' and 'genus' keys).
        opentree_subtree: Subtree dict from bundle's opentree_subtree.json.
                          Used only in 'full_patristic' mode (not yet implemented).
        edge_mode: 'genus_family' (default) — edges based on genus/family match.
                   'full_patristic' — edges based on OTT patristic distances (deferred).

    Returns:
        nx.Graph where nodes are candidate indices (0..N-1) and edges carry
        'weight' attributes. Genus-level LCA: weight=1.0. Family-level: weight=0.3.
    """
    n = len(candidates)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if edge_mode == "genus_family":
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = candidates[i], candidates[j]
                if ci.get("genus") and cj.get("genus") and ci["genus"] == cj["genus"]:
                    G.add_edge(i, j, weight=1.0)
                elif ci.get("family") and cj.get("family") and ci["family"] == cj["family"]:
                    G.add_edge(i, j, weight=0.3)
    elif edge_mode == "full_patristic":
        raise NotImplementedError(
            "full_patristic edge mode requires OTT patristic distances. "
            "Use genus_family (default) until Phase 7+ evaluation."
        )
    else:
        raise ValueError(f"Unknown edge_mode '{edge_mode}'. Use 'genus_family'.")

    return G


def _initial_scores(candidates: list[dict]) -> np.ndarray:
    """Convert FAISS L2 distances to similarity scores: 1 / (1 + d)."""
    return np.array(
        [1.0 / (1.0 + c["distance"]) for c in candidates],
        dtype=np.float64,
    )


def aggregate_scores(
    query_poincare: np.ndarray,
    candidates: list[dict],
    graph: nx.Graph,
    n_rounds: int = 2,
    self_weight: float = 0.7,
    neighbor_weight: float = 0.3,
) -> np.ndarray:
    """Run message-passing score aggregation over the retrieval graph.

    Initial scores are derived from FAISS L2 distances: score = 1 / (1 + d).
    Each round smooths scores toward phylogenetically related neighbors.

    Args:
        query_poincare: Query Poincaré point, shape (hyperbolic_dim,). Currently
                        unused — scores are distance-based. Reserved for future
                        geodesic-distance initialisation.
        candidates: List of candidate dicts (each has 'distance').
        graph: nx.Graph from build_retrieval_graph.
        n_rounds: Number of message-passing rounds (DECISION-9 default=2).
        self_weight: Weight for candidate's own score each round.
        neighbor_weight: Weight for neighbor contribution each round.

    Returns:
        Updated scores array, shape (len(candidates),), higher = better match.
    """
    scores = _initial_scores(candidates)

    for _ in range(n_rounds):
        new_scores = scores.copy()
        for node in graph.nodes:
            neighbors = list(graph[node])
            if not neighbors:
                continue
            total_weight = sum(graph[node][nb]["weight"] for nb in neighbors)
            neighbor_contrib = sum(
                graph[node][nb]["weight"] * scores[nb] for nb in neighbors
            ) / (total_weight + 1e-8)
            new_scores[node] = self_weight * scores[node] + neighbor_weight * neighbor_contrib
        scores = new_scores

    return scores
