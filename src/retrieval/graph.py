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

# TODO(phase6): implement build_retrieval_graph with genus_family edge mode
# TODO(phase6): implement aggregate_scores message passing
# TODO(phase6): add full_patristic edge mode using opentree_distances


def build_retrieval_graph(
    candidates: list[dict],
    opentree_subtree: dict,
    edge_mode: str = "genus_family",
) -> nx.Graph:
    """Build a phylogenetic graph over retrieved candidates.

    Args:
        candidates: List of candidate dicts (each has 'taxon', 'family', 'genus').
        opentree_subtree: Subtree dict from bundle's opentree_subtree.json.
        edge_mode: 'genus_family' (default) or 'full_patristic'.

    Returns:
        nx.Graph where nodes are candidate indices and edges have 'weight' attributes.
        Genus-level LCA: weight=1.0. Family-level LCA: weight=0.3.
    """
    raise NotImplementedError


def aggregate_scores(
    query_poincare: np.ndarray,
    candidates: list[dict],
    graph: nx.Graph,
    n_rounds: int = 2,
    self_weight: float = 0.7,
    neighbor_weight: float = 0.3,
) -> np.ndarray:
    """Run message-passing score aggregation over the retrieval graph.

    Args:
        query_poincare: Query Poincaré point, shape (hyperbolic_dim,).
        candidates: List of candidate dicts (each has 'poincare' embedding).
        graph: nx.Graph from build_retrieval_graph.
        n_rounds: Number of message-passing rounds.
        self_weight: Weight for a candidate's own score.
        neighbor_weight: Weight for neighbor score contribution.

    Returns:
        Updated scores array, shape (len(candidates),).
    """
    raise NotImplementedError
