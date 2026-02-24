"""
Taxonomy GNN regularizer (training only — NOT used at inference).

Responsibilities:
- Build the taxonomy graph: nodes = taxa (OTT IDs), edges = parent-child in OpenTree
- Run 1–2 rounds of message passing over class prototype embeddings
- Compute taxonomy regularization loss:
    L_taxonomy = MSE(embed_dist_matrix, patristic_dist_matrix)
- Enforce that species nearby in the phylogenetic tree have similar embeddings

This module is imported only during training (src/train/).
It has no inference-time dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# TODO(phase3): implement TaxonomyGraph construction from OpenTree subtree
# TODO(phase3): implement message-passing GNN layer (simple mean aggregation)
# TODO(phase3): implement L_taxonomy loss computation


class TaxonomyGNNRegularizer(nn.Module):
    """GNN regularizer that enforces phylogenetic structure on class embeddings.

    Used as a loss term during training. Given class prototype embeddings
    (mean embedding per taxon), penalizes deviations from patristic distances.
    """

    def __init__(self, patristic_distances: dict[tuple[int, int], float]):
        """Initialize with precomputed patristic distances.

        Args:
            patristic_distances: Dict mapping (ott_id_a, ott_id_b) → float distance.
        """
        super().__init__()
        # TODO(phase3): store distance matrix as a buffer
        raise NotImplementedError

    def forward(
        self,
        embeddings: torch.Tensor,
        ott_ids: list[int],
    ) -> torch.Tensor:
        """Compute L_taxonomy loss.

        Args:
            embeddings: Tensor of shape (N, D) — one embedding per taxon.
            ott_ids: List of OTT IDs corresponding to each embedding row.

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError
