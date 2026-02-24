"""
Loss functions for global and regional training.

Combined loss: L = L_hier + α·L_hyperbolic + β·L_taxonomy

L_hier:
    Hierarchical cross-entropy with soft label smoothing.
    10–20% probability mass distributed to sibling taxa (same genus),
    smaller fraction to parent genus/family.
    Fine-to-coarse weighting: species > genus > family.

L_hyperbolic:
    Margin loss on geodesic distances in Poincaré ball.
    Positive pairs (same species) pulled together,
    negative pairs (different species) pushed apart.

L_taxonomy:
    MSE between pairwise embedding distances and patristic distances.
    Only active during Phase 1 global training.
    Computed by TaxonomyGNNRegularizer in src/taxonomy/gnn.py.

Starting weights: α=0.1, β=0.05. Config keys: training.alpha_hyperbolic, training.beta_taxonomy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(phase3): implement HierarchicalCrossEntropyLoss with soft sibling smoothing
# TODO(phase3): implement HyperbolicMarginLoss on geodesic distances
# TODO(phase4): integrate with TaxonomyGNNRegularizer for L_taxonomy


class HierarchicalCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with soft label smoothing across taxonomy hierarchy.

    Args:
        sibling_smoothing: Fraction of probability mass to distribute to sibling taxa.
        family_weight: Loss weight for family head.
        genus_weight: Loss weight for genus head.
        species_weight: Loss weight for species head.
    """

    def __init__(
        self,
        sibling_smoothing: float = 0.15,
        family_weight: float = 0.2,
        genus_weight: float = 0.3,
        species_weight: float = 0.5,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        family_logits: torch.Tensor,
        genus_logits: torch.Tensor,
        species_logits: torch.Tensor,
        family_labels: torch.Tensor,
        genus_labels: torch.Tensor,
        species_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return combined hierarchical loss scalar."""
        raise NotImplementedError


class HyperbolicMarginLoss(nn.Module):
    """Margin loss on geodesic distances in the Poincaré ball.

    Positive pairs: same species (distance < margin_pos).
    Negative pairs: different species (distance > margin_neg).

    Args:
        margin: Separation between positive and negative distance targets.
        manifold: geoopt PoincareBall manifold instance.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hyperbolic margin loss.

        Args:
            embeddings: Poincaré ball points, shape (B, hyperbolic_dim).
            labels: Species label indices, shape (B,).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError


def combined_loss(
    hier_loss: torch.Tensor,
    hyp_loss: torch.Tensor,
    tax_loss: torch.Tensor | None,
    alpha: float = 0.1,
    beta: float = 0.05,
) -> torch.Tensor:
    """Combine loss terms: L = L_hier + α·L_hyperbolic + β·L_taxonomy."""
    loss = hier_loss + alpha * hyp_loss
    if tax_loss is not None:
        loss = loss + beta * tax_loss
    return loss
