"""
Loss functions for global and regional training.

Combined loss: L = L_hier + α·L_hyperbolic + β·L_taxonomy

L_hier:
    Hierarchical cross-entropy with soft label smoothing.
    Weighted combination: species (0.5) + genus (0.3) + family (0.2).
    label_smoothing=0.1 distributes mass to all classes uniformly;
    full sibling-aware smoothing is deferred to post-Phase-4.

L_hyperbolic:
    Triplet margin loss on geodesic distances in the Poincaré ball.
    For each anchor in the batch, find the hardest positive (same species,
    farthest) and hardest negative (different species, closest), then apply
    max(0, d_pos − d_neg + margin).

L_taxonomy:
    MSE between pairwise embedding distances and patristic distances.
    Only active during Phase 1 global training (beta > 0).
    Computed by TaxonomyGNNRegularizer in src/taxonomy/gnn.py.

Starting weights: α=0.1, β=0.05. Config keys: training.alpha_hyperbolic, training.beta_taxonomy.
"""

from __future__ import annotations

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy across family, genus, and species heads.

    Args:
        sibling_smoothing: Reserved for future sibling-aware label smoothing.
                           Currently unused; uniform label_smoothing=0.1 is applied.
        family_weight:  Loss weight for family head.
        genus_weight:   Loss weight for genus head.
        species_weight: Loss weight for species head (highest — finest granularity).
    """

    def __init__(
        self,
        sibling_smoothing: float = 0.15,
        family_weight: float = 0.2,
        genus_weight: float = 0.3,
        species_weight: float = 0.5,
    ):
        super().__init__()
        self.family_weight = family_weight
        self.genus_weight = genus_weight
        self.species_weight = species_weight
        # Uniform label smoothing as a proxy for sibling smoothing until
        # the full taxonomy-aware smoothing is wired up in Phase 4+.
        self.family_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.genus_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.species_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        family_logits: torch.Tensor,
        genus_logits: torch.Tensor,
        species_logits: torch.Tensor,
        family_labels: torch.Tensor,
        genus_labels: torch.Tensor,
        species_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return combined weighted hierarchical loss scalar."""
        l_fam = self.family_ce(family_logits, family_labels)
        l_gen = self.genus_ce(genus_logits, genus_labels)
        l_spe = self.species_ce(species_logits, species_labels)
        return (
            self.family_weight * l_fam
            + self.genus_weight * l_gen
            + self.species_weight * l_spe
        )


class HyperbolicMarginLoss(nn.Module):
    """Triplet margin loss on geodesic distances in the Poincaré ball.

    For each sample in the batch:
    - Positive: same species, farthest geodesic distance (hardest positive).
    - Negative: different species, closest geodesic distance (hardest negative).
    - Loss: max(0, d_pos − d_neg + margin).

    If a sample has no valid positive or no valid negative in the batch,
    it is skipped.  Returns zero loss if no valid triplets exist.

    Args:
        margin: Separation between positive and negative distance targets.
        curvature: Poincaré ball curvature (must match HyperbolicProjection).
    """

    def __init__(self, margin: float = 0.5, curvature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.curvature = curvature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hyperbolic triplet margin loss.

        Args:
            embeddings: Poincaré ball points of shape (B, hyperbolic_dim).
                        All norms must be < 1 (enforced by HyperbolicProjection).
            labels:     Species label indices of shape (B,).

        Returns:
            Scalar loss tensor (≥ 0).
        """
        B = embeddings.size(0)
        if B < 2:
            return embeddings.sum() * 0.0  # differentiable zero

        ball = geoopt.PoincareBall(c=float(self.curvature))

        # Compute full B×B pairwise geodesic distance matrix
        # geoopt dist is defined for pairs; vectorise with broadcasting
        emb_i = embeddings.unsqueeze(1).expand(B, B, -1)  # (B, B, D)
        emb_j = embeddings.unsqueeze(0).expand(B, B, -1)  # (B, B, D)
        # Flatten to (B*B, D), compute dist, reshape to (B, B)
        dist_matrix = ball.dist(
            emb_i.reshape(B * B, -1),
            emb_j.reshape(B * B, -1),
        ).reshape(B, B)

        same_class = labels.unsqueeze(1) == labels.unsqueeze(0)   # (B, B)
        diff_class = ~same_class
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        same_class_no_diag = same_class & ~eye

        triplet_losses = []
        for i in range(B):
            pos_mask = same_class_no_diag[i]
            neg_mask = diff_class[i]
            if not pos_mask.any() or not neg_mask.any():
                continue
            d_pos = dist_matrix[i][pos_mask].max()    # hardest positive
            d_neg = dist_matrix[i][neg_mask].min()    # hardest negative
            triplet_losses.append(F.relu(d_pos - d_neg + self.margin))

        if not triplet_losses:
            return embeddings.sum() * 0.0

        return torch.stack(triplet_losses).mean()


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
