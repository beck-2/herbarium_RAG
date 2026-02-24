"""
Hierarchical classifier heads (family → genus → species).

Responsibilities:
- Three independent linear heads operating on Euclidean features (before hyperbolic projection)
- Fine-to-coarse conditioning following BiLT (AAAI 2025): species head → genus head
- Output softmax logits for family (~500 classes), genus (~4000), species (~15501)
- Heads are trained jointly with the rest of the model in Phase 1 and frozen per-region in Phase 2

From SPEC §4.4:
    Family head:  ~500 output classes (NA families)   → Stage 2 confidence gating
    Genus head:   ~4000 output classes (NA genera)    → Stage 2b re-retrieval trigger
    Species head: ~15501 output classes (NA species)  → Final confidence bar
"""

from __future__ import annotations

import torch
import torch.nn as nn

# TODO(phase3): implement HierarchicalHeads with BiLT-style genus←species conditioning
# TODO(phase3): implement soft label smoothing (sibling_smoothing fraction to sibling taxa)
# TODO(phase3): load/save head weights separately from backbone for bundle packing


class HierarchicalHeads(nn.Module):
    """Three-level hierarchical classifier: family → genus → species.

    Args:
        in_dim: Input feature dimension (backbone embed_dim before hyperbolic projection).
        n_families: Number of family classes (~500 for North America).
        n_genera: Number of genus classes (~4000 for North America).
        n_species: Number of species classes (~15501 for NAFlora-1M).
    """

    def __init__(
        self,
        in_dim: int = 768,
        n_families: int = 500,
        n_genera: int = 4000,
        n_species: int = 15501,
    ):
        super().__init__()
        # TODO(phase3): define family_head, genus_head, species_head as nn.Linear
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical logits.

        Args:
            x: Euclidean features of shape (B, in_dim).

        Returns:
            (family_logits, genus_logits, species_logits)
            Each of shape (B, n_classes) — raw logits, not softmax.
        """
        raise NotImplementedError
