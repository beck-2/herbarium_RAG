"""
Hierarchical classifier heads (family → genus → species).

Responsibilities:
- Three linear heads operating on Euclidean features (before hyperbolic projection)
- Fine-to-coarse conditioning following BiLT (AAAI 2025): species → genus → family
  Genus head sees concatenation of features + species softmax; similarly for family.
- Output raw logits for family (~500 classes), genus (~4000), species (~15501)
- Heads trained jointly with the rest of the model in Phase 1

From SPEC §4.4:
    Family head:  ~500 output classes (NA families)   → Stage 2 confidence gating
    Genus head:   ~4000 output classes (NA genera)    → Stage 2b re-retrieval trigger
    Species head: ~15501 output classes (NA species)  → Final confidence bar
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalHeads(nn.Module):
    """Three-level hierarchical classifier: species → genus (conditioned) → family (conditioned).

    Following BiLT (AAAI 2025) fine-to-coarse conditioning:
    - species_head: linear(in_dim → n_species)
    - genus_head:   linear(in_dim + n_species → n_genera)  [conditioned on species probs]
    - family_head:  linear(in_dim + n_genera  → n_families) [conditioned on genus probs]

    At inference, pass softmax(species_logits) into genus head, etc.
    During training the same conditioning is applied end-to-end (gradients flow through).

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
        self.species_head = nn.Linear(in_dim, n_species)
        self.genus_head = nn.Linear(in_dim + n_species, n_genera)
        self.family_head = nn.Linear(in_dim + n_genera, n_families)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical logits with fine-to-coarse conditioning.

        Args:
            x: Euclidean features of shape (B, in_dim).

        Returns:
            (family_logits, genus_logits, species_logits)
            Each of shape (B, n_classes) — raw logits, not softmax.
        """
        species_logits = self.species_head(x)
        species_probs = species_logits.softmax(dim=-1).detach()

        genus_logits = self.genus_head(torch.cat([x, species_probs], dim=-1))
        genus_probs = genus_logits.softmax(dim=-1).detach()

        family_logits = self.family_head(torch.cat([x, genus_probs], dim=-1))

        return family_logits, genus_logits, species_logits
