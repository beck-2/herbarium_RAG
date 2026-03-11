"""
Query encoding — maps an input image (+ optional habitat text) to Poincaré space.

Responsibilities:
- Preprocess query image with backbone-specific transforms
- Encode image with BioCLIP-2 + LoRA adapter → Euclidean features
- Optionally fuse habitat text embedding (DECISION-7: weighted average, weight 0.2)
- Project fused features to Poincaré ball via HyperbolicProjection
- Compute hierarchical classifier logits (family, genus, species)
- Return a structured query dict used by retrieval.search

CRITICAL: No LLM calls in this module. All computation is geometric or linear.

From SPEC §8.1:
    def encode_query(image, habitat_text=None):
        img_embed = backbone(preprocess(image))
        if habitat_text:
            combined = 0.8 * img_embed + 0.2 * text_embed  # DECISION-7
        poincare_point = hyperbolic_proj(combined)
        ...
        return {'euclidean': combined, 'poincare': poincare_point, ...}
"""

from __future__ import annotations

from PIL import Image

# TODO(phase6): implement full encode_query pipeline
# TODO(phase6): implement text tokenization and fusion (DECISION-7 weighted average)


def encode_query(
    image: Image.Image,
    model,
    habitat_text: str | None = None,
    text_fusion_weight: float = 0.2,
    device: str = "cpu",
) -> dict:
    """Encode a query image (and optional habitat text) to Poincaré space.

    Args:
        image: PIL Image of the specimen.
        model: Assembled model object with backbone, lora, hyperbolic_proj, heads.
        habitat_text: Optional habitat description string (e.g. "serpentine soil, chaparral").
        text_fusion_weight: Weight for text embedding in fusion (DECISION-7).
        device: Target device.

    Returns:
        Dict with keys:
            'euclidean':     Tensor (1, embed_dim) — Euclidean features before projection
            'poincare':      Tensor (1, hyperbolic_dim) — Poincaré ball point
            'family_probs':  Tensor (1, n_families) — softmax family probabilities
            'genus_probs':   Tensor (1, n_genera) — softmax genus probabilities
            'species_probs': Tensor (1, n_species) — softmax species probabilities
    """
    import torch  # lazy import — not required at module level
    raise NotImplementedError
