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
        model: Assembled model object with:
                 .preprocess      — torchvision transform (PIL → CHW tensor)
                 .backbone        — frozen BioCLIP-2 (+LoRA) encoder
                 .hyperbolic_proj — HyperbolicProjection layer
                 .heads           — HierarchicalHeads
                 .text_encoder    — optional text encoder (callable, str → (1, D) tensor)
        habitat_text: Optional habitat description string.
        text_fusion_weight: Weight for text embedding (DECISION-7, default 0.2).
        device: Target device string.

    Returns:
        Dict with keys:
            'euclidean':     Tensor (1, embed_dim) — fused Euclidean features
            'poincare':      Tensor (1, hyperbolic_dim) — Poincaré ball point
            'family_probs':  Tensor (1, n_families) — softmax probabilities
            'genus_probs':   Tensor (1, n_genera)
            'species_probs': Tensor (1, n_species)
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        # --- Image encoding ---
        img_tensor = model.preprocess(image).unsqueeze(0).to(device)  # (1, C, H, W)
        img_embed = model.backbone(img_tensor)                          # (1, embed_dim)

        # --- Optional text fusion (DECISION-7: weighted average) ---
        if habitat_text and hasattr(model, "text_encoder"):
            text_embed = model.encode_text(habitat_text).to(device)    # (1, embed_dim)
            combined = (1.0 - text_fusion_weight) * img_embed + text_fusion_weight * text_embed
        else:
            combined = img_embed  # (1, embed_dim)

        # --- Hyperbolic projection ---
        poincare = model.hyperbolic_proj(combined)                      # (1, hyper_dim)

        # --- Hierarchical classifier heads ---
        fam_logits, gen_logits, spe_logits = model.heads(combined)

    return {
        "euclidean":     combined,
        "poincare":      poincare,
        "family_probs":  F.softmax(fam_logits, dim=-1),
        "genus_probs":   F.softmax(gen_logits, dim=-1),
        "species_probs": F.softmax(spe_logits, dim=-1),
    }
