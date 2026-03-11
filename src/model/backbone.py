"""
BioCLIP-2 backbone loader.

Responsibilities:
- Load BioCLIP-2 (or BioCLIP-1) from HuggingFace via open_clip_torch
- Expose image encoder and text encoder separately
- Support fp16 and int8 quantization for inference bundles
- Freeze backbone weights for training (LoRA adapts them via peft, not direct grad)
- Select active backbone from config/backbone.yaml

DECISION-1: BioCLIP-2 ViT-L/14 (default) vs BioCLIP-1 ViT-B/16.
            Active profile set in config/backbone.yaml under `active`.
"""

from __future__ import annotations

import open_clip
import torch
import torch.nn as nn


def load_backbone(
    backbone_config: dict,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, object, object]:
    """Load image encoder, full CLIP model (for text encoding), and image preprocessor.

    The image encoder (model.visual) is frozen immediately after loading.
    Inject LoRA adapters via lora.inject_lora() before training.

    Args:
        backbone_config: Single profile dict, e.g.:
            {'model_id': 'hf-hub:imageomics/bioclip-2', 'embed_dim': 768}
        device: Target device ('cpu', 'cuda').
        dtype: Weight dtype (torch.float16 for training, torch.float32 for debug).

    Returns:
        (image_encoder, clip_model, preprocess_fn)
        image_encoder: model.visual — nn.Module, input image tensor → (B, embed_dim).
                       Frozen (no requires_grad). Wrap with inject_lora() for training.
        clip_model:    Full open_clip CLIP model. Use clip_model.encode_text(tokens)
                       for text embeddings.
        preprocess_fn: Validation/inference torchvision transform (no random augments).
    """
    model_id = backbone_config.get("model_id", "hf-hub:imageomics/bioclip-2")

    # create_model_and_transforms returns (model, preprocess_train, preprocess_val)
    clip_model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_id
    )
    clip_model = clip_model.to(device=device, dtype=dtype)
    clip_model.eval()

    image_encoder = clip_model.visual
    freeze_backbone(image_encoder)

    return image_encoder, clip_model, preprocess_val


def freeze_backbone(image_encoder: nn.Module) -> None:
    """Freeze all parameters in the backbone (LoRA adapters remain trainable)."""
    for p in image_encoder.parameters():
        p.requires_grad_(False)
