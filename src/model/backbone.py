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

import torch
import torch.nn as nn

# TODO(phase3): implement backbone loading via open_clip.create_model_and_transforms
# TODO(phase3): implement int8 quantization path for inference bundle export
# TODO(phase3): expose embed_dim from backbone config (768 for ViT-L/14, 512 for ViT-B/16)


def load_backbone(
    backbone_config: dict,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, nn.Module, object]:
    """Load image encoder, text encoder, and image preprocessor.

    Args:
        backbone_config: Single profile dict from backbone.yaml (e.g. the 'bioclip2' entry).
        device: Target device ('cpu', 'cuda').
        dtype: Weight dtype (torch.float16 for training, torch.float32 for debug).

    Returns:
        (image_encoder, text_encoder, preprocess_fn)
        image_encoder: callable, input PIL image → (1, embed_dim) tensor
        text_encoder:  callable, input token tensor → (1, embed_dim) tensor
        preprocess_fn: torchvision transform for input images
    """
    raise NotImplementedError


def freeze_backbone(image_encoder: nn.Module) -> None:
    """Freeze all parameters in the backbone (LoRA adapters remain trainable)."""
    for p in image_encoder.parameters():
        p.requires_grad_(False)
