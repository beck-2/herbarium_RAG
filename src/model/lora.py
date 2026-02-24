"""
LoRA adapter injection via the peft library.

Responsibilities:
- Inject LoRA adapters into the frozen BioCLIP-2 image encoder
- Target q_proj, v_proj, out_proj in all ViT attention blocks
- Support loading and saving adapter weights in safetensors format
- Validate that only LoRA parameters are trainable after injection

DECISION-6: LoRA rank 16 (default). Config key: lora.rank in default.yaml.
DECISION-1: Target modules differ between ViT-L/14 and ViT-B/16; handle both.
"""

from __future__ import annotations

import torch.nn as nn

# TODO(phase3): implement LoRA injection via peft.get_peft_model + LoraConfig
# TODO(phase3): implement adapter save (safetensors) and load
# TODO(phase3): validate trainable parameter count after injection


def inject_lora(
    image_encoder: nn.Module,
    lora_config: dict,
) -> nn.Module:
    """Wrap image_encoder with LoRA adapters using peft.

    Args:
        image_encoder: Frozen BioCLIP-2 image encoder (output of load_backbone).
        lora_config: Dict from default.yaml `lora` section:
            {rank, alpha, dropout, target_modules}.

    Returns:
        peft.PeftModel wrapping the image encoder.
        Only LoRA parameters have requires_grad=True.
    """
    raise NotImplementedError


def save_adapter(peft_model: nn.Module, output_path: str) -> None:
    """Save LoRA adapter weights to output_path in safetensors format."""
    raise NotImplementedError


def load_adapter(base_encoder: nn.Module, adapter_path: str) -> nn.Module:
    """Load a saved LoRA adapter onto a base encoder.

    Args:
        base_encoder: Frozen base image encoder (not yet wrapped with peft).
        adapter_path: Path to directory containing adapter_model.safetensors.

    Returns:
        peft.PeftModel with adapter weights loaded.
    """
    raise NotImplementedError


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
