"""
LoRA adapter injection via the peft library.

Responsibilities:
- Inject LoRA adapters into the frozen BioCLIP-2 image encoder
- Target modules: out_proj, c_fc, c_proj (see DECISION-15)
- Support loading and saving adapter weights in safetensors format
- Validate that only LoRA parameters are trainable after injection

DECISION-6: LoRA rank 16 (default). Config key: lora.rank in default.yaml.
DECISION-15: BioCLIP-2 uses fused QKV (nn.MultiheadAttention.in_proj_weight).
             Target out_proj + c_fc + c_proj instead of q_proj + v_proj + out_proj.
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

_DEFAULT_TARGET_MODULES = ["out_proj", "c_fc", "c_proj"]


def inject_lora(
    image_encoder: nn.Module,
    lora_config: dict,
) -> PeftModel:
    """Wrap image_encoder with LoRA adapters using peft.

    Args:
        image_encoder: Frozen BioCLIP-2 image encoder (model.visual from load_backbone).
        lora_config: Dict from default.yaml `lora` section:
            {
                'rank': int,
                'alpha': int,
                'dropout': float,
                'target_modules': list[str],  # default: ['out_proj', 'c_fc', 'c_proj']
            }

    Returns:
        peft.PeftModel wrapping the image encoder.
        Only LoRA parameters have requires_grad=True.
    """
    config = LoraConfig(
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get("target_modules", _DEFAULT_TARGET_MODULES),
        lora_dropout=lora_config.get("dropout", 0.1),
        bias="none",
    )
    return get_peft_model(image_encoder, config)


def save_adapter(peft_model: PeftModel, output_path: str) -> None:
    """Save LoRA adapter weights to output_path directory in safetensors format."""
    peft_model.save_pretrained(output_path)


def load_adapter(base_encoder: nn.Module, adapter_path: str) -> PeftModel:
    """Load a saved LoRA adapter onto a base encoder.

    Args:
        base_encoder: Frozen base image encoder (not yet wrapped with peft).
        adapter_path: Path to directory containing adapter_model.safetensors.

    Returns:
        peft.PeftModel with adapter weights loaded.
    """
    return PeftModel.from_pretrained(base_encoder, adapter_path)


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
