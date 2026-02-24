"""
Phase 2: Regional LoRA adapter training.

Starting from the global baseline checkpoint, injects LoRA adapters into the
BioCLIP-2 ViT-L/14 image encoder and fine-tunes them on regional specimen data.
The taxonomy GNN regularizer (L_taxonomy) is NOT active in this phase (per-region
patristic distances are not worth the overhead).

Outputs: checkpoints/lora/{region}/adapter_model.safetensors

Usage (from SPEC §6.2):
    python src/train/train_regional_lora.py \\
        --base-checkpoint checkpoints/global/best.pt \\
        --region california \\
        --lora-rank 16 \\
        --lora-alpha 32 \\
        --data data/processed/regions/california/ \\
        --epochs 8 \\
        --output checkpoints/lora/california/
"""

from __future__ import annotations

import argparse

# TODO(phase4): implement argument parsing
# TODO(phase4): load global checkpoint → inject LoRA → fine-tune on regional data
# TODO(phase4): implement early stopping on val species accuracy for rare taxa
# TODO(phase4): save adapter weights in safetensors format


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for regional LoRA training."""
    raise NotImplementedError


def train(args: argparse.Namespace) -> None:
    """Run regional LoRA fine-tuning."""
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    train(args)
