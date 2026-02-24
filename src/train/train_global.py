"""
Phase 1: Global baseline training.

Trains the hyperbolic projection layer and hierarchical classifier heads
on the full NAFlora-1M capped dataset, with the BioCLIP-2 backbone frozen.
The taxonomy GNN regularizer (L_taxonomy) is active in this phase.

Outputs: checkpoints/global/best.pt (projection + heads weights)

Usage (from SPEC §6.2):
    python src/train/train_global.py \\
        --backbone imageomics/bioclip-2 \\
        --dataset data/processed/naflora1m_capped/ \\
        --hyperbolic-dim 512 \\
        --curvature -1.0 \\
        --epochs 15 \\
        --batch-size 64 \\
        --lr 2e-4 \\
        --hierarchical-loss \\
        --taxonomy-gnn \\
        --output checkpoints/global/
"""

from __future__ import annotations

import argparse

# TODO(phase4): implement argument parsing
# TODO(phase4): implement training loop with wandb logging
# TODO(phase4): implement early stopping on val family accuracy
# TODO(phase4): save best checkpoint by val loss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for global training."""
    raise NotImplementedError


def train(args: argparse.Namespace) -> None:
    """Run global baseline training."""
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    train(args)
