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

Smoke test (no data required):
    python src/train/train_global.py --smoke-test --epochs 3 --output /tmp/smoke/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Checkpoint helpers (also used by tests)
# ---------------------------------------------------------------------------

def save_checkpoint(
    proj: nn.Module,
    heads: nn.Module,
    path: str,
    epoch: int,
    val_loss: float,
) -> None:
    """Save projection + heads state dicts with metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "proj_state": proj.state_dict(),
            "heads_state": heads.state_dict(),
        },
        path,
    )


def load_checkpoint(
    proj: nn.Module,
    heads: nn.Module,
    path: str,
) -> dict:
    """Load projection + heads weights from checkpoint. Returns metadata dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    proj.load_state_dict(ckpt["proj_state"])
    heads.load_state_dict(ckpt["heads_state"])
    return {"epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"]}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    proj: nn.Module,
    heads: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    hier_loss_fn,
    hyp_loss_fn,
    alpha: float,
    device: str,
) -> float:
    from src.train.loss import combined_loss

    proj.train()
    heads.train()
    total_loss = 0.0
    steps = 0

    for feats, fam_lbl, gen_lbl, spe_lbl in loader:
        feats = feats.to(device)
        fam_lbl = fam_lbl.to(device)
        gen_lbl = gen_lbl.to(device)
        spe_lbl = spe_lbl.to(device)

        optimizer.zero_grad()
        poincare = proj(feats)
        fam_logits, gen_logits, spe_logits = heads(feats)

        l_hier = hier_loss_fn(fam_logits, gen_logits, spe_logits,
                              fam_lbl, gen_lbl, spe_lbl)
        l_hyp = hyp_loss_fn(poincare, spe_lbl)
        loss = combined_loss(l_hier, l_hyp, tax_loss=None, alpha=alpha)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(proj.parameters()) + list(heads.parameters()), max_norm=1.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(
    proj: nn.Module,
    heads: nn.Module,
    loader: DataLoader,
    hier_loss_fn,
    hyp_loss_fn,
    alpha: float,
    device: str,
) -> dict:
    from src.train.loss import combined_loss

    proj.eval()
    heads.eval()
    total_loss = 0.0
    correct_fam = correct_gen = correct_spe = total = 0

    for feats, fam_lbl, gen_lbl, spe_lbl in loader:
        feats = feats.to(device)
        fam_lbl = fam_lbl.to(device)
        gen_lbl = gen_lbl.to(device)
        spe_lbl = spe_lbl.to(device)

        poincare = proj(feats)
        fam_logits, gen_logits, spe_logits = heads(feats)

        l_hier = hier_loss_fn(fam_logits, gen_logits, spe_logits,
                              fam_lbl, gen_lbl, spe_lbl)
        l_hyp = hyp_loss_fn(poincare, spe_lbl)
        loss = combined_loss(l_hier, l_hyp, tax_loss=None, alpha=alpha)
        total_loss += loss.item()

        correct_fam += (fam_logits.argmax(dim=-1) == fam_lbl).sum().item()
        correct_gen += (gen_logits.argmax(dim=-1) == gen_lbl).sum().item()
        correct_spe += (spe_logits.argmax(dim=-1) == spe_lbl).sum().item()
        total += feats.size(0)

    n = max(total, 1)
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "family_acc": correct_fam / n,
        "genus_acc": correct_gen / n,
        "species_acc": correct_spe / n,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for global training."""
    p = argparse.ArgumentParser(description="Global baseline training")
    p.add_argument("--backbone", default="hf-hub:imageomics/bioclip-2")
    p.add_argument("--dataset", default="data/processed/naflora1m_capped/",
                   help="Directory with train.txt, val.txt, specimens.parquet")
    p.add_argument("--hyperbolic-dim", type=int, default=512)
    p.add_argument("--curvature", type=float, default=-1.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Weight for hyperbolic margin loss")
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--output", default="checkpoints/global/")
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke-test", action="store_true",
                   help="Use synthetic data (no images needed). Overrides --dataset.")
    p.add_argument("--smoke-n-samples", type=int, default=10_000)
    p.add_argument("--smoke-n-species", type=int, default=50)
    p.add_argument("--smoke-n-genera", type=int, default=20)
    p.add_argument("--smoke-n-families", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=768,
                   help="Backbone output dim. 768 for ViT-L/14, 512 for ViT-B/16.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Run global baseline training (or smoke test)."""
    from src.model.hyperbolic import HyperbolicProjection
    from src.model.heads import HierarchicalHeads
    from src.train.loss import HierarchicalCrossEntropyLoss, HyperbolicMarginLoss

    device = args.device
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if args.smoke_test:
        from src.data.dataset import SyntheticSpecimenDataset
        n_val = max(args.smoke_n_samples // 5, 32)
        train_ds = SyntheticSpecimenDataset(
            n_samples=args.smoke_n_samples,
            embed_dim=args.embed_dim,
            n_families=args.smoke_n_families,
            n_genera=args.smoke_n_genera,
            n_species=args.smoke_n_species,
            seed=0,
        )
        val_ds = SyntheticSpecimenDataset(
            n_samples=n_val,
            embed_dim=args.embed_dim,
            n_families=args.smoke_n_families,
            n_genera=args.smoke_n_genera,
            n_species=args.smoke_n_species,
            seed=1,
        )
        n_families = args.smoke_n_families
        n_genera = args.smoke_n_genera
        n_species = args.smoke_n_species
        embed_dim = args.embed_dim
        print(f"[smoke-test] {len(train_ds)} train / {len(val_ds)} val synthetic specimens")
    else:
        # Real dataset path — backbone encodes images on the fly
        # (requires downloaded NAFlora images; see DECISION-14)
        raise NotImplementedError(
            "Real dataset training not yet implemented. Use --smoke-test to run "
            "the training loop on synthetic data."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    proj = HyperbolicProjection(
        in_dim=embed_dim,
        out_dim=args.hyperbolic_dim,
        curvature=args.curvature,
    ).to(device)
    heads = HierarchicalHeads(
        in_dim=embed_dim,
        n_families=n_families,
        n_genera=n_genera,
        n_species=n_species,
    ).to(device)

    params = list(proj.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = _warmup_cosine_scheduler(optimizer, args.warmup_steps, total_steps)

    hier_loss_fn = HierarchicalCrossEntropyLoss()
    hyp_loss_fn = HyperbolicMarginLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(args.output, "best.pt")

    print(f"Training for up to {args.epochs} epochs on {device}.")
    print(f"  proj params: {sum(p.numel() for p in proj.parameters()):,}")
    print(f"  heads params: {sum(p.numel() for p in heads.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            proj, heads, train_loader, optimizer, scheduler,
            hier_loss_fn, hyp_loss_fn, args.alpha, device,
        )
        val_metrics = evaluate(
            proj, heads, val_loader, hier_loss_fn, hyp_loss_fn, args.alpha, device,
        )
        val_loss = val_metrics["val_loss"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"fam={val_metrics['family_acc']:.3f} "
            f"gen={val_metrics['genus_acc']:.3f} "
            f"spe={val_metrics['species_acc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(proj, heads, best_ckpt, epoch=epoch, val_loss=val_loss)
            print(f"  ✓ saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping after {epoch} epochs (patience={args.patience})")
                break

    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
