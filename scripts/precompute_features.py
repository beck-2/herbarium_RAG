#!/usr/bin/env python
"""
Pre-compute BioCLIP-2 features for all specimens and save to disk.

Run this ONCE on Colab (T4 GPU). Then training uses the cached features
instead of re-downloading and re-encoding every epoch.

Output files (save to Google Drive for reuse):
    features.npy          float16 array (N, 768)
    feature_ids.json      list of occurrence_ids in same row order
    feature_labels.npy    int32 array (N, 3) = [family_idx, genus_idx, species_idx]

Usage (Colab):
    python scripts/precompute_features.py \\
        --parquet data/processed/regions/california/specimens_encoded.parquet \\
        --manifest data/processed/regions/california/train_full.txt \\
        --output /content/drive/MyDrive/hyperbolic_herbarium/features/ \\
        --device cuda --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",   default="data/processed/regions/california/specimens_encoded.parquet")
    p.add_argument("--manifest",  default="data/processed/regions/california/train_full.txt")
    p.add_argument("--output",    default="data/processed/regions/california/features/")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--limit",     type=int, default=None, help="Only encode first N specimens (for smoke tests)")
    return p.parse_args()


def main():
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    from src.data.dataset import StreamingSpecimenDataset, streaming_collate_fn
    from src.model.backbone import load_backbone

    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    if (out_dir / "features.npy").exists():
        ids = json.loads((out_dir / "feature_ids.json").read_text())
        print(f"Features already exist: {len(ids):,} vectors. Delete to recompute.")
        return

    # Load parquet + manifest
    df = pd.read_parquet(args.parquet)
    if Path(args.manifest).exists():
        ids_in_manifest = set(Path(args.manifest).read_text().strip().splitlines())
        df = df[df["occurrence_id"].isin(ids_in_manifest)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Specimens to encode: {len(df):,}")

    # Load backbone
    backbone_cfg_path = ROOT / "config" / "backbone.yaml"
    with open(backbone_cfg_path) as f:
        backbone_yaml = yaml.safe_load(f)
    active = backbone_yaml.get("active", "bioclip2")
    bb_cfg = backbone_yaml["profiles"][active]
    embed_dim = bb_cfg.get("embed_dim", 768)

    device = args.device
    bb_dtype = torch.float16 if "cuda" in device else torch.float32
    image_encoder, _, preprocess_fn = load_backbone(bb_cfg, device=device, dtype=bb_dtype)
    image_encoder.eval()
    print(f"Backbone loaded on {device} (embed_dim={embed_dim})")

    # Dataset: stream images, no labels needed here
    records = df[["image_url", "family_idx", "genus_idx", "species_idx"]].to_dict("records")
    ds = StreamingSpecimenDataset(records, transform=preprocess_fn)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=streaming_collate_fn,
        pin_memory=False, prefetch_factor=2, persistent_workers=True,
    )

    # Pre-allocate output arrays
    all_features = np.zeros((len(df), embed_dim), dtype=np.float16)
    all_labels   = np.zeros((len(df), 3), dtype=np.int32)
    occurrence_ids = df["occurrence_id"].tolist()

    row = 0
    t0 = time.time()
    skipped = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                skipped += args.batch_size
                continue
            images, fam, gen, spe = batch
            images = images.to(device, dtype=bb_dtype)
            feats = image_encoder(images).to(torch.float16).cpu().numpy()

            bs = feats.shape[0]
            all_features[row:row + bs] = feats
            all_labels[row:row + bs, 0] = fam.numpy()
            all_labels[row:row + bs, 1] = gen.numpy()
            all_labels[row:row + bs, 2] = spe.numpy()
            row += bs

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = row / elapsed
                remaining = (len(df) - row) / max(rate, 1)
                print(f"  [{row:,}/{len(df):,}] {rate:.0f} img/s | "
                      f"~{remaining/60:.0f} min remaining"
                      f"{f' | skipped~{skipped}' if skipped else ''}",
                      flush=True)

    # Trim to actual rows encoded (some may have been skipped)
    all_features = all_features[:row]
    all_labels   = all_labels[:row]
    occurrence_ids = occurrence_ids[:row]

    # Save
    np.save(str(out_dir / "features.npy"), all_features)
    np.save(str(out_dir / "feature_labels.npy"), all_labels)
    (out_dir / "feature_ids.json").write_text(json.dumps(occurrence_ids))

    elapsed = time.time() - t0
    size_mb = all_features.nbytes / 1e6
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"  {row:,} features saved ({size_mb:.0f} MB float16)")
    print(f"  {out_dir}/features.npy")


if __name__ == "__main__":
    main()
