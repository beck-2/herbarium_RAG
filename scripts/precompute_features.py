#!/usr/bin/env python
"""
Pre-compute BioCLIP-2 features for all specimens and save to disk.

Supports resuming after Colab disconnects — saves a checkpoint to Drive every
--save-every images and picks up from where it left off on restart.

Output files:
    features.npy          float16 array (N, 768)
    feature_ids.json      list of occurrence_ids in same row order
    feature_labels.npy    int32 array (N, 3) = [family_idx, genus_idx, species_idx]

Usage (Colab, full run with resume support):
    python scripts/precompute_features.py \\
        --parquet data/processed/regions/california/specimens_encoded.parquet \\
        --manifest data/processed/regions/california/train_full.txt \\
        --output /content/drive/MyDrive/hyperbolic_herbarium/features/ \\
        --device cuda --batch-size 256 --download-threads 64 --save-every 5000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",   default="data/processed/regions/california/specimens_encoded.parquet")
    p.add_argument("--manifest",  default="data/processed/regions/california/train_full.txt")
    p.add_argument("--output",    default="data/processed/regions/california/features/")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--download-threads", type=int, default=64)
    p.add_argument("--timeout",   type=int, default=15)
    p.add_argument("--save-every", type=int, default=5000,
                   help="Save checkpoint to output dir every N encoded images")
    p.add_argument("--limit",     type=int, default=None)
    return p.parse_args()


def fetch_image(args_tuple):
    idx, url, transform, timeout = args_tuple
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HyperbolicHerbarium/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            from PIL import Image
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        return idx, transform(img)
    except Exception:
        return idx, None


def save_checkpoint(out_dir, all_features, all_labels, occurrence_ids, row):
    import numpy as np
    np.save(str(out_dir / "features.npy"),       all_features[:row])
    np.save(str(out_dir / "feature_labels.npy"), all_labels[:row])
    (out_dir / "feature_ids.json").write_text(json.dumps(occurrence_ids[:row]))
    (out_dir / "progress.json").write_text(json.dumps({"row": row}))


def main():
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    from src.model.backbone import load_backbone

    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet + manifest
    df = pd.read_parquet(args.parquet)
    if Path(args.manifest).exists():
        ids_in_manifest = set(Path(args.manifest).read_text().strip().splitlines())
        df = df[df["occurrence_id"].isin(ids_in_manifest)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    # Resume from checkpoint if available
    start_row = 0
    if (out_dir / "progress.json").exists():
        start_row = json.loads((out_dir / "progress.json").read_text())["row"]
        print(f"Resuming from row {start_row:,} / {len(df):,}")
        df = df.iloc[start_row:].reset_index(drop=True)
    else:
        print(f"Specimens to encode: {len(df):,}")

    if len(df) == 0:
        print("Already complete.")
        return

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
    print(f"Download threads: {args.download_threads} | Batch size: {args.batch_size} | Save every: {args.save_every:,}")

    records = df[["image_url", "family_idx", "genus_idx", "species_idx"]].to_dict("records")
    occurrence_ids_new = df["occurrence_id"].tolist()
    n = len(records)

    # Load existing arrays if resuming, else allocate fresh
    total_n = start_row + n
    if start_row > 0:
        all_features = np.zeros((total_n, embed_dim), dtype=np.float16)
        all_labels   = np.zeros((total_n, 3), dtype=np.int32)
        all_features[:start_row] = np.load(str(out_dir / "features.npy"))
        all_labels[:start_row]   = np.load(str(out_dir / "feature_labels.npy"))
        occurrence_ids = json.loads((out_dir / "feature_ids.json").read_text()) + occurrence_ids_new
    else:
        all_features = np.zeros((total_n, embed_dim), dtype=np.float16)
        all_labels   = np.zeros((total_n, 3), dtype=np.int32)
        occurrence_ids = occurrence_ids_new

    row = start_row
    skipped = 0
    t0 = time.time()
    last_save = row

    with ThreadPoolExecutor(max_workers=args.download_threads) as pool:
        for batch_start in range(0, n, args.batch_size):
            batch_records = records[batch_start: batch_start + args.batch_size]

            tasks = [(i, rec["image_url"], preprocess_fn, args.timeout)
                     for i, rec in enumerate(batch_records)]
            results = [None] * len(batch_records)
            for i, tensor in pool.map(fetch_image, tasks):
                results[i] = tensor

            valid_indices = [i for i, t in enumerate(results) if t is not None]
            skipped += len(batch_records) - len(valid_indices)
            if not valid_indices:
                continue

            images = torch.stack([results[i] for i in valid_indices]).to(device, dtype=bb_dtype)
            with torch.no_grad():
                feats = image_encoder(images).to(torch.float16).cpu().numpy()

            bs = feats.shape[0]
            all_features[row:row + bs] = feats
            for j, src_i in enumerate(valid_indices):
                rec = batch_records[src_i]
                all_labels[row + j, 0] = rec["family_idx"]
                all_labels[row + j, 1] = rec["genus_idx"]
                all_labels[row + j, 2] = rec["species_idx"]
            row += bs

            # Progress
            elapsed = time.time() - t0
            rate = (row - start_row) / elapsed
            remaining = (total_n - row) / max(rate, 1)
            print(f"  [{row:,}/{total_n:,}] {rate:.0f} img/s | "
                  f"~{remaining/60:.0f} min remaining"
                  f"{f' | skipped={skipped}' if skipped else ''}",
                  flush=True)

            # Periodic checkpoint save
            if row - last_save >= args.save_every:
                save_checkpoint(out_dir, all_features, all_labels, occurrence_ids, row)
                print(f"  ✓ Checkpoint saved ({row:,} rows)", flush=True)
                last_save = row

    # Final save
    save_checkpoint(out_dir, all_features, all_labels, occurrence_ids, row)
    elapsed = time.time() - t0
    size_mb = all_features[:row].nbytes / 1e6
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"  {row:,} features saved ({size_mb:.0f} MB float16)")
    print(f"  {out_dir}/features.npy")


if __name__ == "__main__":
    main()
