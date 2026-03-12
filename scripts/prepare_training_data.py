#!/usr/bin/env python3.14
"""
Prepare CA specimens parquet for training by adding integer label indices.

Reads:  data/processed/regions/california/specimens.parquet
Writes: data/processed/regions/california/specimens_encoded.parquet
        data/processed/regions/california/label_encoders.json
        data/processed/regions/california/train_5k.txt   (5K subset manifest)

Usage:
    python3.14 scripts/prepare_training_data.py
    python3.14 scripts/prepare_training_data.py --parquet data/processed/regions/california/specimens.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=str(ROOT / "data/processed/regions/california/specimens.parquet"))
    p.add_argument("--output-dir", default=str(ROOT / "data/processed/regions/california"))
    p.add_argument("--subset-n", type=int, default=5000, help="Size of local 5K training subset")
    p.add_argument("--subset-seed", type=int, default=42)
    return p.parse_args()


def main():
    import pandas as pd
    from src.data.label_encoder import build_label_encoders, save_label_encoders
    from src.data.parse import save_parquet

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} specimens")

    # Only keep rows with image URLs for training
    has_url = df["image_url"].notna() & df["image_url"].astype(str).str.startswith("http")
    df_img = df[has_url].copy().reset_index(drop=True)
    print(f"  {len(df_img):,} specimens with image URLs")

    # Build and save label encoders
    print("Building label encoders ...")
    encoders = build_label_encoders(df_img)
    for col, enc in encoders.items():
        print(f"  {col}: {len(enc)} classes (including <UNK>)")

    enc_path = out_dir / "label_encoders.json"
    save_label_encoders(encoders, str(enc_path))
    print(f"  Saved: {enc_path}")

    # Add integer index columns
    df_img["family_idx"]  = encoders["family"].transform(df_img["family"]).astype(int)
    df_img["genus_idx"]   = encoders["genus"].transform(df_img["genus"]).astype(int)
    df_img["species_idx"] = encoders["scientific_name"].transform(df_img["scientific_name"]).astype(int)

    enc_parquet = out_dir / "specimens_encoded.parquet"
    save_parquet(df_img, str(enc_parquet))
    print(f"  Saved: {enc_parquet} ({len(df_img):,} rows)")

    # Build 5K subset manifest: stratified by family to cover diversity
    print(f"\nBuilding {args.subset_n}-sample subset ...")
    rng_state = args.subset_seed
    n_families = df_img["family"].nunique()
    per_family = max(1, args.subset_n // n_families)

    subset = (
        df_img.groupby("family", group_keys=False)
        .apply(lambda g: g.sample(min(per_family, len(g)), random_state=rng_state))
    )
    # If we got fewer than subset_n, top up randomly
    if len(subset) < args.subset_n:
        remaining = df_img[~df_img["occurrence_id"].isin(subset["occurrence_id"])]
        extra = remaining.sample(
            min(args.subset_n - len(subset), len(remaining)),
            random_state=rng_state,
        )
        subset = pd.concat([subset, extra], ignore_index=True)
    subset = subset.sample(frac=1, random_state=rng_state).reset_index(drop=True)

    manifest_path = out_dir / f"train_{args.subset_n // 1000}k.txt"
    manifest_path.write_text("\n".join(subset["occurrence_id"].tolist()))
    print(f"  Saved: {manifest_path} ({len(subset):,} occurrence IDs)")
    print(f"  Families covered: {subset['family'].nunique()}")
    print(f"  Species covered:  {subset['scientific_name'].nunique()}")


if __name__ == "__main__":
    main()
