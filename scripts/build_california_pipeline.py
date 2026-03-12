#!/usr/bin/env python3.14
"""
Build California regional bundle from CCH2 DwCA collections.

Steps:
    1. Parse all CCH2 DwCA collections in data/raw/symbiota/cch2/
    2. Filter to California bounding box
    3. Quality-filter (drop missing scientific_name)
    4. Deduplicate by occurrence_id
    5. Cap at max_images per taxon (default 150)
    6. Save specimens.parquet
    7. Build FAISS global index + family sub-indexes
       - With --real-model: encode images using trained checkpoint (requires images)
       - Without: build index from stored embeddings or skip (metadata-only bundle)
    8. Pack bundle → data/processed/bundles/california/

Usage:
    python3.14 scripts/build_california_pipeline.py
    python3.14 scripts/build_california_pipeline.py --max-images 50 --dry-run
    python3.14 scripts/build_california_pipeline.py --real-model checkpoints/global/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build California regional bundle from CCH2 data")
    p.add_argument(
        "--cch2-dir",
        default=str(ROOT / "data" / "raw" / "symbiota" / "cch2"),
        help="Directory containing extracted CCH2 DwCA collections",
    )
    p.add_argument(
        "--regions-config",
        default=str(ROOT / "config" / "regions.yaml"),
        help="Path to regions.yaml",
    )
    p.add_argument(
        "--opentree-fixture",
        default=str(ROOT / "tests" / "fixtures" / "opentree_mini_subtree.json"),
        help="Path to OpenTree subtree JSON (use scripts/fetch_opentree_fixture.py for full tree)",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "bundles" / "california"),
        help="Output bundle directory",
    )
    p.add_argument(
        "--parquet-dir",
        default=str(ROOT / "data" / "processed" / "regions" / "california"),
        help="Directory to write specimens.parquet and split manifests",
    )
    p.add_argument("--max-images", type=int, default=150, help="Max images per taxon")
    p.add_argument(
        "--real-model",
        default=None,
        help="Path to checkpoint for real encoding (skipped if not provided)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and filter only — skip index build and bundle pack",
    )
    p.add_argument(
        "--limit-collections",
        type=int,
        default=None,
        help="Process only first N collections (for quick testing)",
    )
    return p.parse_args()


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    import yaml

    from src.data.parse import parse_dwca, save_parquet
    from src.data.filter import filter_by_region, filter_quality, deduplicate
    from src.data.balance import cap_per_taxon, assign_rarity_tier, stratified_split, write_split_manifests

    # ── 1. Parse all CCH2 collections ────────────────────────────────────────

    cch2_dir = Path(args.cch2_dir)
    if not cch2_dir.exists():
        _log(f"ERROR: CCH2 directory not found: {cch2_dir}")
        _log("Run:  bash scripts/download_data.sh")
        sys.exit(1)

    collections = sorted(
        d for d in cch2_dir.iterdir()
        if d.is_dir() and (d / "occurrences.csv").exists()
    )
    if args.limit_collections:
        collections = collections[: args.limit_collections]

    _log(f"Found {len(collections)} CCH2 collections to parse")

    dfs: list[pd.DataFrame] = []
    for i, col_dir in enumerate(collections, 1):
        _log(f"  [{i}/{len(collections)}] Parsing {col_dir.name} ...")
        try:
            df = parse_dwca(str(col_dir), "cch2")
            _log(f"      → {len(df):,} records")
            dfs.append(df)
        except Exception as exc:
            _log(f"      → SKIP ({exc})")

    if not dfs:
        _log("ERROR: No records parsed from any collection.")
        sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)
    _log(f"Total parsed: {len(all_df):,} records across {len(dfs)} collections")

    # ── 2. Filter to California ───────────────────────────────────────────────

    with open(args.regions_config) as f:
        regions_config = yaml.safe_load(f)

    ca_df = filter_by_region(all_df, "california", regions_config)
    _log(f"After California bbox filter: {len(ca_df):,} records")

    # ── 3. Quality filter ─────────────────────────────────────────────────────

    # For CCH2 training data we keep specimens even without image URLs
    # (images will be downloaded separately); quality filter requires scientific_name.
    ca_df = ca_df[
        ca_df["scientific_name"].notna()
        & (ca_df["scientific_name"].astype(str).str.strip() != "")
    ].copy()
    _log(f"After scientific_name filter: {len(ca_df):,} records")

    # ── 4. Deduplicate ────────────────────────────────────────────────────────

    ca_df = deduplicate(ca_df)
    _log(f"After deduplication: {len(ca_df):,} unique occurrence IDs")

    # ── 5. Cap per taxon ──────────────────────────────────────────────────────

    ca_df = cap_per_taxon(ca_df, max_images=args.max_images)
    _log(f"After cap (max {args.max_images}/taxon): {len(ca_df):,} records")

    n_taxa = ca_df["scientific_name"].nunique()
    n_families = ca_df["family"].dropna().nunique()
    _log(f"  → {n_taxa:,} unique taxa, {n_families:,} families")

    # ── 6. Save parquet + splits ──────────────────────────────────────────────

    parquet_dir = Path(args.parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / "specimens.parquet"
    save_parquet(ca_df, str(parquet_path))
    _log(f"Saved: {parquet_path}")

    ca_df = assign_rarity_tier(ca_df)
    train_df, val_df, test_df = stratified_split(ca_df, seed=42)
    write_split_manifests(train_df, val_df, test_df, str(parquet_dir))
    _log(f"Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    if args.dry_run:
        _log("--dry-run: stopping before index build.")
        _print_summary(ca_df, n_taxa, n_families, parquet_path)
        return

    # ── 7. Build FAISS indexes ────────────────────────────────────────────────

    import numpy as np
    import json as _json
    import faiss
    from src.index.build import build_family_subindexes, build_ivfpq_index, save_index, verify_recall

    index_dir = Path(args.output_dir).parent / "california_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "faiss_families").mkdir(exist_ok=True)

    specimen_ids = ca_df["occurrence_id"].tolist()
    family_labels = ca_df["family"].fillna("Unknown").tolist()
    n = len(ca_df)

    if args.real_model:
        _log(f"Loading model from {args.real_model} for real encoding …")
        embeddings, specimen_ids = _encode_with_model(args.real_model, ca_df)
        family_labels = (
            ca_df.set_index("occurrence_id")
            .loc[specimen_ids, "family"]
            .fillna("Unknown")
            .tolist()
        )
        _log(f"Encoded {len(specimen_ids):,} specimens with real model")
    else:
        _log(
            "No --real-model provided. Using random placeholder embeddings (dim=512). "
            "Re-run with --real-model after training to build a real index."
        )
        rng = np.random.default_rng(42)
        dim = 512
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8) * 0.9

    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    dim = emb.shape[1]

    # Global index: IVF-PQ for large n, FlatL2 fallback for small n
    _log("Building global FAISS index …")
    if len(emb) >= 256:
        n_clusters = min(256, len(emb) // 4)
        global_index = build_ivfpq_index(emb, n_clusters=n_clusters)
    else:
        global_index = faiss.IndexFlatL2(dim)
        global_index.add(emb)
    save_index(global_index, str(index_dir / "faiss_global.bin"))
    _log(f"  Global index: {global_index.ntotal:,} vectors")

    recall = verify_recall(global_index, emb[:min(200, len(emb))], specimen_ids[:min(200, len(emb))])
    _log(f"  Global recall@10 (on {min(200, len(emb))} probes): {recall:.3f}")

    # Family sub-indexes
    _log("Building family sub-indexes …")
    family_indexes = build_family_subindexes(emb, family_labels)
    for family, idx in family_indexes.items():
        safe = family.replace("/", "_").replace(" ", "_")
        save_index(idx, str(index_dir / "faiss_families" / f"{safe}.bin"))
    _log(f"  Family sub-indexes: {len(family_indexes)} families")

    # Save specimen IDs (for load_bundle ordering verification)
    (index_dir / "specimen_ids.json").write_text(
        _json.dumps(specimen_ids), encoding="utf-8"
    )

    # ── 8. Pack bundle ────────────────────────────────────────────────────────

    from src.index.bundle import pack_bundle

    ckpt_dir = Path(args.real_model).parent if args.real_model else (ROOT / "checkpoints" / "global")

    _log(f"Packing bundle → {args.output_dir}")
    manifest = pack_bundle(
        region="california",
        checkpoint_dir=str(ckpt_dir),
        index_dir=str(index_dir),
        specimens_parquet=str(parquet_path),
        image_dir=None,
        opentree_subtree_json=args.opentree_fixture,
        output_dir=args.output_dir,
    )
    _log(f"Bundle packed: {manifest['bundle_size_mb']:.1f} MB")
    _log(f"  n_specimens={manifest['n_specimens']:,}  n_families={manifest['n_families']}")

    _print_summary(ca_df, n_taxa, n_families, parquet_path, manifest=manifest)


def _encode_with_model(checkpoint_path: str, df) -> tuple:
    """Encode specimens using the trained hyperbolic projection model."""
    import torch
    import numpy as np
    from src.model.backbone import load_backbone
    from src.model.projection import HyperbolicProjection

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    _log(f"  device={device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone, preprocess = load_backbone(device=device)
    proj = HyperbolicProjection()
    proj.load_state_dict(ckpt.get("hyperbolic_proj", ckpt))
    proj.to(device).eval()
    backbone.eval()

    specimen_ids = df["occurrence_id"].tolist()
    embeddings: list = []
    valid_ids: list = []

    from PIL import Image as PILImage
    import os

    image_root = Path("data/raw/images")
    for sid in specimen_ids:
        for ext in (".jpg", ".jpeg", ".png"):
            p = image_root / f"{sid}{ext}"
            if p.exists():
                img = PILImage.open(p).convert("RGB")
                with torch.no_grad():
                    feat = backbone(preprocess(img).unsqueeze(0).to(device))
                    emb = proj(feat).cpu().numpy()
                embeddings.append(emb[0])
                valid_ids.append(sid)
                break

    return np.stack(embeddings).astype(np.float32), valid_ids


def _print_summary(df, n_taxa: int, n_families: int, parquet_path, manifest: dict | None = None) -> None:
    print()
    print("=" * 60)
    print("California Pipeline Summary")
    print("=" * 60)
    print(f"  Records:     {len(df):,}")
    print(f"  Taxa:        {n_taxa:,}")
    print(f"  Families:    {n_families:,}")
    print(f"  Parquet:     {parquet_path}")
    if manifest:
        print(f"  Bundle size: {manifest['bundle_size_mb']:.1f} MB")
        print(f"  Output:      {manifest.get('output_dir', '(see --output-dir)')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
