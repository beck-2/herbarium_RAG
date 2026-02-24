#!/usr/bin/env python3
"""
Load NAFlora-1M metadata (train_metadata.json or directory) and print a short summary.
Run this after placing real metadata to confirm the data was loaded properly.

Usage:
  python scripts/verify_naflora_load.py [path_to_json_or_dir]
  Default path: data/raw/naflora1m/ (or . if that doesn't exist)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("data/raw/naflora1m")
        if not path.exists():
            path = Path(".")
    if path.is_dir():
        json_path = path / "train_metadata.json"
        if not json_path.exists():
            print(f"No train_metadata.json in {path}. Place Kaggle train_metadata.json there and re-run.")
            sys.exit(1)
        path = json_path
    elif not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    from src.data import parse
    print(f"Loading: {path}")
    df = parse.parse_naflora_json(str(path))
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nScientific name value counts (top 10):")
    print(df["scientific_name"].value_counts().head(10).to_string())
    print("\nSample (first 3 rows, key cols):")
    print(df[["occurrence_id", "scientific_name", "image_url", "source"]].head(3).to_string())
    print("\nIf this looks correct, the data was loaded properly.")


if __name__ == "__main__":
    main()
