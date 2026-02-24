"""
DwCA (Darwin Core Archive) and NAFlora-1M metadata parsing.

Responsibilities:
- Parse DwCA archives using dwca-reader into pandas DataFrames
- Parse NAFlora-1M JSON (Kaggle train_metadata.json / test_metadata.json) into same schema
- Normalize field names to canonical schema
- Save/load Parquet via pyarrow

Canonical schema: occurrence_id, scientific_name, latitude, longitude,
  state_province, image_url, reproductive_condition, source, region, event_date.
  event_date is optional (used for capping by recency); NAFlora JSON has no date.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Optional: dwca-reader for DwCA parsing (extra: data)
try:
    from dwca.read import DwCAReader
except ImportError:
    DwCAReader = None  # type: ignore[misc, assignment]

CANONICAL_COLUMNS = [
    "occurrence_id",
    "scientific_name",
    "latitude",
    "longitude",
    "state_province",
    "image_url",
    "reproductive_condition",
    "source",
    "region",
    "event_date",
]


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all CANONICAL_COLUMNS exist; fill missing with null; return in order."""
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[CANONICAL_COLUMNS].copy()


def parse_naflora_json(json_path: str, source: str = "naflora1m") -> pd.DataFrame:
    """Parse NAFlora-1M metadata JSON (Kaggle train_metadata.json / test_metadata.json).

    COCO-style structure: images[], annotations[], categories[]. Joins to produce
    one row per image with scientificName from category. Lat/lon/state_province
    are null (not in the published JSON). event_date is null.

    Args:
        json_path: Path to a single .json file or directory containing
            train_metadata.json and optionally test_metadata.json.
        source: Source label (default 'naflora1m').

    Returns:
        DataFrame with CANONICAL_COLUMNS.
    """
    path = Path(json_path)
    if path.is_dir():
        files = []
        for name in ("train_metadata.json", "test_metadata.json"):
            p = path / name
            if p.exists():
                files.append(p)
        if not files:
            raise FileNotFoundError(f"No train_metadata.json or test_metadata.json in {path}")
    else:
        files = [path]

    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        images = {im["image_id"]: im for im in data.get("images", [])}
        annotations = {a["image_id"]: a for a in data.get("annotations", [])}
        categories = {c["category_id"]: c for c in data.get("categories", [])}
        for image_id, im in images.items():
            ann = annotations.get(image_id, {})
            cat_id = ann.get("category_id")
            cat = categories.get(cat_id, {}) if cat_id is not None else {}
            scientific_name = cat.get("scientificName", "")
            file_name = im.get("file_name", "")
            rows.append({
                "occurrence_id": str(image_id),
                "scientific_name": scientific_name,
                "latitude": pd.NA,
                "longitude": pd.NA,
                "state_province": pd.NA,
                "image_url": file_name,
                "reproductive_condition": pd.NA,
                "source": source,
                "region": pd.NA,
                "event_date": pd.NaT,
            })
    df = pd.DataFrame(rows)
    return _ensure_canonical_columns(df)


def parse_naflora_csv(csv_path: str) -> pd.DataFrame:
    """Parse NAFlora-1M metadata; accepts .json (delegates to parse_naflora_json) or CSV.

    Args:
        csv_path: Path to metadata file (.json or .csv). NAFlora-1M is distributed
            as JSON from Kaggle; CSV is not provided by the dataset.

    Returns:
        DataFrame with CANONICAL_COLUMNS.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)
    if path.suffix.lower() == ".json" or path.name in (
        "train_metadata.json",
        "test_metadata.json",
    ):
        return parse_naflora_json(str(path))
    raise NotImplementedError(
        "NAFlora-1M metadata is JSON from Kaggle (herbarium-2022-fgvc9). "
        "Use parse_naflora_json() or pass a .json path to parse_naflora_csv()."
    )


def parse_dwca(dwca_dir: str, source_name: str) -> pd.DataFrame:
    """Parse a DwCA directory or archive into the canonical DataFrame schema.

    Args:
        dwca_dir: Path to DwCA (zip or extracted directory with meta.xml + core).
        source_name: Source identifier (e.g. 'cch2', 'sernec').

    Returns:
        DataFrame with CANONICAL_COLUMNS.
    """
    if DwCAReader is None:
        raise ImportError(
            "DwCA parsing requires the data extra: pip install 'hyperbolic-herbarium[data]'"
        )
    dwca_path = os.path.abspath(dwca_dir)
    with DwCAReader(dwca_path) as dwca:
        core_location = dwca.core_file_location
        core_df = dwca.pd_read(core_location, parse_dates=True)

    # Map Darwin Core columns (short names) to canonical; prefer standard terms
    canonical_to_dc = [
        ("occurrence_id", ["occurrenceID", "id"]),
        ("scientific_name", ["scientificName"]),
        ("latitude", ["decimalLatitude"]),
        ("longitude", ["decimalLongitude"]),
        ("state_province", ["stateProvince"]),
        ("image_url", ["imageURL", "associatedMedia"]),
        ("reproductive_condition", ["reproductiveCondition"]),
        ("event_date", ["eventDate"]),
    ]
    rename = {}
    for canon, candidates in canonical_to_dc:
        for dc in candidates:
            if dc in core_df.columns:
                rename[dc] = canon
                break
    core_df = core_df.rename(columns=rename)
    core_df["source"] = source_name
    core_df["region"] = pd.NA
    return _ensure_canonical_columns(core_df)


def save_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Write DataFrame to Parquet using pyarrow with snappy compression."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def load_parquet(parquet_path: str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    return table.to_pandas()
