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


def parse_naflora_tsv(tsv_path: str, source: str = "naflora1m") -> pd.DataFrame:
    """Parse NAFlora-mini TSV from GitHub (h22_miniv1_train.tsv / h22_miniv1_val.tsv).

    One row per image. Header has 9 names but rows have 10 fields: the 5th is file_id
    (e.g. 00026__001). Lat/lon and event_date are null. Use when you don't have Kaggle JSON.

    Args:
        tsv_path: Path to a .tsv file.
        source: Source label (default 'naflora1m').

    Returns:
        DataFrame with CANONICAL_COLUMNS.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    # GitHub header has 9 names but 10 fields: 5th is file_id (00026__001), 6th is scientificName
    file_id = df.get("file_id", df.iloc[:, 4] if len(df.columns) > 4 else df["image_id"])
    scientific_name = df["scientificName"] if "scientificName" in df.columns else (df.iloc[:, 5] if len(df.columns) > 5 else pd.Series(dtype=object))
    out = pd.DataFrame({
        "occurrence_id": file_id.astype(str),
        "scientific_name": scientific_name.astype(str) if hasattr(scientific_name, "astype") else scientific_name,
        "latitude": pd.NA,
        "longitude": pd.NA,
        "state_province": pd.NA,
        "image_url": file_id.astype(str),
        "reproductive_condition": pd.NA,
        "source": source,
        "region": pd.NA,
        "event_date": pd.NaT,
    })
    return _ensure_canonical_columns(out)


def parse_naflora_csv(csv_path: str) -> pd.DataFrame:
    """Parse NAFlora-1M metadata; accepts .json (Kaggle), .tsv (GitHub NAFlora-mini), or path.

    Args:
        csv_path: Path to metadata file. .json → Kaggle COCO-style; .tsv → GitHub NAFlora-mini.

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
    if path.suffix.lower() == ".tsv":
        return parse_naflora_tsv(str(path))
    raise NotImplementedError(
        "NAFlora metadata: use .json (Kaggle train_metadata.json) or .tsv (GitHub NAFlora-mini). "
        "A JSON that only lists 'number of images per species' is not sufficient; we need per-image rows."
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
    """Load a Parquet file into a DataFrame.

    Normalizes None → pd.NA in object-dtype columns to match the representation
    used when building DataFrames (pyarrow converts pd.NA to null on write;
    reading back yields Python None, which we normalize back to pd.NA).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].where(df[col].notna(), other=pd.NA)
    return df
