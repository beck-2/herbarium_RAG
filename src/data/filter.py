"""
Geographic and quality filtering of specimen records.

Responsibilities:
- Filter records to a region using bounding-box coordinates (DECISION-4: state-line BBOXes)
- Remove records missing required fields or with implausible coordinates
- Deduplicate by occurrence_id within a source (keep most recent by event_date)
- Region bounding boxes loaded from config/regions.yaml
"""

from __future__ import annotations

import pandas as pd


def filter_by_region(df: pd.DataFrame, region: str, regions_config: dict) -> pd.DataFrame:
    """Filter records to a geographic bounding box.

    Rows with null latitude or longitude are dropped (not considered inside any bbox).
    Use regions_config['regions'][region]['bbox'] with lat_min, lat_max, lon_min, lon_max.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.
        region: Region key matching a key in regions_config['regions'].
        regions_config: Loaded regions.yaml as dict (e.g. from config.load_regions()).

    Returns:
        Filtered DataFrame.
    """
    regions = regions_config.get("regions", {})
    if region not in regions:
        raise KeyError(f"Unknown region {region!r}; known: {list(regions.keys())}")
    bbox = regions[region].get("bbox", {})
    lat_min = bbox.get("lat_min", -90)
    lat_max = bbox.get("lat_max", 90)
    lon_min = bbox.get("lon_min", -180)
    lon_max = bbox.get("lon_max", 180)

    out = df.dropna(subset=["latitude", "longitude"])
    out = out[
        (out["latitude"] >= lat_min)
        & (out["latitude"] <= lat_max)
        & (out["longitude"] >= lon_min)
        & (out["longitude"] <= lon_max)
    ]
    return out.copy()


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records missing required fields or with implausible coordinates.

    Drops rows where:
    - scientific_name, image_url is null/empty (latitude/longitude may be null for NAFlora)
    - latitude or longitude is 0.0 when present (common default-value artifact)
    - latitude not in [-90, 90] or longitude not in [-180, 180] when present

    Returns:
        Filtered DataFrame.
    """
    out = df.copy()
    # Required for all records
    out = out[out["scientific_name"].notna() & (out["scientific_name"].astype(str).str.strip() != "")]
    out = out[out["image_url"].notna() & (out["image_url"].astype(str).str.strip() != "")]
    # Plausible coordinates when present (allow null for NAFlora)
    if "latitude" in out.columns and "longitude" in out.columns:
        has_coords = out["latitude"].notna() & out["longitude"].notna()
        bad_coords = (
            (out["latitude"] == 0.0) & (out["longitude"] == 0.0)
            | (out["latitude"] < -90)
            | (out["latitude"] > 90)
            | (out["longitude"] < -180)
            | (out["longitude"] > 180)
        )
        out = out[~(has_coords & bad_coords)]
    return out.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate occurrence_ids, keeping one row per occurrence_id.

    When 'event_date' is present, keeps the most recently collected record; otherwise
    keeps the first occurrence.
    """
    if "occurrence_id" not in df.columns:
        return df
    if "event_date" in df.columns and df["event_date"].notna().any():
        df = df.sort_values("event_date", ascending=False, na_position="last")
    return df.drop_duplicates(subset=["occurrence_id"], keep="first").reset_index(drop=True)
