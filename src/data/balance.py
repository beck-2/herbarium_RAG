"""
Long-tail capping and stratified train/val/test splitting.

Responsibilities:
- Cap specimens per taxon at max_images_per_taxon (default 150), selecting most recent
- Assign rarity tier per taxon for stratification (abundant >50, moderate 10–50, rare 5–10)
- Produce stratified splits (70/15/15) by rarity_tier and optionally family/genus
- Reserve open_set_genus_fraction (default 10%) of genera for test-only open-set eval
- Write split manifests: train.txt, val.txt, test.txt (one occurrence_id per line)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def add_genus_from_scientific_name(
    df: pd.DataFrame,
    scientific_name_col: str = "scientific_name",
    genus_col: str = "genus",
) -> pd.DataFrame:
    """Add or fill genus_col from first word of scientific_name when missing."""
    out = df.copy()
    if genus_col not in out.columns:
        out[genus_col] = pd.NA
    mask = out[genus_col].isna() & out[scientific_name_col].notna()
    out.loc[mask, genus_col] = (
        out.loc[mask, scientific_name_col]
        .astype(str)
        .str.strip()
        .str.split(n=1)
        .str[0]
    )
    return out


def assign_rarity_tier(df: pd.DataFrame, taxon_col: str = "scientific_name") -> pd.DataFrame:
    """Add a 'rarity_tier' column: 'abundant', 'moderate', or 'rare'.

    Based on count of records per taxon. Abundant: >50; moderate: 10–50; rare: 5–9.
    Taxa with <5 records get rarity_tier 'excluded' (for downstream min_images filtering).
    """
    out = df.copy()
    counts = out.groupby(taxon_col, dropna=False).size().reindex(out[taxon_col])
    out["_count"] = counts.values
    out["rarity_tier"] = "excluded"
    out.loc[out["_count"] > 50, "rarity_tier"] = "abundant"
    out.loc[(out["_count"] >= 10) & (out["_count"] <= 50), "rarity_tier"] = "moderate"
    out.loc[(out["_count"] >= 5) & (out["_count"] < 10), "rarity_tier"] = "rare"
    out = out.drop(columns=["_count"])
    return out


def cap_per_taxon(
    df: pd.DataFrame,
    max_images: int = 150,
    taxon_col: str = "scientific_name",
    date_col: str = "event_date",
) -> pd.DataFrame:
    """Cap specimens per taxon to max_images, preferring most recently collected.

    When date_col is present and has values, sorts by date descending (nulls last)
    and keeps first max_images per taxon. Otherwise keeps first max_images by order.
    """
    if date_col in df.columns and df[date_col].notna().any():
        df = df.sort_values(date_col, ascending=False, na_position="last")
    else:
        df = df.copy()
    capped = df.groupby(taxon_col, dropna=False).head(max_images).reset_index(drop=True)
    return capped


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float | None = None,
    open_set_genus_fraction: float = 0.10,
    seed: int = 42,
    rarity_col: str = "rarity_tier",
    genus_col: str | None = "genus",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce stratified train/val/test splits.

    Stratifies by rarity_tier. If genus_col is present and open_set_genus_fraction > 0,
    holds out that fraction of genera for test-only (open-set). Remaining data is split
    by train_ratio / val_ratio / test_ratio (test_ratio defaults to 1 - train - val).

    Returns:
        (train_df, val_df, test_df).
    """
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    rng = np.random.default_rng(seed)

    # Open-set genus holdout
    test_open = pd.DataFrame()
    if open_set_genus_fraction > 0 and genus_col and genus_col in df.columns:
        genera = df[genus_col].dropna().unique()
        n_holdout = max(1, int(len(genera) * open_set_genus_fraction))
        holdout_genera = rng.choice(genera, size=n_holdout, replace=False)
        test_open = df[df[genus_col].isin(holdout_genera)]
        df = df[~df[genus_col].isin(holdout_genera)]

    if rarity_col not in df.columns:
        # No stratification: random split
        idx = rng.permutation(len(df))
        n = len(idx)
        t1 = int(n * train_ratio)
        t2 = int(n * (train_ratio + val_ratio))
        train_df = df.iloc[idx[:t1]]
        val_df = df.iloc[idx[t1:t2]]
        test_in = df.iloc[idx[t2:]]
    else:
        train_parts, val_parts, test_parts = [], [], []
        for tier, grp in df.groupby(rarity_col, dropna=False):
            n = len(grp)
            perm = rng.permutation(n)
            t1 = int(n * train_ratio)
            t2 = int(n * (train_ratio + val_ratio))
            train_parts.append(grp.iloc[perm[:t1]])
            val_parts.append(grp.iloc[perm[t1:t2]])
            test_parts.append(grp.iloc[perm[t2:]])
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
        test_in = pd.concat(test_parts, ignore_index=True)

    test_df = pd.concat([test_in, test_open], ignore_index=True) if len(test_open) else test_in
    return train_df, val_df, test_df


def write_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    id_col: str = "occurrence_id",
) -> None:
    """Write train.txt, val.txt, test.txt with one occurrence_id per line."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if id_col not in frame.columns:
            raise ValueError(f"DataFrame for {name} has no column {id_col!r}")
        ids = frame[id_col].astype(str).tolist()
        (path / f"{name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
