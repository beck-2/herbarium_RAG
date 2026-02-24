"""
Tests for src/data/: download, parse, filter, balance.
Phase 0: import/API. Phase 1: implementation with fixtures.
"""

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_download_module_importable():
    from src.data import download
    assert hasattr(download, "download_naflora_metadata")
    assert hasattr(download, "download_symbiota_dwca")
    assert hasattr(download, "download_naflora_images")


def test_parse_module_importable():
    from src.data import parse
    assert hasattr(parse, "parse_dwca")
    assert hasattr(parse, "parse_naflora_csv")
    assert hasattr(parse, "parse_naflora_json")
    assert hasattr(parse, "CANONICAL_COLUMNS")


def test_filter_module_importable():
    from src.data import filter as data_filter
    assert hasattr(data_filter, "filter_by_region")
    assert hasattr(data_filter, "filter_quality")
    assert hasattr(data_filter, "deduplicate")


def test_balance_module_importable():
    from src.data import balance
    assert hasattr(balance, "cap_per_taxon")
    assert hasattr(balance, "stratified_split")
    assert hasattr(balance, "assign_rarity_tier")
    assert hasattr(balance, "write_split_manifests")


# --- Phase 1: parse ---


def test_parse_naflora_json_fixture():
    from src.data import parse
    path = str(FIXTURES_DIR / "naflora_mini.json")
    df = parse.parse_naflora_json(path)
    assert len(df) == 3
    assert list(df.columns) == parse.CANONICAL_COLUMNS
    assert df["source"].iloc[0] == "naflora1m"
    assert set(df["scientific_name"]) == {"Quercus alba", "Acer rubrum"}
    assert df["latitude"].isna().all()
    assert df["occurrence_id"].astype(str).tolist() == ["1", "2", "3"]


def test_parse_naflora_csv_accepts_json_path():
    from src.data import parse
    path = str(FIXTURES_DIR / "naflora_mini.json")
    df = parse.parse_naflora_csv(path)
    assert len(df) == 3
    assert "scientific_name" in df.columns


def test_save_and_load_parquet(tmp_path):
    from src.data import parse
    path = str(FIXTURES_DIR / "naflora_mini.json")
    df = parse.parse_naflora_json(path)
    out = tmp_path / "specimens.parquet"
    parse.save_parquet(df, str(out))
    assert out.exists()
    back = parse.load_parquet(str(out))
    pd.testing.assert_frame_equal(df, back)


# --- Phase 1: filter ---


def test_filter_quality_drops_empty_names_and_urls():
    from src.data import filter as data_filter
    df = pd.DataFrame({
        "occurrence_id": ["a", "b", "c"],
        "scientific_name": ["Quercus alba", "", "Acer rubrum"],
        "image_url": ["http://x/1.jpg", "http://x/2.jpg", ""],
        "latitude": [40.0, 41.0, 42.0],
        "longitude": [-120.0, -121.0, -122.0],
    })
    out = data_filter.filter_quality(df)
    assert len(out) == 1
    assert out["occurrence_id"].iloc[0] == "a"


def test_filter_quality_drops_bad_coords():
    from src.data import filter as data_filter
    df = pd.DataFrame({
        "occurrence_id": ["a", "b"],
        "scientific_name": ["X", "Y"],
        "image_url": ["u1", "u2"],
        "latitude": [0.0, 45.0],
        "longitude": [0.0, -120.0],
    })
    out = data_filter.filter_quality(df)
    assert len(out) == 1
    assert out["occurrence_id"].iloc[0] == "b"


def test_filter_by_region_bbox():
    from src.data import filter as data_filter
    regions_config = {
        "regions": {
            "california": {
                "bbox": {"lat_min": 32.5, "lat_max": 42.0, "lon_min": -124.5, "lon_max": -114.1},
            },
        },
    }
    df = pd.DataFrame({
        "occurrence_id": ["a", "b", "c"],
        "latitude": [36.0, 50.0, 35.0],
        "longitude": [-120.0, -120.0, -100.0],
    })
    out = data_filter.filter_by_region(df, "california", regions_config)
    assert len(out) == 2
    assert set(out["occurrence_id"]) == {"a", "c"}


def test_deduplicate_keeps_first():
    from src.data import filter as data_filter
    df = pd.DataFrame({
        "occurrence_id": ["a", "a", "b"],
        "scientific_name": ["X", "Y", "Z"],
    })
    out = data_filter.deduplicate(df)
    assert len(out) == 2
    assert out["occurrence_id"].tolist() == ["a", "b"]


# --- Phase 1: balance ---


def test_assign_rarity_tier():
    from src.data import balance
    df = pd.DataFrame({
        "scientific_name": ["A", "A", "A", "B", "B", "C"] * 10,  # 30 A, 20 B, 10 C
    })
    out = balance.assign_rarity_tier(df)
    assert "rarity_tier" in out.columns
    assert set(out["rarity_tier"]).issubset({"abundant", "moderate", "rare", "excluded"})
    # 30 and 20 and 10 are all in [10, 50] -> moderate
    assert (out["rarity_tier"] == "moderate").all()


def test_cap_per_taxon():
    from src.data import balance
    df = pd.DataFrame({
        "scientific_name": ["A"] * 5 + ["B"] * 2,
        "occurrence_id": [str(i) for i in range(7)],
    })
    out = balance.cap_per_taxon(df, max_images=2)
    assert len(out) == 4
    assert out.groupby("scientific_name").size()["A"] == 2
    assert out.groupby("scientific_name").size()["B"] == 2


def test_stratified_split_returns_three():
    from src.data import balance
    df = pd.DataFrame({
        "scientific_name": ["A", "B", "C"] * 20,
        "occurrence_id": [str(i) for i in range(60)],
    })
    df = balance.assign_rarity_tier(df)
    train, val, test = balance.stratified_split(df, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert len(train) + len(val) + len(test) == 60
    assert len(train) >= 30
    assert len(val) >= 5
    assert len(test) >= 5


def test_write_split_manifests(tmp_path):
    from src.data import balance
    train_df = pd.DataFrame({"occurrence_id": ["1", "2"]})
    val_df = pd.DataFrame({"occurrence_id": ["3"]})
    test_df = pd.DataFrame({"occurrence_id": ["4"]})
    balance.write_split_manifests(train_df, val_df, test_df, str(tmp_path))
    assert (tmp_path / "train.txt").read_text().strip().split() == ["1", "2"]
    assert (tmp_path / "val.txt").read_text().strip() == "3"
    assert (tmp_path / "test.txt").read_text().strip() == "4"
