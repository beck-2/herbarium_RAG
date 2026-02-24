"""
Phase 0 test stubs for src/data/.
Each test verifies the module is importable and the public API exists.
Implementation tests will be added in Phase 1.
"""

import pytest


def test_download_module_importable():
    from src.data import download
    assert hasattr(download, "download_naflora_metadata")
    assert hasattr(download, "download_symbiota_dwca")
    assert hasattr(download, "download_naflora_images")


def test_parse_module_importable():
    from src.data import parse
    assert hasattr(parse, "parse_dwca")
    assert hasattr(parse, "parse_naflora_csv")
    assert hasattr(parse, "CANONICAL_COLUMNS")


def test_filter_module_importable():
    from src.data import filter
    assert hasattr(filter, "filter_by_region")
    assert hasattr(filter, "filter_quality")
    assert hasattr(filter, "deduplicate")


def test_balance_module_importable():
    from src.data import balance
    assert hasattr(balance, "cap_per_taxon")
    assert hasattr(balance, "stratified_split")
    assert hasattr(balance, "assign_rarity_tier")
