"""
Phase 0 test stubs for src/index/.
"""

import pytest


def test_build_module_importable():
    from src.index import build
    assert hasattr(build, "encode_specimens")
    assert hasattr(build, "build_ivfpq_index")
    assert hasattr(build, "build_family_subindexes")
    assert hasattr(build, "verify_recall")
    assert hasattr(build, "save_index")
    assert hasattr(build, "load_index")


def test_bundle_module_importable():
    from src.index import bundle
    assert hasattr(bundle, "pack_bundle")
    assert hasattr(bundle, "generate_thumbnails")
    assert hasattr(bundle, "create_specimens_db")
    assert hasattr(bundle, "check_bundle_size")
