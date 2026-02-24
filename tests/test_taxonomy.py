"""
Phase 0 test stubs for src/taxonomy/.

Note: gnn.py depends on torch. test_gnn_module_importable is skipped without torch.
"""

import pytest


def test_tnrs_module_importable():
    from src.taxonomy import tnrs
    assert hasattr(tnrs, "TNRSResolver")
    assert hasattr(tnrs, "TNRS_BATCH_SIZE")
    assert hasattr(tnrs, "TNRS_CONTEXT")


def test_opentree_module_importable():
    from src.taxonomy import opentree
    assert hasattr(opentree, "fetch_induced_subtree")
    assert hasattr(opentree, "compute_patristic_distances")
    assert hasattr(opentree, "get_lca_rank")
    assert hasattr(opentree, "export_subtree_json")


def test_gnn_module_importable():
    pytest.importorskip("torch", reason="torch not installed; run: pip install 'hyperbolic-herbarium[ml]'")
    from src.taxonomy import gnn
    assert hasattr(gnn, "TaxonomyGNNRegularizer")
