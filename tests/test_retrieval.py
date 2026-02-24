"""
Phase 0 test stubs for src/retrieval/.
"""

import pytest


def test_encode_module_importable():
    from src.retrieval import encode
    assert hasattr(encode, "encode_query")


def test_search_module_importable():
    from src.retrieval import search
    assert hasattr(search, "global_faiss_search")
    assert hasattr(search, "family_index_search")
    assert hasattr(search, "merge_and_dedup")
    assert hasattr(search, "retrieve")
    assert hasattr(search, "FAMILY_CONFIDENCE_THRESHOLD")
    assert hasattr(search, "GENUS_CONFIDENCE_THRESHOLD")


def test_graph_module_importable():
    from src.retrieval import graph
    assert hasattr(graph, "build_retrieval_graph")
    assert hasattr(graph, "aggregate_scores")


def test_confidence_thresholds_in_range():
    """Sanity check: confidence thresholds are between 0 and 1."""
    from src.retrieval.search import FAMILY_CONFIDENCE_THRESHOLD, GENUS_CONFIDENCE_THRESHOLD
    assert 0.0 < FAMILY_CONFIDENCE_THRESHOLD < 1.0
    assert 0.0 < GENUS_CONFIDENCE_THRESHOLD < 1.0
    assert GENUS_CONFIDENCE_THRESHOLD < FAMILY_CONFIDENCE_THRESHOLD
