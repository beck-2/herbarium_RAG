"""
Phase 0 test stubs for src/eval/.
"""

import pytest


def test_metrics_module_importable():
    from src.eval import metrics
    assert hasattr(metrics, "precision_at_k")
    assert hasattr(metrics, "hierarchical_accuracy")
    assert hasattr(metrics, "mistake_severity")
    assert hasattr(metrics, "expected_calibration_error")
    assert hasattr(metrics, "open_set_recall")


def test_stratified_module_importable():
    from src.eval import stratified
    assert hasattr(stratified, "stratified_evaluate")
    assert hasattr(stratified, "convergent_pair_confusion_rate")
    assert hasattr(stratified, "CONVERGENT_PAIRS")


def test_convergent_pairs_nonempty():
    """CONVERGENT_PAIRS must be defined and non-empty for evaluation."""
    from src.eval.stratified import CONVERGENT_PAIRS
    assert len(CONVERGENT_PAIRS) > 0
    for pair in CONVERGENT_PAIRS:
        assert len(pair) == 2
        assert all(isinstance(name, str) for name in pair)
