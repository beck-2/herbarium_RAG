"""
Phase 0 test stubs for src/viz/.
"""

import pytest


def test_poincare_module_importable():
    from src.viz import poincare
    assert hasattr(poincare, "project_to_2d")
    assert hasattr(poincare, "compute_uncertainty")
    assert hasattr(poincare, "export_disk_json")


def test_confidence_module_importable():
    from src.viz import confidence
    assert hasattr(confidence, "export_confidence_json")
    assert hasattr(confidence, "export_specimens_panel_json")
