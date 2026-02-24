"""
Phase 0 test stubs for src/train/.

Note: train stubs depend on torch. Tests are skipped if torch is not installed.
Install with: pip install 'hyperbolic-herbarium[ml]'
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; run: pip install 'hyperbolic-herbarium[ml]'")


def test_loss_module_importable():
    from src.train import loss
    assert hasattr(loss, "HierarchicalCrossEntropyLoss")
    assert hasattr(loss, "HyperbolicMarginLoss")
    assert hasattr(loss, "combined_loss")


def test_train_global_module_importable():
    from src.train import train_global
    assert hasattr(train_global, "parse_args")
    assert hasattr(train_global, "train")


def test_train_regional_lora_module_importable():
    from src.train import train_regional_lora
    assert hasattr(train_regional_lora, "parse_args")
    assert hasattr(train_regional_lora, "train")


def test_combined_loss_zero_taxonomy():
    """combined_loss should work with tax_loss=None (Phase 2 training)."""
    import torch
    from src.train.loss import combined_loss

    hier = torch.tensor(1.0)
    hyp = torch.tensor(0.5)
    result = combined_loss(hier, hyp, tax_loss=None, alpha=0.1, beta=0.05)
    expected = 1.0 + 0.1 * 0.5
    assert abs(result.item() - expected) < 1e-5, f"Expected {expected}, got {result.item()}"
