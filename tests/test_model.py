"""
Phase 0 test stubs for src/model/.
Phase 3 gate: HyperbolicProjection forward pass must produce points with norm < 1.

Note: model stubs depend on torch. Tests are skipped if torch is not installed.
Install with: pip install 'hyperbolic-herbarium[ml]'
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; run: pip install 'hyperbolic-herbarium[ml]'")


def test_backbone_module_importable():
    from src.model import backbone
    assert hasattr(backbone, "load_backbone")
    assert hasattr(backbone, "freeze_backbone")


def test_lora_module_importable():
    from src.model import lora
    assert hasattr(lora, "inject_lora")
    assert hasattr(lora, "save_adapter")
    assert hasattr(lora, "load_adapter")
    assert hasattr(lora, "count_trainable_params")


def test_hyperbolic_module_importable():
    from src.model import hyperbolic
    assert hasattr(hyperbolic, "HyperbolicProjection")


def test_heads_module_importable():
    from src.model import heads
    assert hasattr(heads, "HierarchicalHeads")


# Phase 3 gate (not yet implemented — will be filled in Phase 3)
@pytest.mark.skip(reason="Implemented in Phase 3")
def test_hyperbolic_projection_norm_lt_1():
    """HyperbolicProjection forward pass must produce points with norm < 1."""
    import torch
    from src.model.hyperbolic import HyperbolicProjection

    proj = HyperbolicProjection(in_dim=768, out_dim=512)
    x = torch.randn(4, 768)
    out = proj(x)
    norms = out.norm(dim=-1)
    assert (norms < 1.0).all(), f"Some norms >= 1.0: {norms}"
    assert (norms < 0.99).all(), f"Some norms >= norm_clip (0.99): {norms}"
