"""
Tests for src/model/: backbone, lora, hyperbolic, heads.
Phase 0: import stubs.
Phase 3: functional tests — HyperbolicProjection norm gate, LoRA injection,
         HierarchicalHeads shapes, adapter save/load roundtrip.

All tests use CPU. Tests that require downloading BioCLIP-2 are marked network.
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Import / API surface (Phase 0)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tiny fixture encoder — mirrors BioCLIP-2 ViT structure without downloading
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    def __init__(self, d_model: int, ratio: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * ratio)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(d_model * ratio, d_model)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class _TinyResBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = _TinyMLP(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return x + self.mlp(attn_out)


class _TinyVisualEncoder(nn.Module):
    """Minimal ViT-like encoder mirroring BioCLIP-2's layer naming."""
    output_dim = 16

    def __init__(self, d_model: int = 16, n_blocks: int = 2):
        super().__init__()
        self.transformer = nn.ModuleList(
            [_TinyResBlock(d_model) for _ in range(n_blocks)]
        )
        self._d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer:
            x = block(x)
        return x.mean(dim=1)  # global average pool


def _tiny_encoder() -> _TinyVisualEncoder:
    return _TinyVisualEncoder(d_model=16, n_blocks=2)


# ---------------------------------------------------------------------------
# HyperbolicProjection — Phase 3 gate (was @skip in Phase 0)
# ---------------------------------------------------------------------------

def test_hyperbolic_projection_norm_lt_1():
    """Phase 3 gate: all output norms must be strictly < norm_clip (0.99)."""
    from src.model.hyperbolic import HyperbolicProjection

    proj = HyperbolicProjection(in_dim=768, out_dim=512)
    x = torch.randn(4, 768)
    out = proj(x)
    norms = out.norm(dim=-1)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"
    assert (norms < 1.0).all(), f"Some norms >= 1.0: {norms}"
    assert (norms < 0.99).all(), f"Some norms >= norm_clip (0.99): {norms}"


def test_hyperbolic_projection_large_inputs_clipped():
    """Very large input values should still produce valid Poincaré points."""
    from src.model.hyperbolic import HyperbolicProjection

    proj = HyperbolicProjection(in_dim=64, out_dim=32)
    x = torch.randn(8, 64) * 1000  # extreme magnitudes
    out = proj(x)
    norms = out.norm(dim=-1)
    assert (norms < 1.0).all()


def test_hyperbolic_projection_batch_independence():
    """Each sample should be projected independently (no cross-sample leakage)."""
    from src.model.hyperbolic import HyperbolicProjection

    proj = HyperbolicProjection(in_dim=32, out_dim=16)
    proj.eval()
    x = torch.randn(4, 32)
    out_batch = proj(x)
    out_single = torch.cat([proj(x[i : i + 1]) for i in range(4)])
    assert torch.allclose(out_batch, out_single, atol=1e-5)


def test_hyperbolic_projection_fixed_curvature():
    """Fixed curvature should not appear in trainable parameters."""
    from src.model.hyperbolic import HyperbolicProjection

    proj = HyperbolicProjection(in_dim=32, out_dim=16, learn_curvature=False)
    param_names = [n for n, _ in proj.named_parameters()]
    assert not any("curv" in n for n in param_names), (
        f"Curvature should not be a parameter when learn_curvature=False. Found: {param_names}"
    )


def test_hyperbolic_projection_learned_curvature():
    """Learned curvature should appear in trainable parameters."""
    from src.model.hyperbolic import HyperbolicProjection

    proj = HyperbolicProjection(in_dim=32, out_dim=16, learn_curvature=True)
    param_names = [n for n, _ in proj.named_parameters()]
    assert any("curv" in n for n in param_names), (
        f"Curvature should be a parameter when learn_curvature=True. Found: {param_names}"
    )


def test_hyperbolic_projection_custom_norm_clip():
    """norm_clip should be enforced at forward pass."""
    from src.model.hyperbolic import HyperbolicProjection

    clip = 0.7
    proj = HyperbolicProjection(in_dim=32, out_dim=16, norm_clip=clip)
    x = torch.randn(16, 32)
    out = proj(x)
    norms = out.norm(dim=-1)
    assert (norms < clip).all(), f"Some norms >= {clip}: {norms}"


# ---------------------------------------------------------------------------
# HierarchicalHeads
# ---------------------------------------------------------------------------

def test_hierarchical_heads_output_shapes():
    from src.model.heads import HierarchicalHeads

    B, D = 8, 64
    heads = HierarchicalHeads(in_dim=D, n_families=10, n_genera=50, n_species=200)
    x = torch.randn(B, D)
    fam, gen, spe = heads(x)
    assert fam.shape == (B, 10), fam.shape
    assert gen.shape == (B, 50), gen.shape
    assert spe.shape == (B, 200), spe.shape


def test_hierarchical_heads_returns_logits_not_probs():
    """Heads should return raw logits, not softmax probabilities."""
    from src.model.heads import HierarchicalHeads

    heads_mod = HierarchicalHeads(in_dim=32, n_families=5, n_genera=20, n_species=100)
    x = torch.randn(4, 32)
    fam, gen, spe = heads_mod(x)
    # Softmax probabilities always sum to 1; raw logits typically do not
    assert not torch.allclose(fam.softmax(dim=-1).sum(dim=-1), fam.sum(dim=-1)), \
        "family logits appear to already be softmax probabilities"


def test_hierarchical_heads_differentiable():
    """All heads must produce gradients for training."""
    from src.model.heads import HierarchicalHeads

    heads_mod = HierarchicalHeads(in_dim=32, n_families=5, n_genera=20, n_species=100)
    x = torch.randn(2, 32, requires_grad=True)
    fam, gen, spe = heads_mod(x)
    loss = fam.sum() + gen.sum() + spe.sum()
    loss.backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# LoRA injection (no network — uses tiny fixture encoder)
# ---------------------------------------------------------------------------

def test_inject_lora_produces_peft_model():
    from src.model.lora import inject_lora

    encoder = _tiny_encoder()
    lora_config = {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.0,
        "target_modules": ["out_proj", "c_fc", "c_proj"],
    }
    peft_model = inject_lora(encoder, lora_config)
    # peft wraps the model; it should still be callable
    x = torch.randn(1, 4, 16)  # (batch, seq, d_model)
    out = peft_model(x)
    assert out.shape[0] == 1


def test_inject_lora_only_lora_params_trainable():
    from src.model.lora import inject_lora, count_trainable_params

    encoder = _tiny_encoder()
    # Freeze base first (as in real pipeline)
    for p in encoder.parameters():
        p.requires_grad_(False)

    lora_config = {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.0,
        "target_modules": ["out_proj", "c_fc", "c_proj"],
    }
    peft_model = inject_lora(encoder, lora_config)
    trainable, total = count_trainable_params(peft_model)
    assert trainable > 0, "LoRA params should be trainable"
    assert trainable < total, "Not all params should be trainable"
    # All trainable params should be LoRA-related
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"Non-LoRA param is trainable: {name}"


def test_count_trainable_params():
    from src.model.lora import count_trainable_params

    model = nn.Linear(10, 5)
    trainable, total = count_trainable_params(model)
    assert trainable == total  # all params trainable by default

    for p in model.parameters():
        p.requires_grad_(False)
    trainable, total = count_trainable_params(model)
    assert trainable == 0


def test_freeze_backbone_zeros_grad():
    from src.model.backbone import freeze_backbone

    encoder = _tiny_encoder()
    freeze_backbone(encoder)
    for p in encoder.parameters():
        assert not p.requires_grad, "All params should be frozen"


def test_save_load_adapter_roundtrip(tmp_path):
    """LoRA weights saved and reloaded should produce identical outputs."""
    from src.model.lora import inject_lora, save_adapter, load_adapter

    # Use fixed seed so both encoders have identical base weights
    torch.manual_seed(0)
    encoder = _tiny_encoder()
    for p in encoder.parameters():
        p.requires_grad_(False)

    lora_config = {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.0,
        "target_modules": ["out_proj", "c_fc", "c_proj"],
    }
    peft_model = inject_lora(encoder, lora_config)

    # Save adapter
    adapter_dir = str(tmp_path / "adapter")
    save_adapter(peft_model, adapter_dir)

    # Load adapter onto a fresh base encoder with the same base weights
    torch.manual_seed(0)
    fresh_encoder = _tiny_encoder()
    for p in fresh_encoder.parameters():
        p.requires_grad_(False)
    loaded_model = load_adapter(fresh_encoder, adapter_dir)

    # Outputs should be identical
    x = torch.randn(1, 4, 16)
    peft_model.eval()
    loaded_model.eval()
    with torch.no_grad():
        out_orig = peft_model(x)
        out_loaded = loaded_model(x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-5), (
        f"Max diff: {(out_orig - out_loaded).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Network — requires downloading BioCLIP-2 from HuggingFace
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_load_backbone_bioclip2():
    from src.model.backbone import load_backbone

    config = {
        "model_id": "hf-hub:imageomics/bioclip-2",
        "embed_dim": 768,
    }
    image_encoder, clip_model, preprocess = load_backbone(
        config, device="cpu", dtype=torch.float32
    )
    assert image_encoder.output_dim == 768
    # All image encoder params frozen by default after load_backbone
    trainable = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    assert trainable == 0, "Backbone should be frozen after load_backbone"


@pytest.mark.network
def test_bioclip2_forward_pass_norm_constraint():
    """End-to-end: image → BioCLIP-2 → HyperbolicProjection, norms < 1."""
    import numpy as np
    from PIL import Image as PILImage
    from src.model.backbone import load_backbone
    from src.model.hyperbolic import HyperbolicProjection

    config = {"model_id": "hf-hub:imageomics/bioclip-2", "embed_dim": 768}
    image_encoder, _, preprocess = load_backbone(
        config, device="cpu", dtype=torch.float32
    )
    proj = HyperbolicProjection(in_dim=768, out_dim=512)

    fake_img = PILImage.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    x = preprocess(fake_img).unsqueeze(0)

    with torch.no_grad():
        feats = image_encoder(x)
        poincare = proj(feats)

    assert feats.shape == (1, 768)
    assert poincare.shape == (1, 512)
    assert (poincare.norm(dim=-1) < 1.0).all()
