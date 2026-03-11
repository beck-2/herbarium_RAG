"""
Tests for src/train/: loss, dataset, train_global.
Phase 0: import stubs.
Phase 4: loss unit tests, smoke train (3 epochs, synthetic data, CPU).
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; run: pip install 'hyperbolic-herbarium[ml]'")
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Import / API surface (Phase 0)
# ---------------------------------------------------------------------------

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
    from src.train.loss import combined_loss
    hier = torch.tensor(1.0)
    hyp = torch.tensor(0.5)
    result = combined_loss(hier, hyp, tax_loss=None, alpha=0.1, beta=0.05)
    expected = 1.0 + 0.1 * 0.5
    assert abs(result.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_synthetic_dataset_importable():
    from src.data.dataset import SyntheticSpecimenDataset
    assert SyntheticSpecimenDataset is not None


def test_synthetic_dataset_len_and_shape():
    from src.data.dataset import SyntheticSpecimenDataset
    ds = SyntheticSpecimenDataset(n_samples=100, embed_dim=64,
                                   n_families=4, n_genera=10, n_species=20)
    assert len(ds) == 100
    feat, fam, gen, spe = ds[0]
    assert feat.shape == (64,)
    assert feat.dtype == torch.float32
    assert 0 <= fam.item() < 4
    assert 0 <= gen.item() < 10
    assert 0 <= spe.item() < 20


def test_synthetic_dataset_dataloader():
    from src.data.dataset import SyntheticSpecimenDataset
    from torch.utils.data import DataLoader
    ds = SyntheticSpecimenDataset(n_samples=64, embed_dim=32,
                                   n_families=3, n_genera=8, n_species=16)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    feats, fam, gen, spe = next(iter(loader))
    assert feats.shape == (16, 32)
    assert fam.shape == (16,)


# ---------------------------------------------------------------------------
# HierarchicalCrossEntropyLoss
# ---------------------------------------------------------------------------

def test_hierarchical_loss_is_scalar():
    from src.train.loss import HierarchicalCrossEntropyLoss
    B, NF, NG, NS = 8, 5, 20, 50
    loss_fn = HierarchicalCrossEntropyLoss()
    fam_logits = torch.randn(B, NF)
    gen_logits = torch.randn(B, NG)
    spe_logits = torch.randn(B, NS)
    fam_labels = torch.randint(0, NF, (B,))
    gen_labels = torch.randint(0, NG, (B,))
    spe_labels = torch.randint(0, NS, (B,))
    loss = loss_fn(fam_logits, gen_logits, spe_logits,
                   fam_labels, gen_labels, spe_labels)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() > 0


def test_hierarchical_loss_gradients_flow():
    from src.train.loss import HierarchicalCrossEntropyLoss
    B, NF, NG, NS = 4, 3, 10, 20
    loss_fn = HierarchicalCrossEntropyLoss()
    fam_logits = torch.randn(B, NF, requires_grad=True)
    gen_logits = torch.randn(B, NG, requires_grad=True)
    spe_logits = torch.randn(B, NS, requires_grad=True)
    loss = loss_fn(fam_logits, gen_logits, spe_logits,
                   torch.randint(0, NF, (B,)),
                   torch.randint(0, NG, (B,)),
                   torch.randint(0, NS, (B,)))
    loss.backward()
    assert fam_logits.grad is not None
    assert gen_logits.grad is not None
    assert spe_logits.grad is not None


def test_hierarchical_loss_weights_matter():
    """Higher species weight should dominate when species loss is large."""
    from src.train.loss import HierarchicalCrossEntropyLoss
    # Make species task very hard (many classes, wrong labels), family easy
    B = 16
    loss_high_species = HierarchicalCrossEntropyLoss(species_weight=1.0, family_weight=0.0, genus_weight=0.0)
    loss_high_family = HierarchicalCrossEntropyLoss(species_weight=0.0, family_weight=1.0, genus_weight=0.0)
    fam_logits = torch.zeros(B, 3); fam_logits[:, 0] = 10  # family easy
    gen_logits = torch.randn(B, 20)
    spe_logits = torch.randn(B, 100)  # species hard
    fam_labels = torch.zeros(B, dtype=torch.long)
    gen_labels = torch.randint(0, 20, (B,))
    spe_labels = torch.randint(0, 100, (B,))
    l_spe = loss_high_species(fam_logits, gen_logits, spe_logits, fam_labels, gen_labels, spe_labels)
    l_fam = loss_high_family(fam_logits, gen_logits, spe_logits, fam_labels, gen_labels, spe_labels)
    assert l_spe > l_fam, "Species-dominated loss should be larger when species task is hard"


# ---------------------------------------------------------------------------
# HyperbolicMarginLoss
# ---------------------------------------------------------------------------

def test_hyperbolic_margin_loss_is_scalar():
    from src.train.loss import HyperbolicMarginLoss
    from src.model.hyperbolic import HyperbolicProjection
    proj = HyperbolicProjection(in_dim=32, out_dim=16)
    loss_fn = HyperbolicMarginLoss(margin=0.5)
    feats = torch.randn(8, 32)
    labels = torch.randint(0, 4, (8,))
    embeddings = proj(feats)
    loss = loss_fn(embeddings, labels)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() >= 0


def test_hyperbolic_margin_loss_zero_when_well_separated():
    """When same-class points cluster tightly and cross-class points are far apart,
    the margin loss should be near zero."""
    from src.train.loss import HyperbolicMarginLoss
    loss_fn = HyperbolicMarginLoss(margin=0.1)
    # Construct embeddings: class 0 near origin, class 1 near boundary
    emb_class0 = torch.zeros(4, 8) + 0.01  # near center
    emb_class1 = torch.zeros(4, 8); emb_class1[:, 0] = 0.95  # near boundary
    # Normalize to be inside ball
    emb_class0 = emb_class0 / emb_class0.norm(dim=-1, keepdim=True) * 0.05
    emb_class1 = emb_class1 / emb_class1.norm(dim=-1, keepdim=True) * 0.90
    embeddings = torch.cat([emb_class0, emb_class1])
    labels = torch.cat([torch.zeros(4, dtype=torch.long),
                        torch.ones(4, dtype=torch.long)])
    loss = loss_fn(embeddings, labels)
    # Loss should be zero (or very small) when already well-separated
    assert loss.item() < 0.5, f"Expected near-zero loss, got {loss.item()}"


def test_hyperbolic_margin_loss_gradients_flow():
    from src.train.loss import HyperbolicMarginLoss
    from src.model.hyperbolic import HyperbolicProjection
    proj = HyperbolicProjection(in_dim=32, out_dim=16)
    loss_fn = HyperbolicMarginLoss(margin=0.5)
    feats = torch.randn(8, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    embeddings = proj(feats)
    loss = loss_fn(embeddings, labels)
    loss.backward()
    assert feats.grad is not None
    assert not feats.grad.isnan().any()


# ---------------------------------------------------------------------------
# Smoke train — 3 epochs, synthetic data, CPU
# ---------------------------------------------------------------------------

def _make_smoke_model(embed_dim=64, n_families=4, n_genera=10, n_species=20):
    """Assemble a tiny model: HyperbolicProjection + HierarchicalHeads."""
    from src.model.hyperbolic import HyperbolicProjection
    from src.model.heads import HierarchicalHeads
    proj = HyperbolicProjection(in_dim=embed_dim, out_dim=32)
    heads = HierarchicalHeads(in_dim=embed_dim, n_families=n_families,
                              n_genera=n_genera, n_species=n_species)
    return proj, heads


def test_one_step_no_nan():
    """One forward + backward pass — no NaN/Inf in gradients."""
    from src.train.loss import HierarchicalCrossEntropyLoss, HyperbolicMarginLoss, combined_loss
    from src.data.dataset import SyntheticSpecimenDataset
    from torch.utils.data import DataLoader

    NF, NG, NS, D = 4, 10, 20, 64
    proj, heads = _make_smoke_model(D, NF, NG, NS)
    params = list(proj.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    hier_loss_fn = HierarchicalCrossEntropyLoss()
    hyp_loss_fn = HyperbolicMarginLoss()

    ds = SyntheticSpecimenDataset(n_samples=32, embed_dim=D,
                                   n_families=NF, n_genera=NG, n_species=NS)
    loader = DataLoader(ds, batch_size=16)
    feats, fam_lbl, gen_lbl, spe_lbl = next(iter(loader))

    optimizer.zero_grad()
    poincare = proj(feats)
    fam_logits, gen_logits, spe_logits = heads(feats)
    l_hier = hier_loss_fn(fam_logits, gen_logits, spe_logits, fam_lbl, gen_lbl, spe_lbl)
    l_hyp = hyp_loss_fn(poincare, spe_lbl)
    loss = combined_loss(l_hier, l_hyp, tax_loss=None)
    loss.backward()

    for name, p in list(proj.named_parameters()) + list(heads.named_parameters()):
        assert p.grad is not None, f"No grad for {name}"
        assert not p.grad.isnan().any(), f"NaN grad in {name}"
        assert not p.grad.isinf().any(), f"Inf grad in {name}"

    optimizer.step()


def test_smoke_train_loss_decreases():
    """3 epochs on 200 synthetic samples — epoch 3 avg loss < epoch 1 avg loss."""
    from src.train.loss import HierarchicalCrossEntropyLoss, HyperbolicMarginLoss, combined_loss
    from src.data.dataset import SyntheticSpecimenDataset
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    NF, NG, NS, D = 4, 10, 20, 64
    proj, heads = _make_smoke_model(D, NF, NG, NS)
    params = list(proj.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    hier_loss_fn = HierarchicalCrossEntropyLoss()
    hyp_loss_fn = HyperbolicMarginLoss()

    ds = SyntheticSpecimenDataset(n_samples=200, embed_dim=D,
                                   n_families=NF, n_genera=NG, n_species=NS,
                                   seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    epoch_losses = []
    for epoch in range(3):
        proj.train(); heads.train()
        total = 0.0
        steps = 0
        for feats, fam_lbl, gen_lbl, spe_lbl in loader:
            optimizer.zero_grad()
            poincare = proj(feats)
            fam_l, gen_l, spe_l = heads(feats)
            l = combined_loss(
                hier_loss_fn(fam_l, gen_l, spe_l, fam_lbl, gen_lbl, spe_lbl),
                hyp_loss_fn(poincare, spe_lbl),
                tax_loss=None,
            )
            l.backward()
            optimizer.step()
            total += l.item()
            steps += 1
        epoch_losses.append(total / steps)

    assert epoch_losses[2] < epoch_losses[0], (
        f"Loss did not decrease: {epoch_losses}"
    )


def test_checkpoint_save_load(tmp_path):
    """Save and reload projection + heads; outputs must match."""
    from src.train.train_global import save_checkpoint, load_checkpoint

    torch.manual_seed(0)
    proj, heads = _make_smoke_model()
    feats = torch.randn(4, 64)

    ckpt_path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(proj, heads, ckpt_path, epoch=1, val_loss=0.5)

    proj2, heads2 = _make_smoke_model()
    meta = load_checkpoint(proj2, heads2, ckpt_path)

    assert meta["epoch"] == 1
    assert abs(meta["val_loss"] - 0.5) < 1e-6

    proj.eval(); proj2.eval()
    heads.eval(); heads2.eval()
    with torch.no_grad():
        p1 = proj(feats)
        p2 = proj2(feats)
        f1, g1, s1 = heads(feats)
        f2, g2, s2 = heads2(feats)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)
