"""
Phase 7 evaluation metric tests.

All tests use synthetic data or the committed OpenTree fixture —
no network access, no model, no real specimens required.

Fixture: tests/fixtures/opentree_mini_subtree.json
  Species: Poa pratensis, Poa annua, Festuca rubra, Quercus robur,
           Helianthus annuus, Carnegiea gigantea, Euphorbia milii,
           Drosera rotundifolia, Nepenthes mirabilis, Taraxacum officinale
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture: load the real OpenTree mini subtree once per session
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "opentree_mini_subtree.json"


@pytest.fixture(scope="session")
def opentree_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 0 stubs (kept)
# ---------------------------------------------------------------------------


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
    from src.eval.stratified import CONVERGENT_PAIRS
    assert len(CONVERGENT_PAIRS) > 0
    for pair in CONVERGENT_PAIRS:
        assert len(pair) == 2
        assert all(isinstance(name, str) for name in pair)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_perfect():
    from src.eval.metrics import precision_at_k
    retrieved = [["sp1", "sp2", "sp3"]]
    true = ["sp1"]
    assert precision_at_k(retrieved, true, k=3) == 1.0


def test_precision_at_k_miss():
    from src.eval.metrics import precision_at_k
    retrieved = [["sp2", "sp3", "sp4"]]
    true = ["sp1"]
    assert precision_at_k(retrieved, true, k=3) == 0.0


def test_precision_at_k_partial():
    from src.eval.metrics import precision_at_k
    retrieved = [["sp1", "sp2"], ["sp3", "sp4"], ["sp5", "sp6"]]
    true = ["sp1", "sp3", "sp99"]  # 2 hits out of 3
    assert abs(precision_at_k(retrieved, true, k=2) - 2 / 3) < 1e-9


def test_precision_at_k_cutoff_respected():
    from src.eval.metrics import precision_at_k
    # true_id is at position k+1 (index 3) but k=3 means we only look at [:3]
    retrieved = [["sp2", "sp3", "sp4", "sp1"]]
    true = ["sp1"]
    assert precision_at_k(retrieved, true, k=3) == 0.0
    assert precision_at_k(retrieved, true, k=4) == 1.0


def test_precision_at_k_k_larger_than_retrieved():
    from src.eval.metrics import precision_at_k
    retrieved = [["sp1"]]  # only 1 candidate
    true = ["sp1"]
    assert precision_at_k(retrieved, true, k=50) == 1.0  # no crash, still found


# ---------------------------------------------------------------------------
# hierarchical_accuracy
# ---------------------------------------------------------------------------


def test_hierarchical_accuracy_perfect():
    from src.eval.metrics import hierarchical_accuracy
    preds = [{"family": "Poaceae", "genus": "Poa", "species": "Poa pratensis"}]
    gt = [{"family": "Poaceae", "genus": "Poa", "species": "Poa pratensis"}]
    result = hierarchical_accuracy(preds, gt)
    assert result["family"] == 1.0
    assert result["genus"] == 1.0
    assert result["species"] == 1.0


def test_hierarchical_accuracy_family_right_species_wrong():
    from src.eval.metrics import hierarchical_accuracy
    preds = [{"family": "Poaceae", "genus": "Festuca", "species": "Festuca rubra"}]
    gt = [{"family": "Poaceae", "genus": "Poa", "species": "Poa pratensis"}]
    result = hierarchical_accuracy(preds, gt)
    assert result["family"] == 1.0
    assert result["genus"] == 0.0
    assert result["species"] == 0.0


def test_hierarchical_accuracy_all_wrong():
    from src.eval.metrics import hierarchical_accuracy
    preds = [{"family": "Fagaceae", "genus": "Quercus", "species": "Quercus robur"}]
    gt = [{"family": "Poaceae", "genus": "Poa", "species": "Poa pratensis"}]
    result = hierarchical_accuracy(preds, gt)
    assert result["family"] == 0.0
    assert result["genus"] == 0.0
    assert result["species"] == 0.0


def test_hierarchical_accuracy_multiple_queries():
    from src.eval.metrics import hierarchical_accuracy
    preds = [
        {"family": "Poaceae", "genus": "Poa", "species": "Poa pratensis"},  # all correct
        {"family": "Poaceae", "genus": "Festuca", "species": "Festuca rubra"},  # family correct
        {"family": "Fagaceae", "genus": "Quercus", "species": "Quercus robur"},  # all wrong
    ]
    gt = [
        {"family": "Poaceae", "genus": "Poa", "species": "Poa pratensis"},
        {"family": "Poaceae", "genus": "Poa", "species": "Poa annua"},
        {"family": "Poaceae", "genus": "Poa", "species": "Poa annua"},
    ]
    result = hierarchical_accuracy(preds, gt)
    assert abs(result["family"] - 2 / 3) < 1e-9
    assert abs(result["genus"] - 1 / 3) < 1e-9
    assert abs(result["species"] - 1 / 3) < 1e-9


# ---------------------------------------------------------------------------
# mistake_severity  (uses real OpenTree fixture for LCA lookups)
# ---------------------------------------------------------------------------


def test_mistake_severity_no_mistakes(opentree_fixture):
    from src.eval.metrics import mistake_severity
    preds = ["Poa pratensis", "Poa annua"]
    truths = ["Poa pratensis", "Poa annua"]
    assert mistake_severity(preds, truths, opentree_fixture) == 0.0


def test_mistake_severity_same_genus(opentree_fixture):
    """Poa pratensis predicted as Poa annua → LCA = genus → height 1."""
    from src.eval.metrics import mistake_severity
    preds = ["Poa annua"]
    truths = ["Poa pratensis"]
    result = mistake_severity(preds, truths, opentree_fixture)
    assert result == 1.0


def test_mistake_severity_same_family(opentree_fixture):
    """Helianthus annuus predicted as Taraxacum officinale → LCA = family → height 2."""
    from src.eval.metrics import mistake_severity
    preds = ["Taraxacum officinale"]
    truths = ["Helianthus annuus"]
    result = mistake_severity(preds, truths, opentree_fixture)
    assert result == 2.0


def test_mistake_severity_order_level(opentree_fixture):
    """Drosera rotundifolia vs Nepenthes mirabilis → LCA = order → height 3."""
    from src.eval.metrics import mistake_severity
    preds = ["Nepenthes mirabilis"]
    truths = ["Drosera rotundifolia"]
    result = mistake_severity(preds, truths, opentree_fixture)
    assert result == 3.0


def test_mistake_severity_missing_name_uses_default(opentree_fixture):
    """Unknown species names fall back to default height 3, no crash."""
    from src.eval.metrics import mistake_severity
    preds = ["Unknown species A"]
    truths = ["Unknown species B"]
    result = mistake_severity(preds, truths, opentree_fixture)
    assert result == 3.0


def test_mistake_severity_mixed(opentree_fixture):
    """genus mistake (h=1) + family mistake (h=2) → mean 1.5."""
    from src.eval.metrics import mistake_severity
    preds = ["Poa annua", "Taraxacum officinale"]
    truths = ["Poa pratensis", "Helianthus annuus"]
    result = mistake_severity(preds, truths, opentree_fixture)
    assert abs(result - 1.5) < 1e-9


def test_mistake_severity_ignores_correct(opentree_fixture):
    """Correct predictions are not included in the mean."""
    from src.eval.metrics import mistake_severity
    preds = ["Poa pratensis", "Poa annua"]       # first is correct, second is mistake
    truths = ["Poa pratensis", "Poa pratensis"]
    result = mistake_severity(preds, truths, opentree_fixture)
    assert result == 1.0  # only the wrong one (genus height 1) counts


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration():
    """Probabilities perfectly match accuracy within each bin → ECE = 0."""
    from src.eval.metrics import expected_calibration_error
    # All-0 confidence + all-wrong, all-1 confidence + all-correct
    probs = np.array([0.0] * 50 + [1.0] * 50)
    labels = np.array([0] * 50 + [1] * 50, dtype=float)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident():
    """Always predict 1.0 but only 50% correct → ECE = 0.5."""
    from src.eval.metrics import expected_calibration_error
    probs = np.ones(100)
    labels = np.array([0, 1] * 50, dtype=float)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.5, abs=1e-6)


def test_ece_empty_bins_ignored():
    """Sparse data → empty bins don't cause errors or inflate ECE."""
    from src.eval.metrics import expected_calibration_error
    probs = np.array([0.2, 0.8])
    labels = np.array([0.0, 1.0])
    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_ece_range_zero_to_one():
    """ECE is always in [0, 1]."""
    from src.eval.metrics import expected_calibration_error
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, size=200)
    labels = rng.integers(0, 2, size=200).astype(float)
    ece = expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# open_set_recall
# ---------------------------------------------------------------------------


def test_open_set_recall_all_flagged():
    from src.eval.metrics import open_set_recall
    scores = np.array([0.9, 0.8, 0.95])
    is_open = np.array([True, True, True])
    assert open_set_recall(scores, is_open, threshold=0.5) == 1.0


def test_open_set_recall_none_flagged():
    from src.eval.metrics import open_set_recall
    scores = np.array([0.1, 0.2, 0.3])
    is_open = np.array([True, True, True])
    assert open_set_recall(scores, is_open, threshold=0.5) == 0.0


def test_open_set_recall_partial():
    from src.eval.metrics import open_set_recall
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
    is_open = np.array([True, True, True, True, True])
    assert open_set_recall(scores, is_open, threshold=0.5) == pytest.approx(3 / 5)


def test_open_set_recall_no_open_set_queries():
    from src.eval.metrics import open_set_recall
    scores = np.array([0.9, 0.8])
    is_open = np.array([False, False])
    assert open_set_recall(scores, is_open, threshold=0.5) == 0.0


def test_open_set_recall_in_set_queries_not_counted():
    """In-distribution queries flagged above threshold should not count as TP."""
    from src.eval.metrics import open_set_recall
    scores = np.array([0.9, 0.9, 0.1])  # first two above threshold
    is_open = np.array([False, True, True])  # only second and third are open-set
    # Only index 1 (open-set AND above threshold) counts
    assert open_set_recall(scores, is_open, threshold=0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# stratified_evaluate
# ---------------------------------------------------------------------------


def _make_synthetic_eval_data(n=20, seed=0):
    """Helper: create n synthetic prediction + ground-truth dicts."""
    rng = np.random.default_rng(seed)
    families = ["Poaceae", "Asteraceae", "Fagaceae"]
    genera = ["Poa", "Helianthus", "Quercus"]
    species = ["Poa pratensis", "Helianthus annuus", "Quercus robur"]
    rarity_tiers = ["abundant", "moderate"]
    subregions = ["california", "pacific_northwest"]

    preds = []
    gt = []
    tiers = []
    regions = []
    for i in range(n):
        true_idx = i % 3
        pred_idx = (i + (1 if i % 5 == 0 else 0)) % 3  # some wrong predictions
        true_id = f"spec_{i:04d}"
        retrieved = [true_id] + [f"other_{j}" for j in range(9)]
        if i % 5 == 0:
            retrieved = [f"other_{j}" for j in range(10)]  # miss for precision

        preds.append({
            "family": families[pred_idx],
            "genus": genera[pred_idx],
            "species": species[pred_idx],
            "retrieved_ids": retrieved,
            "top1_confidence": float(rng.uniform(0.5, 1.0)),
            "uncertainty_score": float(rng.uniform(0.0, 1.0)),
        })
        gt.append({
            "family": families[true_idx],
            "genus": genera[true_idx],
            "species": species[true_idx],
            "true_id": true_id,
            "is_open_set": bool(i % 7 == 0),
        })
        tiers.append(rarity_tiers[i % 2])
        regions.append(subregions[i % 2])

    return preds, gt, tiers, regions


def test_stratified_evaluate_returns_dataframe(opentree_fixture):
    from src.eval.stratified import stratified_evaluate
    preds, gt, tiers, regions = _make_synthetic_eval_data()
    result = stratified_evaluate(preds, gt, tiers, regions, opentree_fixture)
    assert isinstance(result, pd.DataFrame)


def test_stratified_evaluate_columns(opentree_fixture):
    from src.eval.stratified import stratified_evaluate
    preds, gt, tiers, regions = _make_synthetic_eval_data()
    result = stratified_evaluate(preds, gt, tiers, regions, opentree_fixture)
    assert set(result.columns) == {"stratum_type", "stratum_value", "metric", "value"}


def test_stratified_evaluate_has_rarity_and_subregion_strata(opentree_fixture):
    from src.eval.stratified import stratified_evaluate
    preds, gt, tiers, regions = _make_synthetic_eval_data()
    result = stratified_evaluate(preds, gt, tiers, regions, opentree_fixture)
    assert "rarity" in result["stratum_type"].values
    assert "subregion" in result["stratum_type"].values


def test_stratified_evaluate_includes_key_metrics(opentree_fixture):
    from src.eval.stratified import stratified_evaluate
    preds, gt, tiers, regions = _make_synthetic_eval_data()
    result = stratified_evaluate(preds, gt, tiers, regions, opentree_fixture)
    metrics_present = set(result["metric"].unique())
    for expected in ("precision_at_1", "family_accuracy", "genus_accuracy",
                     "species_accuracy", "mistake_severity", "ece"):
        assert expected in metrics_present, f"Missing metric: {expected}"


def test_stratified_evaluate_values_are_floats(opentree_fixture):
    from src.eval.stratified import stratified_evaluate
    preds, gt, tiers, regions = _make_synthetic_eval_data()
    result = stratified_evaluate(preds, gt, tiers, regions, opentree_fixture)
    assert result["value"].dtype == float or result["value"].apply(
        lambda x: isinstance(x, (float, int))
    ).all()


# ---------------------------------------------------------------------------
# convergent_pair_confusion_rate
# ---------------------------------------------------------------------------


def test_convergent_pair_no_confusion():
    from src.eval.stratified import convergent_pair_confusion_rate, CONVERGENT_PAIRS
    # All Cactaceae specimens correctly predicted as Cactaceae
    preds = [{"family": "Cactaceae"}] * 5
    gt = [{"family": "Cactaceae"}] * 5
    result = convergent_pair_confusion_rate(preds, gt)
    key = "Cactaceae_vs_Euphorbiaceae"
    assert result[key] == 0.0


def test_convergent_pair_full_confusion():
    from src.eval.stratified import convergent_pair_confusion_rate
    # All Cactaceae specimens predicted as Euphorbiaceae
    preds = [{"family": "Euphorbiaceae"}] * 4
    gt = [{"family": "Cactaceae"}] * 4
    result = convergent_pair_confusion_rate(preds, gt)
    assert result["Cactaceae_vs_Euphorbiaceae"] == 1.0


def test_convergent_pair_partial_confusion():
    from src.eval.stratified import convergent_pair_confusion_rate
    preds = [
        {"family": "Euphorbiaceae"},  # confused
        {"family": "Cactaceae"},      # correct
        {"family": "Cactaceae"},      # correct
        {"family": "Cactaceae"},      # correct
    ]
    gt = [{"family": "Cactaceae"}] * 4
    result = convergent_pair_confusion_rate(preds, gt)
    assert result["Cactaceae_vs_Euphorbiaceae"] == pytest.approx(0.25)


def test_convergent_pair_returns_all_pairs():
    from src.eval.stratified import convergent_pair_confusion_rate, CONVERGENT_PAIRS
    preds = [{"family": "Poaceae"}]
    gt = [{"family": "Poaceae"}]
    result = convergent_pair_confusion_rate(preds, gt)
    # All convergent pairs should have a key in the result
    for fam_a, fam_b in CONVERGENT_PAIRS:
        assert f"{fam_a}_vs_{fam_b}" in result


def test_convergent_pair_both_directions():
    """Confusion is counted in both directions (A→B and B→A)."""
    from src.eval.stratified import convergent_pair_confusion_rate
    preds = [
        {"family": "Euphorbiaceae"},  # Cactaceae predicted as Euphorbiaceae (A→B)
        {"family": "Cactaceae"},      # Euphorbiaceae predicted as Cactaceae (B→A)
    ]
    gt = [
        {"family": "Cactaceae"},
        {"family": "Euphorbiaceae"},
    ]
    result = convergent_pair_confusion_rate(preds, gt)
    assert result["Cactaceae_vs_Euphorbiaceae"] == 1.0
