"""
Evaluation metrics.

Implements all metrics from SPEC §10.1:

| Metric                        | Target |
|-------------------------------|--------|
| Retrieval precision@1         | >70%   |
| Retrieval precision@5         | >85%   |
| Family accuracy               | >95%   |
| Genus accuracy                | >85%   |
| Species accuracy              | >70%   |
| Mistake severity (mean LCA)   | <2     |
| Calibration ECE               | <0.05  |
| Open-set recall               | >60%   |
| End-to-end latency            | <3s    |
"""

from __future__ import annotations

import numpy as np

# Taxonomic rank → integer height for mistake_severity.
# "no rank" intermediate clades (e.g. Poeae Chloroplast Group 2) are treated
# as order-level (3) since they sit above family in the OpenTree hierarchy.
_RANK_HEIGHT: dict[str, int] = {
    "species": 0,
    "genus":   1,
    "family":  2,
    "order":   3,
    "class":   4,
    "phylum":  5,
    "kingdom": 6,
}
_DEFAULT_HEIGHT = 3  # fallback for "no rank" or unknown names


def precision_at_k(retrieved_ids: list[list[str]], true_ids: list[str], k: int) -> float:
    """Compute precision@k across a set of queries.

    Args:
        retrieved_ids: List of per-query top-k retrieved occurrence IDs.
        true_ids: List of true occurrence IDs (one per query).
        k: Cutoff rank.

    Returns:
        Mean precision@k across queries.
    """
    if not retrieved_ids:
        return 0.0
    hits = sum(
        1 for retrieved, true in zip(retrieved_ids, true_ids)
        if true in retrieved[:k]
    )
    return hits / len(retrieved_ids)


def hierarchical_accuracy(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict[str, float]:
    """Compute family, genus, and species top-1 accuracy.

    Args:
        predictions: List of dicts with keys 'family', 'genus', 'species'.
        ground_truth: List of dicts with the same keys.

    Returns:
        Dict: {'family': float, 'genus': float, 'species': float}.
    """
    n = len(predictions)
    if n == 0:
        return {"family": 0.0, "genus": 0.0, "species": 0.0}
    fam = sum(p["family"] == g["family"] for p, g in zip(predictions, ground_truth))
    gen = sum(p["genus"] == g["genus"] for p, g in zip(predictions, ground_truth))
    spe = sum(p["species"] == g["species"] for p, g in zip(predictions, ground_truth))
    return {"family": fam / n, "genus": gen / n, "species": spe / n}


def mistake_severity(
    predictions: list[str],
    ground_truth: list[str],
    opentree_subtree: dict,
) -> float:
    """Compute mean LCA height for incorrect predictions.

    Height 1 = same genus, 2 = same family, 3 = same order / no-rank, etc.
    Target: <2 (errors within genus on average).

    Uses get_lca_rank from taxonomy.opentree with the real lineage data stored
    in the bundle's opentree_subtree.json.  Requires the subtree to have a
    'name_to_ott' dict mapping species names → OTT IDs (added by
    scripts/fetch_opentree_fixture.py and future bundle construction).

    Args:
        predictions: Predicted species names.
        ground_truth: True species names.
        opentree_subtree: Subtree dict including 'lineages' and 'name_to_ott'.

    Returns:
        Mean LCA height for misclassified queries.  Returns 0.0 if no mistakes.
    """
    from src.taxonomy.opentree import get_lca_rank

    name_to_ott: dict[str, int] = opentree_subtree.get("name_to_ott", {})
    heights: list[int] = []

    for pred, true in zip(predictions, ground_truth):
        if pred == true:
            continue  # correct prediction — not counted
        ott_pred = name_to_ott.get(pred)
        ott_true = name_to_ott.get(true)
        if ott_pred is not None and ott_true is not None:
            rank = get_lca_rank(ott_pred, ott_true, opentree_subtree)
            height = _RANK_HEIGHT.get(rank, _DEFAULT_HEIGHT)
        else:
            height = _DEFAULT_HEIGHT
        heights.append(height)

    return float(np.mean(heights)) if heights else 0.0


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probabilities: Predicted confidence scores, shape (N,).
        labels: Binary correct/incorrect indicators, shape (N,).
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE as a float in [0, 1]. Target: <0.05.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probabilities)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bin_conf = probabilities[mask].mean()
        bin_acc = labels[mask].mean()
        ece += n_bin * abs(bin_conf - bin_acc)

    return ece / n


def open_set_recall(
    uncertainty_scores: np.ndarray,
    is_open_set: np.ndarray,
    threshold: float,
) -> float:
    """Compute fraction of open-set queries (held-out genera) correctly flagged.

    Args:
        uncertainty_scores: Per-query uncertainty scores.
        is_open_set: Boolean array — True if query is from a held-out genus.
        threshold: Uncertainty threshold above which a query is flagged as open-set.

    Returns:
        Recall as float in [0, 1]. Target: >0.60.
    """
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=float)
    is_open_set = np.asarray(is_open_set, dtype=bool)
    n_open = int(is_open_set.sum())
    if n_open == 0:
        return 0.0
    flagged_open = int(((uncertainty_scores > threshold) & is_open_set).sum())
    return flagged_open / n_open
