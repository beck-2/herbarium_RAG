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

# TODO(phase7): implement precision_at_k
# TODO(phase7): implement hierarchical_accuracy (family, genus, species)
# TODO(phase7): implement mistake_severity (mean LCA height)
# TODO(phase7): implement expected_calibration_error (ECE)
# TODO(phase7): implement open_set_recall


def precision_at_k(retrieved_ids: list[list[str]], true_ids: list[str], k: int) -> float:
    """Compute precision@k across a set of queries.

    Args:
        retrieved_ids: List of per-query top-k retrieved occurrence IDs.
        true_ids: List of true occurrence IDs (one per query).
        k: Cutoff rank.

    Returns:
        Mean precision@k across queries.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def mistake_severity(
    predictions: list[str],
    ground_truth: list[str],
    opentree_subtree: dict,
) -> float:
    """Compute mean LCA height for incorrect predictions.

    Height 1 = same genus, 2 = same family, 3 = same order, etc.
    Target: <2 (errors within genus on average).

    Args:
        predictions: Predicted species names.
        ground_truth: True species names.
        opentree_subtree: Subtree dict for LCA computation.

    Returns:
        Mean LCA height for misclassified queries.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
