"""
Stratified evaluation — break down all metrics by subgroup.

Reports metrics stratified by (from SPEC §10.2):
  - Rarity tier: abundant (>50 imgs), moderate (10–50), rare (5–10)
  - Taxonomic difficulty: same-genus pairs, convergent morphology pairs
  - Geographic subregion
  - Phenological stage (where labeled)

Key scientific evaluation (SPEC §10.2):
  Primary evidence for hyperbolic geometry: improvement in confusion rate
  for known convergent pairs (Cactaceae vs. Euphorbiaceae succulents,
  Droseraceae vs. Nepenthaceae carnivores) vs. Euclidean baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONVERGENT_PAIRS = [
    ("Cactaceae", "Euphorbiaceae"),      # succulent convergence
    ("Droseraceae", "Nepenthaceae"),     # carnivorous plant convergence
    ("Apocynaceae", "Asclepiadaceae"),   # milkweed family (taxonomic controversy)
]


def stratified_evaluate(
    predictions: list[dict],
    ground_truth: list[dict],
    rarity_tiers: list[str],
    subregions: list[str],
    opentree_subtree: dict,
) -> pd.DataFrame:
    """Compute all metrics stratified by rarity tier and subregion.

    Each predictions[i] dict must contain:
        'family', 'genus', 'species'   — predicted taxonomy
        'retrieved_ids'                — list[str] top-k retrieved IDs
        'top1_confidence'              — float confidence for top-1 prediction
        'uncertainty_score'            — float uncertainty (for open-set recall)

    Each ground_truth[i] dict must contain:
        'family', 'genus', 'species'   — true taxonomy
        'true_id'                      — str correct occurrence ID
        'is_open_set'                  — bool

    Args:
        predictions:     Per-query prediction dicts.
        ground_truth:    Per-query ground-truth dicts.
        rarity_tiers:    Per-query rarity tier ('abundant', 'moderate', 'rare').
        subregions:      Per-query geographic subregion label.
        opentree_subtree: Bundle subtree dict (for mistake_severity).

    Returns:
        DataFrame with columns: [stratum_type, stratum_value, metric, value].
    """
    from src.eval.metrics import (
        precision_at_k,
        hierarchical_accuracy,
        mistake_severity,
        expected_calibration_error,
        open_set_recall,
    )

    records: list[dict] = []

    def _compute(group_preds: list[dict], group_gt: list[dict],
                 stratum_type: str, stratum_value: str) -> None:
        n = len(group_preds)
        if n == 0:
            return

        # --- hierarchical accuracy ---
        ha = hierarchical_accuracy(group_preds, group_gt)
        for level, acc in ha.items():
            records.append({
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "metric": f"{level}_accuracy",
                "value": float(acc),
            })

        # --- precision@1 and @5 ---
        retrieved = [p.get("retrieved_ids", []) for p in group_preds]
        true_ids = [g.get("true_id", "") for g in group_gt]
        for k in (1, 5):
            p = precision_at_k(retrieved, true_ids, k=k)
            records.append({
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "metric": f"precision_at_{k}",
                "value": float(p),
            })

        # --- mistake severity ---
        pred_species = [p.get("species", "") for p in group_preds]
        true_species = [g.get("species", "") for g in group_gt]
        ms = mistake_severity(pred_species, true_species, opentree_subtree)
        records.append({
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "metric": "mistake_severity",
            "value": float(ms),
        })

        # --- ECE ---
        probs = np.array([p.get("top1_confidence", 0.5) for p in group_preds])
        correct = np.array([
            float(p.get("species", "") == g.get("species", ""))
            for p, g in zip(group_preds, group_gt)
        ])
        ece = expected_calibration_error(probs, correct)
        records.append({
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "metric": "ece",
            "value": float(ece),
        })

        # --- open-set recall (threshold=0.5 for stratified reporting) ---
        unc = np.array([p.get("uncertainty_score", 0.5) for p in group_preds])
        is_open = np.array([g.get("is_open_set", False) for g in group_gt], dtype=bool)
        osr = open_set_recall(unc, is_open, threshold=0.5)
        records.append({
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "metric": "open_set_recall",
            "value": float(osr),
        })

    # --- group by rarity tier ---
    tier_groups: dict[str, list[int]] = {}
    for i, tier in enumerate(rarity_tiers):
        tier_groups.setdefault(tier, []).append(i)
    for tier, indices in tier_groups.items():
        _compute([predictions[i] for i in indices],
                 [ground_truth[i] for i in indices],
                 "rarity", tier)

    # --- group by subregion ---
    region_groups: dict[str, list[int]] = {}
    for i, region in enumerate(subregions):
        region_groups.setdefault(region, []).append(i)
    for region, indices in region_groups.items():
        _compute([predictions[i] for i in indices],
                 [ground_truth[i] for i in indices],
                 "subregion", region)

    return pd.DataFrame(records, columns=["stratum_type", "stratum_value", "metric", "value"])


def convergent_pair_confusion_rate(
    predictions: list[dict],
    ground_truth: list[dict],
    convergent_pairs: list[tuple[str, str]] = CONVERGENT_PAIRS,
) -> dict[str, float]:
    """Compute confusion rate within each convergent morphology pair.

    Returns:
        Dict mapping '{family_a}_vs_{family_b}' → confusion rate (fraction of
        queries from family_a predicted as family_b, or vice versa).
    """
    result: dict[str, float] = {}
    for fam_a, fam_b in convergent_pairs:
        key = f"{fam_a}_vs_{fam_b}"
        pair_indices = [
            i for i, g in enumerate(ground_truth)
            if g.get("family") in (fam_a, fam_b)
        ]
        if not pair_indices:
            result[key] = 0.0
            continue
        confusions = 0
        for i in pair_indices:
            true_fam = ground_truth[i].get("family")
            pred_fam = predictions[i].get("family")
            if (true_fam == fam_a and pred_fam == fam_b) or \
               (true_fam == fam_b and pred_fam == fam_a):
                confusions += 1
        result[key] = confusions / len(pair_indices)
    return result
