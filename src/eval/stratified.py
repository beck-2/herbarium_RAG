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

import pandas as pd

# TODO(phase7): implement stratified_evaluate (calls metrics.py per stratum)
# TODO(phase7): implement convergent_pair_confusion_rate

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

    Returns:
        DataFrame with columns: [stratum_type, stratum_value, metric, value].
    """
    raise NotImplementedError


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
    raise NotImplementedError
