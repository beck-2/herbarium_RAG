"""
Poincaré disk layout export for the React Native frontend.

Outputs a JSON payload consumed by the mobile app's D3-compatible visualization.
NO text generation — all labels come from specimen metadata.

JSON contract (View 1 — Poincaré Disk):
{
  "query": {
    "poincare": [x, y],        // 2D Poincaré disk coordinates (projected from 512-d)
    "uncertainty_radius": 0.12  // blur radius = spread of retrieved points
  },
  "candidates": [
    {
      "id": "occurrence_id",
      "poincare": [x, y],
      "taxon": "Clarkia gracilis",
      "family": "Onagraceae",
      "genus": "Clarkia",
      "score": 0.87,
      "lca_rank": "genus"       // LCA rank with query's top hit
    }, ...
  ],
  "clade_arcs": [
    {
      "family": "Onagraceae",
      "geodesic_points": [[x1, y1], [x2, y2], ...] // boundary arc points
    }, ...
  ],
  "open_set_signal": false,     // true if uncertainty_score > threshold
  "uncertainty_score": 0.23
}
"""

from __future__ import annotations

import json
import numpy as np

# TODO(phase9): implement 2D projection from 512-d Poincaré points (UMAP on disk)
# TODO(phase9): implement clade arc computation (geodesic arcs in Poincaré disk)
# TODO(phase9): implement open-set signal threshold
# TODO(phase9): implement full export_disk_json


def project_to_2d(poincare_points: np.ndarray) -> np.ndarray:
    """Project 512-d Poincaré ball points to 2D Poincaré disk for visualization.

    Uses UMAP with Poincaré metric, constrained to the unit disk.

    Args:
        poincare_points: Array of shape (N, 512).

    Returns:
        Array of shape (N, 2) with all points inside the unit disk (norm < 1).
    """
    raise NotImplementedError


def compute_uncertainty(candidate_poincare_2d: np.ndarray) -> float:
    """Compute uncertainty score as spread (mean pairwise distance) of candidates in 2D disk.

    Returns:
        Float uncertainty score. Higher = more spread = less confident.
    """
    raise NotImplementedError


def export_disk_json(
    query_poincare: np.ndarray,
    candidates: list[dict],
    opentree_subtree: dict,
    open_set_threshold: float = 0.5,
) -> dict:
    """Assemble the full Poincaré disk JSON payload.

    Args:
        query_poincare: 512-d query Poincaré point.
        candidates: Top-k candidate dicts with scores and metadata.
        opentree_subtree: Subtree for clade arc computation.
        open_set_threshold: Uncertainty score above which open_set_signal = true.

    Returns:
        Dict matching the JSON contract above.
    """
    raise NotImplementedError
