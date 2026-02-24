"""
Hierarchical confidence bar data export for the React Native frontend.

Outputs a JSON payload for View 3 (Hierarchical Confidence Bars) and
the Retrieved Specimens Panel (View 2).

JSON contract (View 3 — Confidence Bars):
{
  "hierarchy": [
    {
      "level": "family",
      "name": "Onagraceae",
      "confidence": 0.94
    },
    {
      "level": "genus",
      "name": "Clarkia",
      "confidence": 0.78
    },
    {
      "level": "species",
      "candidates": [
        {"name": "Clarkia gracilis",    "confidence": 0.43},
        {"name": "Clarkia unguiculata", "confidence": 0.38},
        {"name": "Clarkia cylindrica",  "confidence": 0.11}
      ]
    }
  ]
}

JSON contract (View 2 — Retrieved Specimens Panel):
{
  "specimens": [
    {
      "id": "occurrence_id",
      "taxon": "Clarkia gracilis",
      "date": "2019-05-12",
      "locality": "Sonoma County, California",
      "institution": "UC Berkeley Herbarium",
      "thumbnail_path": "thumbnails/occ_12345.jpg",
      "similarity_distance": 0.142   // geodesic distance, NOT a percentage
    }, ...
  ]
}
"""

from __future__ import annotations

# TODO(phase9): implement export_confidence_json
# TODO(phase9): implement export_specimens_panel_json


def export_confidence_json(
    family_probs: "torch.Tensor",
    genus_probs: "torch.Tensor",
    species_probs: "torch.Tensor",
    family_names: list[str],
    genus_names: list[str],
    species_names: list[str],
    top_k_species: int = 5,
) -> dict:
    """Assemble hierarchical confidence bar JSON payload.

    Args:
        family_probs: Softmax probabilities, shape (n_families,).
        genus_probs:  Softmax probabilities, shape (n_genera,).
        species_probs: Softmax probabilities, shape (n_species,).
        family_names, genus_names, species_names: Class name lists.
        top_k_species: Number of top species candidates to include.

    Returns:
        Dict matching the confidence bars JSON contract.
    """
    raise NotImplementedError


def export_specimens_panel_json(
    candidates: list[dict],
    specimens_db_path: str,
    thumbnails_dir: str,
) -> dict:
    """Assemble the retrieved specimens panel JSON payload.

    Distances in the output are Poincaré geodesic distances, NOT percentages.

    Args:
        candidates: Top-k candidate dicts from retrieval.search.retrieve.
        specimens_db_path: Path to SQLite specimens.db in the bundle.
        thumbnails_dir: Path to thumbnails/ directory in the bundle.

    Returns:
        Dict matching the specimens panel JSON contract.
    """
    raise NotImplementedError
