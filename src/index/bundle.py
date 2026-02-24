"""
Regional bundle packing.

Assembles all artifacts for a region into the distributable bundle format:

    bundles/{region}/
      manifest.json               version, taxonomy_version, creation_date, stats
      encoder_base.bin            int8-quantized BioCLIP-2 (~300MB, shared across regions)
      lora_{region}.safetensors   LoRA adapter (~35MB)
      hyperbolic_proj.pt          projection layer (~2MB)
      classifier_heads.pt         hierarchical heads (~8MB)
      faiss_global.bin            IVF-PQ global index (~50MB)
      faiss_families/             family sub-indexes (~20MB total)
      specimens.db                SQLite specimen metadata
      thumbnails/                 128×128 JPEGs (~100MB for 100K specimens)
      opentree_subtree.json       relevant OpenTree subtree

Bundle size targets (SPEC §7.2):
    Base model (once):   ~300MB
    Per-region bundle:   ~215–320MB
    Total (1 region):    ~515–620MB   ← flag if > 400MB (DECISION-1 review trigger)
"""

from __future__ import annotations

import json
from pathlib import Path

# TODO(phase5): implement bundle assembly and manifest generation
# TODO(phase5): implement thumbnail generation (128×128 JPEG resize)
# TODO(phase5): implement specimens.db creation (SQLite metadata)
# TODO(phase5): check total bundle size and warn if > 400MB (DECISION-1 trigger)


def pack_bundle(
    region: str,
    checkpoint_dir: str,
    index_dir: str,
    specimens_parquet: str,
    image_dir: str,
    opentree_subtree_json: str,
    output_dir: str,
    encoder_base_path: str | None = None,
) -> dict:
    """Assemble all artifacts into a regional bundle directory.

    Args:
        region: Region key (e.g. 'california').
        checkpoint_dir: Directory containing global/lora checkpoints.
        index_dir: Directory containing FAISS indexes.
        specimens_parquet: Path to processed specimens Parquet.
        image_dir: Directory containing raw specimen images.
        opentree_subtree_json: Path to exported OpenTree subtree JSON.
        output_dir: Output bundle directory (created if absent).
        encoder_base_path: Path to shared encoder_base.bin (if already quantized).

    Returns:
        manifest dict (also written to manifest.json in output_dir).
    """
    raise NotImplementedError


def generate_thumbnails(
    image_dir: str,
    specimen_ids: list[str],
    output_dir: str,
    size: tuple[int, int] = (128, 128),
) -> None:
    """Resize specimen images to thumbnails for the bundle's `thumbnails/` directory."""
    raise NotImplementedError


def create_specimens_db(df, output_db_path: str) -> None:
    """Write specimen metadata to SQLite.

    Columns: occurrence_id, scientific_name, ott_id, family, genus, species,
             latitude, longitude, state_province, event_date, institution, image_url.
    """
    raise NotImplementedError


def check_bundle_size(bundle_dir: str, warn_threshold_mb: float = 400.0) -> float:
    """Compute total bundle size in MB. Logs a warning if > warn_threshold_mb.

    Returns:
        Total size in MB.
    """
    raise NotImplementedError
