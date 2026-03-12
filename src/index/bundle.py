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
import shutil
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path


_BUNDLE_VERSION = "0.1.0"

# Columns written to specimens.db (subset of canonical schema)
_DB_COLUMNS = [
    "occurrence_id",
    "scientific_name",
    "ott_id",
    "family",
    "genus",
    "species",
    "latitude",
    "longitude",
    "state_province",
    "event_date",
    "institution",
    "image_url",
]


def pack_bundle(
    region: str,
    checkpoint_dir: str,
    index_dir: str,
    specimens_parquet: str,
    image_dir: str | None,
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
        image_dir: Directory containing raw specimen images, or None to skip thumbnails.
        opentree_subtree_json: Path to exported OpenTree subtree JSON.
        output_dir: Output bundle directory (created if absent).
        encoder_base_path: Path to shared encoder_base.bin (if already quantized).

    Returns:
        manifest dict (also written to manifest.json in output_dir).
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "faiss_families").mkdir(exist_ok=True)
    (out / "thumbnails").mkdir(exist_ok=True)

    # --- Load specimens ---
    df = pd.read_parquet(specimens_parquet)
    n_specimens = len(df)
    families = sorted(df["family"].dropna().unique().tolist()) if "family" in df.columns else []

    # --- Copy FAISS global index ---
    global_bin = Path(index_dir) / "faiss_global.bin"
    if global_bin.exists():
        shutil.copy2(global_bin, out / "faiss_global.bin")

    # --- Copy family sub-indexes ---
    families_src = Path(index_dir) / "faiss_families"
    if families_src.exists():
        for f in families_src.iterdir():
            shutil.copy2(f, out / "faiss_families" / f.name)

    # --- Copy model checkpoints ---
    ckpt = Path(checkpoint_dir)
    for fname in ["best.pt", "hyperbolic_proj.pt", "classifier_heads.pt"]:
        src = ckpt / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    # Copy LoRA adapter if present
    for fname in ckpt.glob(f"lora_{region}*"):
        shutil.copy2(fname, out / fname.name)
    for fname in ckpt.glob("adapter_*"):
        shutil.copy2(fname, out / fname.name)

    # --- Copy encoder base if provided ---
    if encoder_base_path is not None:
        shutil.copy2(encoder_base_path, out / "encoder_base.bin")

    # --- Specimens SQLite DB ---
    create_specimens_db(df, str(out / "specimens.db"))

    # --- Thumbnails (skipped if image_dir is None) ---
    if image_dir is not None:
        ids = df["occurrence_id"].tolist() if "occurrence_id" in df.columns else []
        generate_thumbnails(image_dir, ids, str(out / "thumbnails"))

    # --- OpenTree subtree ---
    shutil.copy2(opentree_subtree_json, out / "opentree_subtree.json")

    # --- Manifest ---
    manifest = {
        "version": _BUNDLE_VERSION,
        "region": region,
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "n_specimens": n_specimens,
        "n_families": len(families),
        "families": families,
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    # --- Size check ---
    total_mb = check_bundle_size(output_dir, warn_threshold_mb=400.0)
    manifest["bundle_size_mb"] = round(total_mb, 2)

    return manifest


def generate_thumbnails(
    image_dir: str,
    specimen_ids: list[str],
    output_dir: str,
    size: tuple[int, int] = (128, 128),
) -> None:
    """Resize specimen images to thumbnails for the bundle's `thumbnails/` directory.

    Tries extensions .jpg, .jpeg, .png for each specimen_id. Skips silently
    if no matching file is found.
    """
    from PIL import Image as PILImage

    src_root = Path(image_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    for sid in specimen_ids:
        src_path = None
        for ext in _EXTS:
            p = src_root / f"{sid}{ext}"
            if p.exists():
                src_path = p
                break
        if src_path is None:
            continue
        img = PILImage.open(src_path).convert("RGB").resize(size, PILImage.LANCZOS)
        img.save(out_root / f"{sid}.jpg", "JPEG", quality=85)


def create_specimens_db(df, output_db_path: str) -> None:
    """Write specimen metadata to SQLite.

    Columns written (subset available in df):
        occurrence_id, scientific_name, ott_id, family, genus, species,
        latitude, longitude, state_province, event_date, institution, image_url.
    """
    import pandas as pd

    # Only write columns that are present in the dataframe
    cols = [c for c in _DB_COLUMNS if c in df.columns]
    out_df = df[cols].copy()

    # Convert non-serialisable types to strings for SQLite compatibility
    for col in out_df.select_dtypes(include=["object", "str"]).columns:
        out_df[col] = out_df[col].where(out_df[col].notna(), other=None)

    # Coerce event_date to string (avoids year=0 issues with pandas→SQLite datetime)
    if "event_date" in out_df.columns:
        out_df["event_date"] = out_df["event_date"].astype(str).where(
            out_df["event_date"].notna(), other=None
        )

    conn = sqlite3.connect(output_db_path)
    try:
        out_df.to_sql("specimens", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_occurrence_id ON specimens(occurrence_id)"
        )
        conn.commit()
    finally:
        conn.close()


def load_bundle(bundle_dir: str):
    """Load a packed regional bundle from disk and return a Bundle for retrieval.

    Reads:
        faiss_global.bin          → global_index
        faiss_families/*.bin      → family_indexes (filename stem = family name)
        specimens.db              → specimen_ids (rowid order), specimens_metadata,
                                    family_specimen_ids (reconstructed from rowid order)
        opentree_subtree.json     → opentree_subtree

    The family_specimen_ids order is reconstructed from specimens.db rowid order,
    which matches the order embeddings were added during index construction.

    Returns:
        retrieval.search.Bundle ready for retrieve().
    """
    import faiss

    from src.retrieval.search import Bundle

    bundle_path = Path(bundle_dir)

    # --- Global index ---
    global_index = faiss.read_index(str(bundle_path / "faiss_global.bin"))

    # --- Family sub-indexes ---
    family_indexes: dict[str, object] = {}
    families_dir = bundle_path / "faiss_families"
    if families_dir.exists():
        for index_file in sorted(families_dir.glob("*.bin")):
            family_name = index_file.stem
            family_indexes[family_name] = faiss.read_index(str(index_file))

    # --- Specimens metadata (from SQLite, ordered by rowid) ---
    db_path = bundle_path / "specimens.db"
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT occurrence_id, family, genus, scientific_name FROM specimens ORDER BY rowid"
        ).fetchall()

    specimen_ids: list[str] = [row[0] for row in rows]
    specimens_metadata: dict[str, dict] = {
        row[0]: {
            "family": row[1] or "",
            "genus":  row[2] or "",
            "taxon":  row[3] or "",
        }
        for row in rows
    }

    # Reconstruct family_specimen_ids in rowid order (matches index construction order)
    family_specimen_ids: dict[str, list[str]] = {}
    for occ_id, family, *_ in rows:
        if family:
            family_specimen_ids.setdefault(family, []).append(occ_id)

    # --- OpenTree subtree ---
    subtree_path = bundle_path / "opentree_subtree.json"
    if subtree_path.exists():
        with open(subtree_path, encoding="utf-8") as f:
            opentree_subtree = json.load(f)
    else:
        opentree_subtree = {}

    return Bundle(
        global_index=global_index,
        family_indexes=family_indexes,
        specimen_ids=specimen_ids,
        family_specimen_ids=family_specimen_ids,
        opentree_subtree=opentree_subtree,
        specimens_metadata=specimens_metadata,
    )


def check_bundle_size(bundle_dir: str, warn_threshold_mb: float = 400.0) -> float:
    """Compute total bundle size in MB. Warns if > warn_threshold_mb.

    Returns:
        Total size in MB (float).
    """
    total_bytes = sum(
        f.stat().st_size
        for f in Path(bundle_dir).rglob("*")
        if f.is_file()
    )
    total_mb = total_bytes / (1024 * 1024)
    if total_mb > warn_threshold_mb:
        warnings.warn(
            f"Bundle size {total_mb:.1f}MB exceeds {warn_threshold_mb}MB threshold "
            f"(DECISION-1 review: consider BioCLIP-1 backbone)",
            stacklevel=2,
        )
    return total_mb
