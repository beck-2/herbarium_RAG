"""
NAFlora-1M metadata and Symbiota DwCA download utilities.

Responsibilities:
- NAFlora-1M metadata: JSON from Kaggle (herbarium-2022-fgvc9). Use existing file
  in output_dir, or Kaggle API if configured, or user places train_metadata.json manually.
- DwCA: download from Symbiota portal export URL when available; otherwise use
  manually extracted DwCA path.
- Progress via tqdm where applicable. No LLM calls.

Phase 1 priority: metadata only; images deferred until Phase 4 (training).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


def download_naflora_metadata(output_dir: str) -> str:
    """Obtain NAFlora-1M metadata in output_dir. Returns path to dir containing JSON.

    Prefer: (1) existing train_metadata.json in output_dir, (2) Kaggle API download
    of herbarium-2022-fgvc9 train_metadata.json. If neither, raises with instructions
    to download from Kaggle and place train_metadata.json in output_dir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = output_path / "train_metadata.json"
    if existing.exists():
        return str(output_path)

    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError:
        raise FileNotFoundError(
            f"NAFlora-1M metadata not found at {existing}. "
            "Download train_metadata.json from Kaggle competition herbarium-2022-fgvc9 "
            "(https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data) and place it in "
            f"{output_path}. Alternatively install kaggle (pip install kaggle) and configure "
            "API credentials, then re-run."
        ) from None

    api = kaggle.KaggleApi()
    api.authenticate()
    api.competition_download_file(
        "herbarium-2022-fgvc9",
        "train_metadata.json",
        path=str(output_path),
    )
    return str(output_path)


def download_symbiota_dwca(region: str, portal_url: str | None, output_dir: str) -> str:
    """Download or prepare DwCA for a Symbiota portal. Returns path to DwCA dir.

    If portal_url is None (e.g. aggregated Midwest), raises with instructions to
    provide DwCA manually. Otherwise tries common Symbiota DwCA export URL
    (portal/content/dwca/...) or downloads a .zip and extracts to output_dir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not portal_url:
        raise ValueError(
            f"Region {region!r} has no symbiota_portal URL. "
            "Download the DwCA from the portal manually and extract it to "
            f"{output_path} (or set config regions.{region}.dwca_path to that path)."
        )

    base = portal_url.rstrip("/")
    for suffix in ("/content/dwca/dwca.zip", "/content/dwca/dwca", "/content/dwca/"):
        url = base + suffix
        try:
            zip_path = output_path / "dwca.zip"
            with tqdm(unit="B", unit_scale=True, desc=f"Downloading DwCA {region}") as pbar:

                def _report(blocks, block_size, total):
                    if total and pbar.total is None:
                        pbar.total = total
                    pbar.update(blocks * block_size - pbar.n)

                urlretrieve(url, zip_path, reporthook=_report)
            if zip_path.exists() and zip_path.stat().st_size > 0:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(output_path)
                zip_path.unlink()
                return str(output_path)
        except Exception:
            continue
    raise RuntimeError(
        f"Could not download DwCA from {portal_url}. "
        f"Download the DwCA manually from the portal and extract to {output_path}."
    )


def download_naflora_images(
    csv_path: str,
    output_dir: str,
    occurrence_ids: list[str] | None = None,
) -> None:
    """Download specimen images from NAFlora-1M image URLs.

    Deferred until Phase 4 (training). NAFlora-1M images are provided via Kaggle
    (train_images/ and test_images/); use Kaggle API or manual download.

    Args:
        csv_path: Path to NAFlora metadata JSON or directory (for compatibility).
        output_dir: Directory to save images.
        occurrence_ids: Optional subset of occurrenceIDs to download. Downloads all if None.
    """
    raise NotImplementedError(
        "Image download deferred to Phase 4. Use Kaggle competition herbarium-2022-fgvc9 "
        "train_images / test_images, or Kaggle API."
    )
