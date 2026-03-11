"""
PyTorch Dataset classes for specimen data.

SyntheticSpecimenDataset: generates random feature vectors + labels.
    Used for smoke tests and CI — no images or files required.

SpecimenDataset: loads real images from disk + parquet metadata.
    Used for actual training once NAFlora/DwCA images are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset


class SyntheticSpecimenDataset(Dataset):
    """Generates random pre-encoded feature vectors with random taxonomy labels.

    Simulates BioCLIP-2 visual embeddings without requiring images or the
    backbone model at dataset time.  Useful for smoke tests, unit tests, and
    verifying the training loop before real data is available.

    Args:
        n_samples:  Number of synthetic specimens.
        embed_dim:  Dimensionality of feature vectors (768 for BioCLIP-2 ViT-L/14).
        n_families: Number of family classes.
        n_genera:   Number of genus classes.
        n_species:  Number of species classes.
        seed:       Random seed for reproducibility (None = random each time).
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        embed_dim: int = 768,
        n_families: int = 8,
        n_genera: int = 20,
        n_species: int = 50,
        seed: int | None = None,
    ):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        self.features = torch.randn(n_samples, embed_dim, generator=rng)
        self.family_labels = torch.randint(0, n_families, (n_samples,), generator=rng)
        self.genus_labels = torch.randint(0, n_genera, (n_samples,), generator=rng)
        self.species_labels = torch.randint(0, n_species, (n_samples,), generator=rng)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (feature_vector, family_label, genus_label, species_label)."""
        return (
            self.features[idx],
            self.family_labels[idx],
            self.genus_labels[idx],
            self.species_labels[idx],
        )


class SpecimenDataset(Dataset):
    """Load real specimen images from disk with taxonomy labels from a parquet file.

    The parquet file must contain CANONICAL_COLUMNS (see src/data/parse.py) plus
    integer label columns: family_idx, genus_idx, species_idx.  These are added
    by the data preparation pipeline (Phase 1 balance + label encoding step).

    Args:
        manifest_path: Path to train.txt / val.txt / test.txt (one occurrence_id per line).
        parquet_path:  Path to specimens.parquet with metadata + label columns.
        image_root:    Root directory containing image files (referenced by image_url).
        transform:     torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        manifest_path: str,
        parquet_path: str,
        image_root: str,
        transform: Callable | None = None,
    ):
        import pandas as pd

        self.image_root = Path(image_root)
        self.transform = transform

        ids = Path(manifest_path).read_text().strip().splitlines()
        df = pd.read_parquet(parquet_path)
        df = df[df["occurrence_id"].isin(set(ids))].reset_index(drop=True)

        self.image_urls = df["image_url"].tolist()
        self.family_labels = torch.tensor(df["family_idx"].tolist(), dtype=torch.long)
        self.genus_labels = torch.tensor(df["genus_idx"].tolist(), dtype=torch.long)
        self.species_labels = torch.tensor(df["species_idx"].tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_urls)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from PIL import Image

        img_path = self.image_root / self.image_urls[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            self.family_labels[idx],
            self.genus_labels[idx],
            self.species_labels[idx],
        )
