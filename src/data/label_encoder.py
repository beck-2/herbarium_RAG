"""
Integer label encoding for hierarchical taxonomy labels.

Builds sorted, stable string→int mappings for family, genus, and species
from a DataFrame and saves/loads them as JSON for use at training and inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class LabelEncoder:
    """Stable string→int mapping with an explicit <UNK> token for unseen labels."""

    UNK = "<UNK>"

    def __init__(self) -> None:
        self._label2idx: dict[str, int] = {}
        self._idx2label: list[str] = []

    def fit(self, values: pd.Series) -> "LabelEncoder":
        """Fit on a pandas Series of string values. Sorts alphabetically for stability."""
        unique = sorted(str(v) for v in values.dropna().unique())
        self._idx2label = [self.UNK] + unique
        self._label2idx = {lab: i for i, lab in enumerate(self._idx2label)}
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        """Map a Series of strings to integer indices. Unknown values → 0."""
        return values.map(lambda v: self._label2idx.get(str(v) if pd.notna(v) else self.UNK, 0))

    def __len__(self) -> int:
        return len(self._idx2label)

    def decode(self, idx: int) -> str:
        return self._idx2label[idx] if 0 <= idx < len(self._idx2label) else self.UNK

    def to_dict(self) -> dict:
        return {"label2idx": self._label2idx, "idx2label": self._idx2label}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelEncoder":
        enc = cls()
        enc._label2idx = d["label2idx"]
        enc._idx2label = d["idx2label"]
        return enc


def build_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Build family, genus, and species encoders from a specimens DataFrame."""
    encoders: dict[str, LabelEncoder] = {}
    for col in ("family", "genus", "scientific_name"):
        enc = LabelEncoder()
        enc.fit(df[col] if col in df.columns else pd.Series(dtype=str))
        encoders[col] = enc
    return encoders


def save_label_encoders(encoders: dict[str, LabelEncoder], path: str) -> None:
    """Serialize all encoders to a single JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {name: enc.to_dict() for name, enc in encoders.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_label_encoders(path: str) -> dict[str, LabelEncoder]:
    """Load encoders from JSON. Returns dict keyed by column name."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {name: LabelEncoder.from_dict(d) for name, d in payload.items()}
