"""
FAISS IVF-PQ index construction.

Responsibilities:
- Encode all regional specimens with the trained model to get 512-d Poincaré points
- Train and populate a FAISS IVF-PQ index (global + per-family sub-indexes)
- Use faiss-gpu during training/indexing, faiss-cpu at inference
- Verify index recall@10 ≥ 95% on a held-out probe set before saving

From SPEC §7.1:
    index = faiss.IndexIVFPQ(
        quantizer, d=512,
        n_clusters=256,    # 256 for <500K vectors
        n_subquantizers=32,
        bits_per_code=8
    )
    index.nprobe = 32
    # Expected size: ~6MB per 50K vectors
    # Expected recall@10: 95-98%
"""

from __future__ import annotations

import numpy as np

# TODO(phase5): implement encode_specimens (batch encoding with trained model)
# TODO(phase5): implement build_ivfpq_index (train + add vectors)
# TODO(phase5): implement build_family_subindexes (one index per family)
# TODO(phase5): implement recall@10 verification


def encode_specimens(
    specimen_ids: list[str],
    image_dir: str,
    model,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """Encode all specimens to 512-d Poincaré points.

    Returns:
        (embeddings, ordered_ids): float32 array (N, 512) and corresponding IDs.
    """
    raise NotImplementedError


def build_ivfpq_index(
    embeddings: np.ndarray,
    n_clusters: int = 256,
    n_subquantizers: int = 32,
    bits_per_code: int = 8,
    nprobe: int = 32,
    use_gpu: bool = False,
) -> object:
    """Train and populate a FAISS IVF-PQ index.

    Args:
        embeddings: float32 array of shape (N, D).
        n_clusters: Number of IVF clusters (256 for < 500K vectors).
        n_subquantizers: PQ subquantizers (32 default).
        bits_per_code: Bits per PQ code (8 default).
        nprobe: Cells to visit at query time.
        use_gpu: Use faiss-gpu for faster construction.

    Returns:
        Trained and populated faiss.IndexIVFPQ.
    """
    raise NotImplementedError


def build_family_subindexes(
    embeddings: np.ndarray,
    family_labels: list[str],
    n_clusters: int = 64,
) -> dict[str, object]:
    """Build one small IVF-PQ index per family for re-retrieval.

    Uses exact search (IndexFlatL2) for families with < n_clusters vectors.

    Returns:
        Dict mapping family name → faiss index.
    """
    raise NotImplementedError


def verify_recall(index, probe_embeddings: np.ndarray, probe_ids: list[str], k: int = 10) -> float:
    """Compute recall@k: fraction of probes where true ID is in top-k results.

    Args:
        index: Trained FAISS index containing all specimen embeddings.
        probe_embeddings: Embeddings of probe specimens (must be in index).
        probe_ids: IDs of probe specimens.
        k: Number of top results to check.

    Returns:
        Recall@k as a float in [0, 1].
    """
    raise NotImplementedError


def save_index(index, output_path: str) -> None:
    """Save a FAISS index to disk."""
    raise NotImplementedError


def load_index(index_path: str) -> object:
    """Load a FAISS index from disk."""
    raise NotImplementedError
