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

from pathlib import Path

import numpy as np


def encode_specimens(
    specimen_ids: list[str],
    image_dir: str,
    model,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """Encode specimens to 512-d Poincaré points using model.encode().

    model must expose: encode(images: list[PIL.Image]) -> np.ndarray (N, D) float32

    Images are looked up as {image_dir}/{specimen_id}.jpg (then .jpeg, .png).
    Missing files are silently skipped.

    Returns:
        (embeddings, ordered_ids): float32 array (N, D) and corresponding IDs.
    """
    from PIL import Image as PILImage

    image_root = Path(image_dir)
    _EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    def _find_image(sid: str) -> Path | None:
        for ext in _EXTS:
            p = image_root / f"{sid}{ext}"
            if p.exists():
                return p
        return None

    all_embeddings: list[np.ndarray] = []
    valid_ids: list[str] = []

    # Collect in batches
    batch_imgs: list = []
    batch_ids: list[str] = []

    def _flush():
        if not batch_imgs:
            return
        emb = model.encode(batch_imgs)
        all_embeddings.append(np.array(emb, dtype=np.float32))
        valid_ids.extend(batch_ids)
        batch_imgs.clear()
        batch_ids.clear()

    for sid in specimen_ids:
        img_path = _find_image(sid)
        if img_path is None:
            continue
        batch_imgs.append(PILImage.open(img_path).convert("RGB"))
        batch_ids.append(sid)
        if len(batch_imgs) >= batch_size:
            _flush()

    _flush()

    if not all_embeddings:
        return np.empty((0, 512), dtype=np.float32), []

    return np.concatenate(all_embeddings, axis=0), valid_ids


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
        embeddings: array of shape (N, D) — cast to float32 internally.
        n_clusters: Number of IVF clusters (nlist). Use 256 for < 500K vectors.
        n_subquantizers: PQ subquantizers (must divide D evenly).
        bits_per_code: Bits per PQ code (8 = 256 centroids per sub-quantizer).
        nprobe: Cells to visit at query time (higher = better recall, slower).
        use_gpu: Use faiss-gpu for faster construction (requires faiss-gpu install).

    Returns:
        Trained and populated faiss.IndexIVFPQ with nprobe set.
    """
    import faiss

    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    d = emb.shape[1]

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, n_clusters, n_subquantizers, bits_per_code)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(emb)
    index.add(emb)
    index.nprobe = nprobe
    return index


def build_family_subindexes(
    embeddings: np.ndarray,
    family_labels: list[str],
    n_clusters: int = 64,
) -> dict[str, object]:
    """Build one FAISS index per family for targeted re-retrieval.

    Families with fewer vectors than n_clusters use exact IndexFlatL2.
    Larger families use IndexIVFPQ with min(n_clusters, len//2) clusters.

    Returns:
        Dict mapping family name → faiss index (already trained and populated).
    """
    import faiss

    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    d = emb.shape[1]
    labels_arr = np.array(family_labels)
    unique_families = sorted(set(family_labels))

    subindexes: dict[str, object] = {}

    for family in unique_families:
        mask = labels_arr == family
        sub_emb = emb[mask]
        n = len(sub_emb)

        # IVF-PQ with bits_per_code=8 requires ≥ 256 training vectors (2^8 PQ centroids).
        # For small families, exact search is both correct and faster.
        pq_min = 256
        if n < max(n_clusters, pq_min):
            # Too few vectors for IVF-PQ training — use exact search
            idx = faiss.IndexFlatL2(d)
            idx.add(sub_emb)
        else:
            actual_clusters = min(n_clusters, n // 2)
            quantizer = faiss.IndexFlatL2(d)
            idx = faiss.IndexIVFPQ(quantizer, d, actual_clusters, min(4, d // 4), 8)
            idx.train(sub_emb)
            idx.add(sub_emb)
            idx.nprobe = max(1, actual_clusters // 4)

        subindexes[family] = idx

    return subindexes


def verify_recall(
    index,
    probe_embeddings: np.ndarray,
    probe_ids: list[str],
    k: int = 10,
) -> float:
    """Compute recall@k: fraction of probes where the probe itself is in top-k.

    Assumes probe embeddings were indexed at sequential positions 0..N-1
    (i.e. index.add() was called with the same array in the same order).

    Args:
        index: FAISS index containing all probe embeddings.
        probe_embeddings: Embeddings to search (N, D) float32.
        probe_ids: Specimen IDs corresponding to each probe row (unused in
                   the recall calculation but kept for logging/debugging).
        k: Number of top results to check.

    Returns:
        Recall@k as float in [0, 1].
    """
    probes = np.ascontiguousarray(probe_embeddings, dtype=np.float32)
    n = len(probes)
    if n == 0:
        return 0.0

    _, I = index.search(probes, k)  # (N, k) integer indices

    hits = 0
    for i in range(n):
        if i in I[i]:
            hits += 1

    return hits / n


def save_index(index, output_path: str) -> None:
    """Save a FAISS index to disk."""
    import faiss
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    faiss.write_index(index, output_path)


def load_index(index_path: str) -> object:
    """Load a FAISS index from disk."""
    import faiss

    return faiss.read_index(index_path)
