"""
Tests for streaming image download from CCH2 media server.

Run locally:   python3.14 -m pytest tests/test_streaming.py -m network -v
Run in Colab:  !python -m pytest tests/test_streaming.py -m network -v

These tests hit real URLs — they tell you:
  - Whether media01.symbiota.org is reachable from the current machine
  - Whether the StreamingSpecimenDataset can load a real image end-to-end
  - Whether Colab IPs are blocked (run in Colab to check)
"""

import pytest

# A real CCH2 image URL sampled from the CA parquet
SAMPLE_URL = "https://media01.symbiota.org/media/cch2/DAV/DAV332/DAV332770_lg.jpg"
# A second one for redundancy
SAMPLE_URL_2 = "https://media01.symbiota.org/media/cch2/RSA_VascularPlants/RSA0061/RSA0061160_lg.jpg"


@pytest.mark.network
def test_symbiota_media_reachable():
    """media01.symbiota.org responds to a HEAD request."""
    import urllib.request
    req = urllib.request.Request(SAMPLE_URL, method="HEAD",
                                 headers={"User-Agent": "HyperbolicHerbarium/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"


@pytest.mark.network
def test_streaming_dataset_downloads_real_image():
    """StreamingSpecimenDataset can download and decode a real CCH2 image."""
    import torch
    from src.data.dataset import StreamingSpecimenDataset

    records = [{"image_url": SAMPLE_URL, "family_idx": 0, "genus_idx": 0, "species_idx": 0}]
    ds = StreamingSpecimenDataset(records, transform=None)
    result = ds[0]

    assert result is not None, (
        f"Image download returned None — {SAMPLE_URL} may be blocked from this machine.\n"
        "If running on Colab, media01.symbiota.org may be blocking Google Cloud IPs."
    )
    img, fam, gen, spe = result
    # Without a transform, img is a PIL Image
    from PIL import Image
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


@pytest.mark.network
def test_streaming_dataset_with_transform():
    """StreamingSpecimenDataset returns a float tensor when a transform is applied."""
    import torch
    from torchvision import transforms
    from src.data.dataset import StreamingSpecimenDataset

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    records = [{"image_url": SAMPLE_URL_2, "family_idx": 1, "genus_idx": 2, "species_idx": 3}]
    ds = StreamingSpecimenDataset(records, transform=tf)
    result = ds[0]

    assert result is not None, "Image download failed — server may be blocking this IP."
    img_tensor, fam, gen, spe = result
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert fam.item() == 1
    assert gen.item() == 2
    assert spe.item() == 3


@pytest.mark.network
def test_bad_url_returns_none():
    """A broken URL returns None (doesn't crash the training loop)."""
    from src.data.dataset import StreamingSpecimenDataset

    records = [{"image_url": "https://media01.symbiota.org/does_not_exist_xyz.jpg",
                "family_idx": 0, "genus_idx": 0, "species_idx": 0}]
    ds = StreamingSpecimenDataset(records, transform=None, max_retries=0)
    result = ds[0]
    assert result is None, "Expected None for a bad URL"
