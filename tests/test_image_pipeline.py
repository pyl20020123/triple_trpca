import csv

import numpy as np
import torch
from PIL import Image

from denoise_image import tiled_denoise
from prepare_div2k import process_images, write_manifest
from src.image_dataset import DIV2KPatchDataset


def test_div2k_preprocess_and_rgb_noise(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    rng = np.random.default_rng(5)
    image_path = raw / "0001.png"
    Image.fromarray(rng.integers(0, 256, (12, 14, 3), dtype=np.uint8), "RGB").save(image_path)
    records = process_images([image_path], "train", processed, patch=8, stride=6, max_per_image=3)
    write_manifest(processed, "train", records)
    assert len(records) == 3
    dataset = DIV2KPatchDataset(processed / "manifests/train.csv", rank=3,
                                noise={"type": "impulse", "rates": [0.10],
                                       "rate_range": [0.10, 0.10], "sigmas": [0.0]},
                                dtype=torch.float32, augment=False)
    item = dataset[0]
    assert item["Y"].shape == (8, 8, 3)
    torch.testing.assert_close(item["Y"], item["L_true"] + item["S_true"], rtol=0, atol=0)
    changed_pixels = (item["S_true"].abs().sum(dim=2) > 0).sum()
    assert 0 < changed_pixels <= round(0.10 * 8 * 8)


class IdentityModel(torch.nn.Module):
    def forward(self, value):
        return {"L": value, "S": torch.zeros_like(value)}


def test_overlap_tiled_inference_preserves_identity():
    rng = np.random.default_rng(9)
    image = rng.random((11, 13, 3), dtype=np.float32)
    low, sparse = tiled_denoise(IdentityModel(), image, tile_size=8, overlap=3, batch_size=2)
    np.testing.assert_allclose(low, image, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(sparse, np.zeros_like(image))
