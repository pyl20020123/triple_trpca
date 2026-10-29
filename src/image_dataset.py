"""DIV2K RGB patch loading, augmentation, and configurable corruption."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DIV2KPatchDataset(Dataset):
    """Load lossless RGB patches listed by ``prepare_div2k.py``.

    Noise is generated after loading so clean patches remain reusable. Training
    noise changes deterministically by epoch; validation and test remain fixed.
    """

    def __init__(self, manifest: Union[str, Path], rank: int,
                 noise: dict, dtype: torch.dtype = torch.float32,
                 augment: bool = False, base_seed: int = 2026,
                 max_samples: Optional[int] = None, init_seed: int = 17):
        self.manifest_path = Path(manifest)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"DIV2K manifest not found: {self.manifest_path}")
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.records = list(csv.DictReader(handle))
        if max_samples is not None:
            self.records = self.records[:int(max_samples)]
        if not self.records:
            raise ValueError(f"manifest contains no patches: {self.manifest_path}")
        self.rank = int(rank)
        self.noise = dict(noise)
        self.dtype = dtype
        self.augment = bool(augment)
        self.base_seed = int(base_seed)
        self.init_seed = int(init_seed)
        self.epoch = 0
        self.split = self.records[0].get("split", "unknown")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        epoch = self.epoch if self.augment else 0
        return np.random.default_rng(self.base_seed + epoch * 10_000_019 + index * 104729)

    def _path(self, record: dict) -> Path:
        path = Path(record["path"])
        return path if path.is_absolute() else self.manifest_path.parent.parent / path

    def _noise_level(self, index: int, rng: np.random.Generator) -> tuple[float, float]:
        rate_values = self.noise.get("rates")
        sigma_values = self.noise.get("sigmas")
        if self.augment:
            lo, hi = self.noise.get("rate_range", [0.01, 0.10])
            rate = float(rng.uniform(lo, hi))
            sigma_lo, sigma_hi = self.noise.get("sigma_range", [0.0, 0.0])
            sigma = float(rng.uniform(sigma_lo, sigma_hi))
        else:
            rates = rate_values or [self.noise.get("rate", 0.05)]
            sigmas = sigma_values or [self.noise.get("sigma", 0.0)]
            rate = float(rates[index % len(rates)])
            sigma = float(sigmas[index % len(sigmas)])
        return rate, sigma

    def _augment(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augment:
            return clean
        if rng.random() < 0.5:
            clean = np.flip(clean, axis=1)
        if rng.random() < 0.5:
            clean = np.flip(clean, axis=0)
        clean = np.rot90(clean, int(rng.integers(0, 4)), axes=(0, 1))
        return np.ascontiguousarray(clean)

    def _corrupt(self, clean: np.ndarray, rate: float, sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
        kind = self.noise.get("type", "impulse")
        observed = clean.copy()
        if kind in ("impulse", "mixed") and rate > 0:
            height, width, _ = clean.shape
            count = round(rate * height * width)
            locations = rng.choice(height * width, size=count, replace=False)
            if self.noise.get("value", "salt_pepper") == "random_rgb":
                values = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
            else:
                # Pixel-aligned salt/pepper corruption preserves RGB semantics.
                values = rng.integers(0, 2, size=(count, 1)).astype(np.float32)
                values = np.repeat(values, 3, axis=1)
            observed.reshape(-1, 3)[locations] = values
        if kind in ("gaussian", "mixed") and sigma > 0:
            observed = observed + rng.normal(0.0, sigma, observed.shape).astype(np.float32)
        return np.clip(observed, 0.0, 1.0)

    def metadata(self, index: int) -> dict:
        record = self.records[index]
        rng = self._rng(index)
        rate, sigma = self._noise_level(index, rng)
        return self._metadata(index, rate, sigma)

    def _metadata(self, index: int, rate: float, sigma: float) -> dict:
        record = self.records[index]
        return {"case": record.get("patch_id", Path(record["path"]).stem),
                "image_id": record.get("image_id", ""), "rank": self.rank,
                "rate": rate, "sigma": sigma, "noise_type": self.noise.get("type", "impulse"),
                "data_seed": self.base_seed + index * 104729, "init_seed": self.init_seed}

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        path = self._path(self.records[index])
        with Image.open(path) as image:
            clean = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        clean = self._augment(clean, rng)
        rate, sigma = self._noise_level(index, rng)
        observed = self._corrupt(clean, rate, sigma, rng)
        low = torch.from_numpy(clean).to(self.dtype)
        noisy = torch.from_numpy(observed).to(self.dtype)
        sparse = noisy - low
        return {"Y": noisy, "L_true": low, "S_true": sparse,
                **self._metadata(index, rate, sigma)}

    def write_manifest(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(self.records[0]) + ["noise_type", "init_seed"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow({**record, "noise_type": self.noise.get("type", "impulse"),
                                 "init_seed": self.init_seed})

