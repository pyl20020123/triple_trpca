"""Leakage-safe deterministic synthetic data and split manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .numpy_core import generate_case


def split_seeds(train_count: int, val_count: int, test_count: int) -> dict[str, list[int]]:
    splits = {
        "train": list(range(1_000_000, 1_000_000 + train_count)),
        "val": list(range(2_000_000, 2_000_000 + val_count)),
        "test": list(range(3_000_000, 3_000_000 + test_count)),
    }
    assert_disjoint_splits(splits)
    return splits


def assert_disjoint_splits(splits: dict[str, Sequence[int]]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f"data split leakage between {left} and {right}: {sorted(overlap)[:3]}")


class SyntheticTRPCADataset(Dataset):
    def __init__(self, seeds: Iterable[int], shape: Sequence[int] = (20, 20, 16),
                 rank: int = 2, rates: Sequence[float] = (0.01, 0.05, 0.10),
                 amplitudes: Sequence[float] = (3.0,), dtype: torch.dtype = torch.float32,
                 init_seed: int = 17):
        self.seeds = [int(x) for x in seeds]
        self.shape = tuple(int(x) for x in shape)
        self.rank = int(rank)
        self.rates = tuple(float(x) for x in rates)
        self.amplitudes = tuple(float(x) for x in amplitudes)
        self.dtype = dtype
        self.init_seed = int(init_seed)
        if not self.seeds or not self.rates or not self.amplitudes:
            raise ValueError("seeds, rates, and amplitudes must be non-empty")

    def __len__(self) -> int:
        return len(self.seeds)

    def metadata(self, index: int) -> dict:
        seed = self.seeds[index]
        # Conditions are deterministic and balanced without sharing the data RNG.
        rate = self.rates[index % len(self.rates)]
        amplitude = self.amplitudes[(index // len(self.rates)) % len(self.amplitudes)]
        return {"case": f"r{self.rank}_rho{rate:g}_seed{seed}", "rank": self.rank,
                "rate": rate, "amplitude": amplitude, "data_seed": seed,
                "init_seed": self.init_seed}

    def __getitem__(self, index: int) -> dict:
        meta = self.metadata(index)
        low, sparse, observed = generate_case(
            self.shape, self.rank, meta["rate"], meta["amplitude"], meta["data_seed"])
        low_tensor = torch.as_tensor(low, dtype=self.dtype)
        sparse_tensor = torch.as_tensor(sparse, dtype=self.dtype)
        return {
            # Re-add after dtype conversion so the identity is exact in the
            # dtype actually consumed by training.
            "Y": low_tensor + sparse_tensor,
            "L_true": low_tensor,
            "S_true": sparse_tensor,
            **meta,
        }

    def write_manifest(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["case", "rank", "rate", "amplitude", "data_seed", "init_seed", "shape"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(len(self)):
                row = self.metadata(i)
                row["shape"] = "x".join(map(str, self.shape))
                writer.writerow(row)

