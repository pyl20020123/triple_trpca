#!/usr/bin/env python3
"""Compare ordinary MALS, V1, fixed-stage V2, and a learned checkpoint."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.io_utils import load_config, resolve_device, resolve_dtype
from src.metrics import batch_metrics
from src.numpy_core import fit_trpca
from src.synthetic_dataset import SyntheticTRPCADataset, split_seeds
from src.unfold_net import TriTDHalfUnfold
from src.tritd_torch import input_only_init
from train import build_model


def tensor_metrics(item: dict, low, sparse, threshold: float) -> dict:
    values = batch_metrics(item["L_true"][None], item["S_true"][None],
                           low[None], sparse[None], threshold)
    return {key: float(value[0]) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device, dtype = resolve_device(config.get("device", "auto")), resolve_dtype(config.get("dtype", "float32"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    learned = build_model(checkpoint["config"]).to(device=device, dtype=dtype)
    learned.load_state_dict(checkpoint["model"])
    learned.eval()
    model_cfg = checkpoint["config"]["model"]
    fixed = TriTDHalfUnfold(rank=config["data"]["rank"], stages=model_cfg["stages"],
                            warmup_sweeps=model_cfg.get("warmup_sweeps", 2),
                            lambda_init=model_cfg.get("lambda_init", 0.1),
                            eta_init=model_cfg.get("eta_init", 1e-3),
                            eta_s_init=model_cfg.get("eta_s_init", 1e-3),
                            alpha=model_cfg.get("alpha", 1e-4), learnable=False).to(device=device, dtype=dtype)
    fixed.eval()
    data = config["data"]
    count = int(data.get("comparison_count", data["test_count"]))
    seeds = split_seeds(0, 0, count)["test"]
    dataset = SyntheticTRPCADataset(seeds, data["shape"], data["rank"], data["rates"],
                                    data.get("amplitudes", [3.0]), dtype, learned.init_seed)
    threshold = config["evaluation"].get("support_threshold", 0.05)
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        meta = dataset.metadata(index)
        observed = item["Y"].to(device=device, dtype=dtype)[None]
        for name, model in (("B2_fixed_unfold", fixed), ("B4_learned_unfold", learned)):
            started = time.perf_counter()
            with torch.no_grad():
                result = model(observed)
            elapsed = time.perf_counter() - started
            metric_item = {key: value.cpu() if torch.is_tensor(value) else value for key, value in item.items()}
            metrics = tensor_metrics(metric_item, result["L"][0].cpu(), result["S"][0].cpu(), threshold)
            rows.append({**meta, "method": name, "seconds": elapsed,
                         "sweeps": model.warmup_sweeps + model.num_stages, **metrics})
        y_np = item["Y"].numpy()
        sweeps = fixed.warmup_sweeps + fixed.num_stages
        # Align the initial random factors across NumPy and Torch controls.
        aligned = input_only_init(observed, data["rank"], learned.init_seed)
        aligned_numpy = tuple(factor[0].detach().cpu().double().numpy() for factor in aligned)
        for name, freeze in (("B0_plain_mals", True), ("B1_v1_tritd_half", False)):
            started = time.perf_counter()
            result = fit_trpca(y_np, data["rank"], lam=model_cfg.get("lambda_init", 0.1),
                               alpha=model_cfg.get("alpha", 1e-4), eta=model_cfg.get("eta_init", 1e-3),
                               max_iter=sweeps, seed=meta["init_seed"], freeze_sparse=freeze,
                               initial_factors=aligned_numpy)
            elapsed = time.perf_counter() - started
            metrics = tensor_metrics(item, torch.from_numpy(result["L"]), torch.from_numpy(result["S"]), threshold)
            rows.append({**meta, "method": name, "seconds": elapsed, "sweeps": sweeps, **metrics})
    fields = ["case", "rank", "rate", "amplitude", "data_seed", "method", "sweeps", "seconds",
              "rre_l", "rre_s", "precision", "recall", "f1", "residual_rre"]
    with (output / "comparison_by_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} comparison rows to {output}")


if __name__ == "__main__":
    main()
