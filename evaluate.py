#!/usr/bin/env python3
"""Freeze a checkpoint and evaluate it on the fixed synthetic test split."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.io_utils import load_config, resolve_device, resolve_dtype
from src.metrics import batch_metrics
from src.synthetic_dataset import SyntheticTRPCADataset, split_seeds
from src.unfold_net import TriTDHalfUnfold
from train import build_model


METRIC_FIELDS = ["case", "rank", "rate", "amplitude", "data_seed", "init_seed",
                 "rre_l", "rre_s", "precision", "recall", "f1", "residual_rre",
                 "seconds", "max_cuda_memory"]


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recovery_panel(path: Path, observed: np.ndarray, low_true: np.ndarray,
                   sparse_true: np.ndarray, low_hat: np.ndarray, sparse_hat: np.ndarray) -> None:
    k = observed.shape[2] // 2
    arrays = [low_true[:, :, k], observed[:, :, k], low_hat[:, :, k],
              sparse_true[:, :, k], sparse_hat[:, :, k], np.abs(low_hat - low_true)[:, :, k]]
    titles = ["L true", "Y", "L estimate", "S true", "S estimate", "|L error|"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for axis, array, title in zip(axes.flat, arrays, titles):
        image = axis.imshow(array, cmap="viridis")
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    eval_config = load_config(args.config)
    output = Path(args.out)
    panels = output / "recovery_panels"
    panels.mkdir(parents=True, exist_ok=True)
    device = resolve_device(eval_config.get("device", "auto"))
    dtype = resolve_dtype(eval_config.get("dtype", "float32"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_config = checkpoint["config"]
    model = build_model(train_config).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = eval_config["data"]
    seeds = split_seeds(0, 0, int(data["test_count"]))["test"]
    dataset = SyntheticTRPCADataset(seeds, data["shape"], data["rank"], data["rates"],
                                    data.get("amplitudes", [3.0]), dtype, model.init_seed)
    dataset.write_manifest(output / "data_manifest_test.csv")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    threshold = eval_config["evaluation"].get("support_threshold", 0.05)
    rows, stage_rows = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            observed = batch["Y"].to(device=device, dtype=dtype)
            low_true = batch["L_true"].to(device=device, dtype=dtype)
            sparse_true = batch["S_true"].to(device=device, dtype=dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model(observed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            values = batch_metrics(low_true, sparse_true, result["L"], result["S"], threshold)
            meta = dataset.metadata(index)
            row = {**meta, **{key: float(value[0]) for key, value in values.items()},
                   "seconds": elapsed,
                   "max_cuda_memory": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
            rows.append(row)
            for stage_idx, (low, sparse, active) in enumerate(zip(
                    result["L_stages"], result["S_stages"], result["active_ratios"]), 1):
                stage_value = batch_metrics(low_true, sparse_true, low, sparse, threshold)
                stage_rows.append({"case": meta["case"], "stage": stage_idx,
                                   "rre_l": float(stage_value["rre_l"][0]),
                                   "rre_s": float(stage_value["rre_s"][0]),
                                   "f1": float(stage_value["f1"][0]),
                                   "active_ratio": float(active)})
            recovery_panel(panels / f"{meta['case']}.png", observed[0].cpu().numpy(),
                           low_true[0].cpu().numpy(), sparse_true[0].cpu().numpy(),
                           result["L"][0].cpu().numpy(), result["S"][0].cpu().numpy())

    write_rows(output / "metrics_by_case.csv", rows, METRIC_FIELDS)
    write_rows(output / "stage_metrics.csv", stage_rows,
               ["case", "stage", "rre_l", "rre_s", "f1", "active_ratio"])
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["rank"], row["rate"], row["amplitude"])].append(row)
    summaries = []
    for (rank, rate, amplitude), group in sorted(grouped.items()):
        summary = {"rank": rank, "rate": rate, "amplitude": amplitude, "n_cases": len(group)}
        for key in ("rre_l", "rre_s", "precision", "recall", "f1", "residual_rre", "seconds"):
            values = np.asarray([item[key] for item in group])
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary["success_rate"] = float(np.mean([
            item["rre_l"] < 1e-2 and item["f1"] >= 0.95 for item in group]))
        summaries.append(summary)
    summary_fields = list(summaries[0]) if summaries else ["rank", "rate", "amplitude", "n_cases"]
    write_rows(output / "summary_by_condition.csv", summaries, summary_fields)
    failures = [row for row in rows if not (row["rre_l"] < 1e-2 and row["f1"] >= 0.95)]
    write_rows(output / "failure_cases.csv", failures, METRIC_FIELDS)
    runtime = [{"device": str(device), "dtype": str(dtype), "samples": len(rows),
                "time_per_sample": float(np.mean([row["seconds"] for row in rows])) if rows else 0.0,
                "peak_memory": max((row["max_cuda_memory"] for row in rows), default=0),
                "iters_equiv": model.warmup_sweeps + model.num_stages}]
    write_rows(output / "runtime_memory.csv", runtime, list(runtime[0]))
    print(f"evaluated {len(rows)} cases; results written to {output}")


if __name__ == "__main__":
    main()

