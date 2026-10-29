#!/usr/bin/env python3
"""Train the stage-wise supervised TriTD-Half unfolding network."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.io_utils import (append_csv, environment_info, load_config, resolve_device,
                          resolve_dtype, restore_rng_state, rng_state, save_yaml,
                          seed_everything, write_json)
from src.synthetic_dataset import SyntheticTRPCADataset, split_seeds
from src.trainer import evaluate_loader, train_one_epoch
from src.unfold_net import TriTDHalfUnfold


LOG_FIELDS = ["epoch", "lr", "loss_total", "loss_rec", "loss_sparse", "rre_l_mean",
              "rre_s_mean", "f1_mean", "active_ratio_mean", "grad_norm", "seconds"]


def build_model(config: dict) -> TriTDHalfUnfold:
    model = config["model"]
    return TriTDHalfUnfold(
        rank=config["data"]["rank"], stages=model["stages"],
        warmup_sweeps=model.get("warmup_sweeps", 2),
        lambda_init=model.get("lambda_init", 0.1), eta_init=model.get("eta_init", 1e-3),
        eta_s_init=model.get("eta_s_init", 1e-3), alpha=model.get("alpha", 1e-4),
        positivity_eps=model.get("positivity_eps", 1e-8), init_seed=model.get("init_seed", 17),
    )


def make_datasets(config: dict, dtype: torch.dtype, output: Path):
    data = config["data"]
    seeds = split_seeds(data["train_count"], data["val_count"], data.get("test_count", 0))
    common = dict(shape=data["shape"], rank=data["rank"], rates=data["rates"],
                  amplitudes=data.get("amplitudes", [3.0]), dtype=dtype,
                  init_seed=config["model"].get("init_seed", 17))
    train_set = SyntheticTRPCADataset(seeds["train"], **common)
    val_set = SyntheticTRPCADataset(seeds["val"], **common)
    train_set.write_manifest(output / "data_manifest_train.csv")
    val_set.write_manifest(output / "data_manifest_val.csv")
    return train_set, val_set


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_metric: float, config: dict) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_metric": best_metric, "config": config, "rng_state": rng_state()}, path)


def plot_training_curves(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for filename, label in (("train.csv", "train"), ("val.csv", "val")):
        path = output / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["loss_total"]) for row in rows], label=label)
        axes[1].plot(epochs, [float(row["rre_l_mean"]) for row in rows], label=label)
    axes[0].set(title="Supervised loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Low-rank RRE", xlabel="epoch", ylabel="RRE_L")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output / "train_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--out", required=True, help="new or resumable output directory")
    parser.add_argument("--resume", help="checkpoint to resume (typically last.pt)")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.out)
    if not args.resume and (output / "train.csv").exists():
        raise FileExistsError(f"{output} already contains a run; use a new --out or --resume")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output / "config_resolved.yaml")
    write_json(environment_info(), output / "env.json")

    seed_everything(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dtype = resolve_dtype(config.get("dtype", "float32"))
    train_set, val_set = make_datasets(config, dtype, output)
    batch_size = int(config["training"]["batch_size"])
    loader_generator = torch.Generator().manual_seed(int(config.get("seed", 2026)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=loader_generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=float(config["training"].get("lr_factor", 0.5)),
        patience=int(config["training"].get("lr_patience", 5)))
    start_epoch, best_metric = 1, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])

    stale = 0
    epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(train_loader, optimizer, model, config, device, dtype)
        val_metrics = evaluate_loader(val_loader, model, config, device, dtype)
        lr = optimizer.param_groups[0]["lr"]
        train_row = {"epoch": epoch, "lr": lr, **train_metrics}
        val_row = {"epoch": epoch, "lr": lr, **val_metrics}
        append_csv(output / "train.csv", train_row, LOG_FIELDS)
        append_csv(output / "val.csv", val_row, LOG_FIELDS)
        for params, active in zip(model.stage_parameters(), val_metrics["active_ratios_by_stage"]):
            append_csv(output / "stage_params.csv", {"epoch": epoch, **params, "active_ratio": active},
                       ["epoch", "stage", "lambda", "eta", "eta_s", "alpha", "active_ratio"])
        selected = val_metrics["rre_l_mean"]
        scheduler.step(selected)
        if selected < best_metric:
            best_metric, stale = selected, 0
            save_checkpoint(checkpoints / "best.pt", model, optimizer, scheduler, epoch, best_metric, config)
        else:
            stale += 1
        save_checkpoint(checkpoints / "last.pt", model, optimizer, scheduler, epoch, best_metric, config)
        print(f"epoch={epoch} train_loss={train_metrics['loss_total']:.6g} "
              f"val_rre_l={selected:.6g} active={val_metrics['active_ratio_mean']:.3f}")
        if stale >= int(config["training"].get("early_stopping_patience", 15)):
            break

    (output / "notes.txt").write_text(
        "Half is differentiated only on its active branch. AMP is disabled for matrix solves.\n",
        encoding="utf-8")
    plot_training_curves(output)


if __name__ == "__main__":
    main()
