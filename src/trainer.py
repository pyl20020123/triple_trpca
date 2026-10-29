"""Training/evaluation loops shared by CLI programs and tests."""

from __future__ import annotations

import time

import torch

from .losses import supervised_loss
from .metrics import batch_metrics


def _move(batch: dict, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return tuple(batch[key].to(device=device, dtype=dtype) for key in ("Y", "L_true", "S_true"))


def train_one_epoch(loader, optimizer, model, config: dict,
                    device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in ("loss_total", "loss_rec", "loss_sparse", "rre_l_mean",
                                    "rre_s_mean", "f1_mean", "active_ratio_mean")}
    samples = 0
    active_by_stage = None
    last_grad_norm = 0.0
    started = time.perf_counter()
    loss_cfg = config["loss"]
    for batch in loader:
        observed, low_true, sparse_true = _move(batch, device, dtype)
        optimizer.zero_grad(set_to_none=True)
        # Linear solves intentionally run at the model dtype, outside low-precision AMP.
        output = model(observed)
        losses = supervised_loss(output, low_true, sparse_true,
                                 gamma_s=loss_cfg.get("gamma_s", 0.1),
                                 gamma_d=loss_cfg.get("gamma_d", 0.0), observed=observed,
                                 decay=loss_cfg.get("stage_decay", 0.8))
        losses["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("clip_grad_norm", 1.0))
        last_grad_norm = float(norm.detach())
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        with torch.no_grad():
            metrics = batch_metrics(low_true, sparse_true, output["L"], output["S"],
                                    config["evaluation"].get("support_threshold", 0.05))
        count = observed.shape[0]
        samples += count
        totals["loss_total"] += float(losses["total"].detach()) * count
        totals["loss_rec"] += float(losses["rec"].detach()) * count
        totals["loss_sparse"] += float(losses["sparse"].detach()) * count
        for key in ("rre_l", "rre_s", "f1"):
            totals[f"{key}_mean"] += float(metrics[key].sum())
        totals["active_ratio_mean"] += float(output["active_ratios"].mean().detach()) * count
        stage_values = output["active_ratios"].detach().cpu() * count
        active_by_stage = stage_values if active_by_stage is None else active_by_stage + stage_values
    result = {key: value / max(samples, 1) for key, value in totals.items()}
    result.update({"grad_norm": last_grad_norm, "seconds": time.perf_counter() - started})
    result["active_ratios_by_stage"] = (active_by_stage / max(samples, 1)).tolist()
    return result


@torch.no_grad()
def evaluate_loader(loader, model, config: dict, device: torch.device,
                    dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("loss_total", "loss_rec", "loss_sparse", "rre_l_mean",
                                    "rre_s_mean", "f1_mean", "active_ratio_mean")}
    samples = 0
    active_by_stage = None
    started = time.perf_counter()
    loss_cfg = config["loss"]
    for batch in loader:
        observed, low_true, sparse_true = _move(batch, device, dtype)
        output = model(observed)
        losses = supervised_loss(output, low_true, sparse_true,
                                 gamma_s=loss_cfg.get("gamma_s", 0.1),
                                 gamma_d=loss_cfg.get("gamma_d", 0.0), observed=observed,
                                 decay=loss_cfg.get("stage_decay", 0.8))
        metrics = batch_metrics(low_true, sparse_true, output["L"], output["S"],
                                config["evaluation"].get("support_threshold", 0.05))
        count = observed.shape[0]
        samples += count
        totals["loss_total"] += float(losses["total"]) * count
        totals["loss_rec"] += float(losses["rec"]) * count
        totals["loss_sparse"] += float(losses["sparse"]) * count
        for key in ("rre_l", "rre_s", "f1"):
            totals[f"{key}_mean"] += float(metrics[key].sum())
        totals["active_ratio_mean"] += float(output["active_ratios"].mean()) * count
        stage_values = output["active_ratios"].detach().cpu() * count
        active_by_stage = stage_values if active_by_stage is None else active_by_stage + stage_values
    result = {key: value / max(samples, 1) for key, value in totals.items()}
    result.update({"grad_norm": 0.0, "seconds": time.perf_counter() - started})
    result["active_ratios_by_stage"] = (active_by_stage / max(samples, 1)).tolist()
    return result

