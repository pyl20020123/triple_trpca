"""Evaluation metrics; never used for selecting an initialization from truth."""

from __future__ import annotations

import torch


def _rre(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, truth.ndim))
    numerator = torch.linalg.vector_norm(estimate - truth, dim=dims)
    denominator = torch.linalg.vector_norm(truth, dim=dims).clamp_min(1e-12)
    return numerator / denominator


def batch_metrics(low_true: torch.Tensor, sparse_true: torch.Tensor,
                  low_hat: torch.Tensor, sparse_hat: torch.Tensor,
                  threshold: float = 0.05) -> dict[str, torch.Tensor]:
    true_support = sparse_true != 0
    pred_support = sparse_hat.abs() > threshold
    dims = tuple(range(1, low_true.ndim))
    tp = (true_support & pred_support).sum(dim=dims).to(low_true.dtype)
    fp = (~true_support & pred_support).sum(dim=dims).to(low_true.dtype)
    fn = (true_support & ~pred_support).sum(dim=dims).to(low_true.dtype)
    true_count = true_support.sum(dim=dims)
    pred_count = pred_support.sum(dim=dims)
    precision = torch.where(pred_count == 0, (true_count == 0).to(low_true.dtype), tp / (tp + fp))
    recall = torch.where(true_count == 0, torch.ones_like(tp), tp / (tp + fn))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    observed = low_true + sparse_true
    residual = observed - low_hat - sparse_hat
    residual_rre = torch.linalg.vector_norm(residual, dim=dims) / torch.linalg.vector_norm(
        observed, dim=dims).clamp_min(1e-12)
    return {"rre_l": _rre(low_hat, low_true), "rre_s": _rre(sparse_hat, sparse_true),
            "precision": precision, "recall": recall, "f1": f1,
            "residual_rre": residual_rre}

