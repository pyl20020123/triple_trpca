"""Evaluation metrics; never used for selecting an initialization from truth."""

from __future__ import annotations

import torch
from torch.nn import functional as F


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
    mse = (low_hat.clamp(0.0, 1.0) - low_true).square().mean(dim=dims).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    ssim = batch_ssim(low_hat.clamp(0.0, 1.0), low_true)
    return {"rre_l": _rre(low_hat, low_true), "rre_s": _rre(sparse_hat, sparse_true),
            "precision": precision, "recall": recall, "f1": f1,
            "residual_rre": residual_rre, "psnr": psnr, "ssim": ssim}


def batch_ssim(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Differentiation-free reporting SSIM for RGB HWC or generic tensors."""
    if estimate.ndim != 4 or estimate.shape[-1] not in (1, 3):
        # SSIM is an image metric; use a bounded global variant for synthetic tensors.
        dims = tuple(range(1, truth.ndim))
        mu_x, mu_y = estimate.mean(dim=dims), truth.mean(dim=dims)
        var_x, var_y = estimate.var(dim=dims, unbiased=False), truth.var(dim=dims, unbiased=False)
        covariance = ((estimate - mu_x.view(-1, 1, 1, 1)) *
                      (truth - mu_y.view(-1, 1, 1, 1))).mean(dim=dims)
        return ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2) /
                ((mu_x.square() + mu_y.square() + 0.01 ** 2) *
                 (var_x + var_y + 0.03 ** 2))).clamp(-1.0, 1.0)
    x, y = estimate.permute(0, 3, 1, 2), truth.permute(0, 3, 1, 2)
    kernel = min(11, x.shape[-2], x.shape[-1])
    if kernel % 2 == 0:
        kernel -= 1
    padding = kernel // 2
    mu_x = F.avg_pool2d(x, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel, stride=1, padding=padding)
    var_x = F.avg_pool2d(x.square(), kernel, 1, padding) - mu_x.square()
    var_y = F.avg_pool2d(y.square(), kernel, 1, padding) - mu_y.square()
    covariance = F.avg_pool2d(x * y, kernel, 1, padding) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2) /
             ((mu_x.square() + mu_y.square() + 0.01 ** 2) *
              (var_x + var_y + 0.03 ** 2)))
    return score.mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)

