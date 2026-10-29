"""Stage-wise supervised loss for low-rank recovery."""

from __future__ import annotations

import torch
from typing import Optional


def stage_weights(stages: int, decay: float = 0.8, *, device=None, dtype=None) -> torch.Tensor:
    values = torch.tensor([decay ** (stages - 1 - k) for k in range(stages)],
                          device=device, dtype=dtype)
    return values / values.sum()


def supervised_loss(output: dict, low_true: torch.Tensor,
                    sparse_true: Optional[torch.Tensor] = None,
                    gamma_s: float = 0.1, gamma_d: float = 0.0,
                    observed: Optional[torch.Tensor] = None,
                    decay: float = 0.8) -> dict[str, torch.Tensor]:
    lows = output["L_stages"]
    sparsities = output["S_stages"]
    weights = stage_weights(len(lows), decay, device=low_true.device, dtype=low_true.dtype)
    rec_terms = torch.stack([(estimate - low_true).square().mean() for estimate in lows])
    rec = (weights * rec_terms).sum()
    sparse = torch.zeros((), dtype=low_true.dtype, device=low_true.device)
    if sparse_true is not None:
        sparse_terms = torch.stack([(estimate - sparse_true).abs().mean() for estimate in sparsities])
        sparse = (weights * sparse_terms).sum()
    consistency = torch.zeros_like(rec)
    if observed is not None and gamma_d:
        terms = torch.stack([(observed - low - sparse_est).square().mean()
                             for low, sparse_est in zip(lows, sparsities)])
        consistency = (weights * terms).sum()
    total = rec + gamma_s * sparse + gamma_d * consistency
    return {"total": total, "rec": rec, "sparse": sparse,
            "consistency": consistency, "stage_rec": rec_terms}

