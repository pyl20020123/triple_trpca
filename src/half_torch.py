"""Exact elementwise Half proximal map with active-branch autograd."""

from __future__ import annotations

import math
from typing import Union

import torch


def half_prox(v: torch.Tensor, beta: Union[torch.Tensor, float],
              return_active_ratio: bool = False):
    beta_t = torch.as_tensor(beta, dtype=v.dtype, device=v.device)
    if beta_t.numel() != 1:
        raise ValueError("beta must be a scalar")
    if not torch.isfinite(beta_t).all() or bool((beta_t.detach() < 0).item()):
        raise ValueError("beta must be finite and non-negative")
    if not torch.isfinite(v).all():
        raise ValueError("v must contain only finite values")

    abs_v = v.abs()
    threshold = 1.5 * beta_t.pow(2.0 / 3.0)
    active = abs_v > threshold  # the discrete mask intentionally has no gradient
    safe_a = torch.where(active, abs_v, torch.ones_like(abs_v))
    argument = -3.0 * math.sqrt(3.0) * beta_t / (4.0 * safe_a.pow(1.5))
    theta = torch.acos(argument.clamp(-1.0, 1.0)) / 3.0
    magnitude = (4.0 * safe_a / 3.0) * torch.cos(theta).square()
    output = torch.where(active, v.sign() * magnitude, torch.zeros_like(v))
    if return_active_ratio:
        return output, active.to(v.dtype).mean()
    return output

