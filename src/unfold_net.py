"""K-stage TriTD-Half deep unfolding network."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .stage import HalfMALSStage
from .tritd_torch import TensorFactors, input_only_init, proximal_mals, triple


class TriTDHalfUnfold(nn.Module):
    def __init__(self, rank: int = 2, stages: int = 5, warmup_sweeps: int = 2,
                 lambda_init: float = 0.1, eta_init: float = 1e-3,
                 eta_s_init: float = 1e-3, alpha: float = 1e-4,
                 positivity_eps: float = 1e-8, learnable: bool = True,
                 init_seed: int = 17):
        super().__init__()
        if rank < 1 or stages < 1 or warmup_sweeps < 0:
            raise ValueError("rank/stages must be positive and warmup_sweeps non-negative")
        self.rank = int(rank)
        self.num_stages = int(stages)
        self.warmup_sweeps = int(warmup_sweeps)
        self.alpha = float(alpha)
        self.init_seed = int(init_seed)
        self.stages = nn.ModuleList([
            HalfMALSStage(alpha, lambda_init, eta_init, eta_s_init,
                          positivity_eps, learnable)
            for _ in range(stages)
        ])

    def forward(self, y: torch.Tensor, seed: Optional[int] = None,
                factors0: Optional[TensorFactors] = None) -> dict:
        if y.ndim != 4:
            raise ValueError("Y must have shape (batch,n1,n2,n3)")
        if not torch.is_floating_point(y) or not torch.isfinite(y).all():
            raise ValueError("Y must be a finite floating-point tensor")
        if self.rank > min(y.shape[1:]):
            raise ValueError("model rank exceeds an input dimension")
        if factors0 is None:
            with torch.no_grad():
                factors = input_only_init(y, self.rank, self.init_seed if seed is None else seed)
                warm_eta = self.stages[0].eta.detach().to(dtype=y.dtype, device=y.device)
                for _ in range(self.warmup_sweeps):
                    factors = proximal_mals(y, factors, self.alpha, warm_eta)
        else:
            factors = factors0
        sparse = torch.zeros_like(y)
        low_stages, sparse_stages, active_ratios = [], [], []
        for stage in self.stages:
            low, sparse, factors, active = stage(y, sparse, factors)
            low_stages.append(low)
            sparse_stages.append(sparse)
            active_ratios.append(active)
        return {
            "L": low_stages[-1], "S": sparse_stages[-1], "factors": factors,
            "L_stages": low_stages, "S_stages": sparse_stages,
            "active_ratios": torch.stack(active_ratios),
        }

    def stage_parameters(self) -> list[dict[str, float]]:
        return [{
            "stage": i + 1,
            "lambda": float(stage.lam.detach()),
            "eta": float(stage.eta.detach()),
            "eta_s": float(stage.eta_s.detach()),
            "alpha": stage.alpha,
        } for i, stage in enumerate(self.stages)]
