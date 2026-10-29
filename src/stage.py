"""One interpretable Half -> proximal-MALS unfolding stage."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .half_torch import half_prox
from .tritd_torch import TensorFactors, proximal_mals, triple


def inverse_softplus(value: float, eps: float = 1e-8) -> float:
    target = max(float(value) - eps, 1e-12)
    return math.log(math.expm1(target))


class HalfMALSStage(nn.Module):
    def __init__(self, alpha: float = 1e-4, lambda_init: float = 0.1,
                 eta_init: float = 1e-3, eta_s_init: float = 1e-3,
                 positivity_eps: float = 1e-8, learnable: bool = True):
        super().__init__()
        if alpha < 0 or min(lambda_init, eta_init, eta_s_init) < 0:
            raise ValueError("stage parameters must be non-negative")
        self.alpha = float(alpha)
        self.positivity_eps = float(positivity_eps)
        parameter = nn.Parameter if learnable else lambda x: nn.Parameter(x, requires_grad=False)
        # Start in float64 so a later model.double() parity check does not preserve
        # an avoidable float32 quantization error. model.float() still works normally.
        self.raw_lam = parameter(torch.tensor(inverse_softplus(lambda_init, positivity_eps), dtype=torch.float64))
        self.raw_eta = parameter(torch.tensor(inverse_softplus(eta_init, positivity_eps), dtype=torch.float64))
        self.raw_eta_s = parameter(torch.tensor(inverse_softplus(eta_s_init, positivity_eps), dtype=torch.float64))

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self.raw_lam) + self.positivity_eps

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta) + self.positivity_eps

    @property
    def eta_s(self) -> torch.Tensor:
        return F.softplus(self.raw_eta_s) + self.positivity_eps

    def forward(self, y: torch.Tensor, sparse: torch.Tensor,
                factors: TensorFactors) -> tuple[torch.Tensor, torch.Tensor, TensorFactors, torch.Tensor]:
        previous_low = triple(*factors)
        eta_s = self.eta_s.to(dtype=y.dtype, device=y.device)
        lam = self.lam.to(dtype=y.dtype, device=y.device)
        v = (y - previous_low + eta_s * sparse) / (1.0 + eta_s)
        beta = lam / (1.0 + eta_s)
        sparse_new, active_ratio = half_prox(v, beta, return_active_ratio=True)
        eta = self.eta.to(dtype=y.dtype, device=y.device)
        factors_new = proximal_mals(y - sparse_new, factors, self.alpha, eta)
        low_new = triple(*factors_new)
        return low_new, sparse_new, factors_new, active_ratio

