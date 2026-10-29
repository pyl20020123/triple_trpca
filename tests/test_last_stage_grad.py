import torch

from src.tritd_torch import input_only_init
from src.unfold_net import TriTDHalfUnfold


def test_last_stage_lambda_receives_nonzero_gradient():
    torch.manual_seed(2)
    y = torch.randn(1, 4, 4, 3, dtype=torch.float64)
    y[:, 0, 0, 0] += 6.0
    y[:, 2, 3, 1] -= 5.0
    model = TriTDHalfUnfold(rank=2, stages=2, warmup_sweeps=0,
                            lambda_init=0.1, eta_init=1e-2, eta_s_init=1e-3).double()
    factors0 = input_only_init(y, 2, seed=19)
    output = model(y, factors0=factors0)
    assert output["active_ratios"][-1] > 0
    output["L"].square().mean().backward()
    grad = model.stages[-1].raw_lam.grad
    assert grad is not None and torch.isfinite(grad) and grad.abs() > 1e-12
    for stage in model.stages:
        for parameter in stage.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad)

