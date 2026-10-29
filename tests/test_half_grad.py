import torch

from src.half_torch import half_prox


def test_half_active_branch_gradient_matches_finite_difference():
    v = torch.tensor(2.7, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
    output = half_prox(v, beta)
    grad_v, grad_beta = torch.autograd.grad(output, (v, beta))
    step = 1e-6
    with torch.no_grad():
        fd_v = (half_prox(v + step, beta) - half_prox(v - step, beta)) / (2 * step)
        fd_beta = (half_prox(v, beta + step) - half_prox(v, beta - step)) / (2 * step)
    torch.testing.assert_close(grad_v, fd_v, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(grad_beta, fd_beta, rtol=2e-6, atol=2e-7)


def test_inactive_branch_has_zero_gradient():
    v = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    output = half_prox(v, beta)
    grads = torch.autograd.grad(output, (v, beta), allow_unused=True)
    assert output == 0
    assert grads[0] == 0 and grads[1] == 0

