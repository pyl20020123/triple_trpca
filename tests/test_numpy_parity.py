import numpy as np
import torch

from src.half_torch import half_prox as torch_half
from src.numpy_core import half_prox as numpy_half
from src.numpy_core import proximal_mals as numpy_mals
from src.numpy_core import triple as numpy_triple
from src.stage import HalfMALSStage
from src.tritd_torch import proximal_mals as torch_mals
from src.tritd_torch import triple as torch_triple


def factors(seed=4):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(3, 2, 2)), rng.normal(size=(2, 4, 2)),
            rng.normal(size=(2, 2, 3)))


def test_tensor_contraction_numpy_torch():
    a, b, c = factors()
    expected = numpy_triple(a, b, c)
    actual = torch_triple(*(torch.tensor(x, dtype=torch.float64)[None] for x in (a, b, c)))
    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=1e-12, atol=1e-12)


def test_mals_numpy_torch():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(3, 4, 3))
    initial = factors()
    expected = numpy_mals(z, initial, 1e-4, 1e-3)
    actual = torch_mals(torch.tensor(z, dtype=torch.float64)[None],
                        tuple(torch.tensor(x, dtype=torch.float64)[None] for x in initial), 1e-4, 1e-3)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right[0].detach().numpy(), rtol=1e-10, atol=1e-10)


def test_half_and_stage_order_match_numpy():
    initial = factors(8)
    low0 = numpy_triple(*initial)
    y = low0.copy()
    y.reshape(-1)[::7] += 3.0
    lam, eta_s, eta, alpha = 0.1, 0.02, 0.003, 1e-4
    beta = lam / (1 + eta_s)
    sparse_expected = numpy_half((y - low0) / (1 + eta_s), beta)
    factors_expected = numpy_mals(y - sparse_expected, initial, alpha, eta)

    stage = HalfMALSStage(alpha, lam, eta, eta_s).double()
    low, sparse, factors_actual, _ = stage(
        torch.tensor(y)[None], torch.zeros_like(torch.tensor(y))[None],
        tuple(torch.tensor(x)[None] for x in initial))
    np.testing.assert_allclose(sparse[0].detach(), sparse_expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(low[0].detach(), numpy_triple(*factors_expected), rtol=1e-9, atol=1e-9)


def test_half_threshold_tie_is_zero():
    beta = 0.2
    threshold = 1.5 * beta ** (2 / 3)
    values = np.array([-threshold * 1.0001, -threshold, 0.0, threshold, threshold * 1.0001])
    expected = numpy_half(values, beta)
    actual = torch_half(torch.tensor(values, dtype=torch.float64), torch.tensor(beta, dtype=torch.float64))
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-10, atol=1e-10)
    assert actual[1] == 0 and actual[3] == 0

