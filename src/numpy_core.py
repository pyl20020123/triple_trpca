"""Deterministic NumPy reference implementation used by tests and baselines."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

Factors = Tuple[np.ndarray, np.ndarray, np.ndarray]


def triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.einsum("ipq,sjq,spk->ijk", a, b, c, optimize=True)


def half_prox(v: np.ndarray, beta: float) -> np.ndarray:
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and non-negative")
    x = np.asarray(v)
    if not np.all(np.isfinite(x)):
        raise ValueError("v must contain only finite values")
    if beta == 0:
        return x.copy()
    a = np.abs(x)
    threshold = 1.5 * beta ** (2.0 / 3.0)
    active = a > threshold
    out = np.zeros_like(x)
    if np.any(active):
        aa = a[active]
        arg = -3.0 * np.sqrt(3.0) * beta / (4.0 * aa ** 1.5)
        theta = np.arccos(np.clip(arg, -1.0, 1.0)) / 3.0
        u = (4.0 * aa / 3.0) * np.cos(theta) ** 2
        out[active] = np.sign(x[active]) * u
    return out


def init_factors(shape: Iterable[int], rank: int, seed: int, scale: float = 1.0) -> Factors:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 3 or min(dims) < 1:
        raise ValueError("shape must contain three positive dimensions")
    if rank < 1 or rank > min(dims):
        raise ValueError("rank must be in [1, min(shape)]")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    rng = np.random.default_rng(seed)
    sigma = scale ** (1.0 / 3.0) / math.sqrt(rank)
    n1, n2, n3 = dims
    return (
        rng.normal(0.0, sigma, (n1, rank, rank)),
        rng.normal(0.0, sigma, (rank, n2, rank)),
        rng.normal(0.0, sigma, (rank, rank, n3)),
    )


def _ridge_solve(x: np.ndarray, design: np.ndarray, prev: np.ndarray,
                 alpha: float, eta: float) -> np.ndarray:
    r2 = design.shape[0]
    gram = design @ design.T + (alpha + eta) * np.eye(r2, dtype=x.dtype)
    rhs = x @ design.T + eta * prev
    return np.linalg.solve(gram, rhs.T).T


def proximal_mals(z: np.ndarray, factors: Factors, alpha: float, eta: float) -> Factors:
    a, b, c = factors
    n1, n2, n3 = z.shape
    rank = a.shape[1]
    f = np.einsum("sjq,spk->pqjk", b, c, optimize=True).reshape(rank * rank, n2 * n3)
    am = _ridge_solve(z.reshape(n1, n2 * n3), f, a.reshape(n1, rank * rank), alpha, eta)
    a_new = am.reshape(n1, rank, rank)

    g = np.einsum("ipq,spk->sqik", a_new, c, optimize=True).reshape(rank * rank, n1 * n3)
    b_old = b.transpose(1, 0, 2).reshape(n2, rank * rank)
    bm = _ridge_solve(z.transpose(1, 0, 2).reshape(n2, n1 * n3), g, b_old, alpha, eta)
    b_new = bm.reshape(n2, rank, rank).transpose(1, 0, 2)

    q = np.einsum("ipq,sjq->spij", a_new, b_new, optimize=True).reshape(rank * rank, n1 * n2)
    c_old = c.transpose(2, 0, 1).reshape(n3, rank * rank)
    cm = _ridge_solve(z.transpose(2, 0, 1).reshape(n3, n1 * n2), q, c_old, alpha, eta)
    c_new = cm.reshape(n3, rank, rank).transpose(1, 2, 0)
    return a_new, b_new, c_new


def generate_case(shape: Iterable[int], rank: int, rate: float, amplitude: float,
                  seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = tuple(int(x) for x in shape)
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    if rate > 0 and amplitude <= 0:
        raise ValueError("amplitude must be positive when rate > 0")
    factors = init_factors(dims, rank, seed + 900001)
    low = triple(*factors)
    low = low / max(float(low.std()), 1e-10)
    sparse = np.zeros(dims, dtype=np.float64)
    count = round(rate * sparse.size)
    if count:
        rng = np.random.default_rng(seed + 400003)
        indices = rng.choice(sparse.size, size=count, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=count)
        magnitudes = rng.uniform(0.75 * amplitude, 1.25 * amplitude, size=count)
        sparse.reshape(-1)[indices] = signs * magnitudes
    return low, sparse, low + sparse


def objective(y: np.ndarray, factors: Factors, sparse: np.ndarray,
              lam: float, alpha: float) -> float:
    residual = y - triple(*factors) - sparse
    factor_norm = sum(float(np.sum(x * x)) for x in factors)
    return float(0.5 * np.sum(residual * residual)
                 + lam * np.sqrt(np.abs(sparse)).sum()
                 + 0.5 * alpha * factor_norm)


def fit_trpca(y: np.ndarray, rank: int, lam: float = 0.1, alpha: float = 1e-4,
              eta: float = 1e-3, max_iter: int = 100, seed: int = 0,
              freeze_sparse: bool = False, initial_factors: Factors = None) -> dict:
    y = np.asarray(y, dtype=np.float64)
    factors = (init_factors(y.shape, rank, seed, scale=max(float(y.std()), 1e-8))
               if initial_factors is None
               else tuple(np.asarray(factor, dtype=np.float64).copy() for factor in initial_factors))
    sparse = np.zeros_like(y)
    history = []
    for iteration in range(max_iter + 1):
        history.append((iteration, objective(y, factors, sparse, lam, alpha)))
        if iteration == max_iter:
            break
        factors = proximal_mals(y - sparse, factors, alpha, eta)
        if not freeze_sparse:
            sparse = half_prox(y - triple(*factors), lam)
    return {"L": triple(*factors), "S": sparse, "factors": factors,
            "history": np.asarray(history), "final_objective": history[-1][1]}
