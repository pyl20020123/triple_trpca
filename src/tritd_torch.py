"""Batched differentiable TriTD contractions and proximal MALS updates."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

TensorFactors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def triple(a: torch.Tensor, b_factor: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Contract batched factors using the fixed ipq,sjq,spk convention."""
    if a.ndim != 4 or b_factor.ndim != 4 or c.ndim != 4:
        raise ValueError("all factors must be rank-4 batched tensors")
    return torch.einsum("bipq,bsjq,bspk->bijk", a, b_factor, c)


def design_a(b_factor: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    batch, rank, n2, _ = b_factor.shape
    n3 = c.shape[-1]
    return torch.einsum("bsjq,bspk->bpqjk", b_factor, c).reshape(batch, rank * rank, n2 * n3)


def design_b(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    batch, n1, rank, _ = a.shape
    n3 = c.shape[-1]
    return torch.einsum("bipq,bspk->bsqik", a, c).reshape(batch, rank * rank, n1 * n3)


def design_c(a: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
    batch, n1, rank, _ = a.shape
    n2 = b_factor.shape[2]
    return torch.einsum("bipq,bsjq->bspij", a, b_factor).reshape(batch, rank * rank, n1 * n2)


def batched_ridge_solve(x: torch.Tensor, design: torch.Tensor, prev: torch.Tensor,
                        alpha: Union[torch.Tensor, float],
                        eta: Union[torch.Tensor, float]) -> torch.Tensor:
    """Solve U M = R without explicitly forming an inverse."""
    rank2 = design.shape[-2]
    eye = torch.eye(rank2, dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
    gram = design @ design.transpose(-1, -2) + (alpha + eta) * eye
    rhs = x @ design.transpose(-1, -2) + eta * prev
    return torch.linalg.solve(gram, rhs.transpose(-1, -2)).transpose(-1, -2)


def proximal_mals(z: torch.Tensor, factors: TensorFactors,
                  alpha: Union[torch.Tensor, float],
                  eta: Union[torch.Tensor, float]) -> TensorFactors:
    """Perform one Gauss-Seidel A -> B -> C proximal MALS sweep."""
    if z.ndim != 4 or not torch.isfinite(z).all():
        raise ValueError("z must be a finite (batch,n1,n2,n3) tensor")
    a, b_factor, c = factors
    batch, n1, n2, n3 = z.shape
    rank = a.shape[2]

    f = design_a(b_factor, c)
    am = batched_ridge_solve(
        z.reshape(batch, n1, n2 * n3), f,
        a.reshape(batch, n1, rank * rank), alpha, eta,
    )
    a_new = am.reshape(batch, n1, rank, rank)

    g = design_b(a_new, c)
    b_old = b_factor.permute(0, 2, 1, 3).reshape(batch, n2, rank * rank)
    bm = batched_ridge_solve(
        z.permute(0, 2, 1, 3).reshape(batch, n2, n1 * n3), g,
        b_old, alpha, eta,
    )
    b_new = bm.reshape(batch, n2, rank, rank).permute(0, 2, 1, 3)

    q = design_c(a_new, b_new)
    c_old = c.permute(0, 3, 1, 2).reshape(batch, n3, rank * rank)
    cm = batched_ridge_solve(
        z.permute(0, 3, 1, 2).reshape(batch, n3, n1 * n2), q,
        c_old, alpha, eta,
    )
    c_new = cm.reshape(batch, n3, rank, rank).permute(0, 2, 3, 1)
    return a_new, b_new, c_new


def input_only_init(y: torch.Tensor, rank: int, seed: int = 0,
                    generator: Optional[torch.Generator] = None) -> TensorFactors:
    """Initialize from input shape/scale only; no synthetic truth is accepted."""
    if y.ndim != 4:
        raise ValueError("Y must have shape (batch,n1,n2,n3)")
    batch, n1, n2, n3 = y.shape
    if rank < 1 or rank > min(n1, n2, n3):
        raise ValueError("rank must be in [1, min(spatial shape)]")
    if generator is None:
        generator = torch.Generator(device=y.device)
        generator.manual_seed(int(seed))
    scale = y.std(dim=(1, 2, 3), unbiased=False).clamp_min(1e-8)
    sigma = scale.pow(1.0 / 3.0) / math.sqrt(rank)

    def draw(shape: tuple[int, ...]) -> torch.Tensor:
        # A shared fixed random pattern makes each sample's initialization
        # independent of its batch position; only observed scale differs.
        base_shape = (1, *shape[1:])
        values = torch.randn(base_shape, dtype=y.dtype, device=y.device, generator=generator)
        return values.expand(shape) * sigma.reshape(batch, *([1] * (len(shape) - 1)))

    return (
        draw((batch, n1, rank, rank)),
        draw((batch, rank, n2, rank)),
        draw((batch, rank, rank, n3)),
    )

