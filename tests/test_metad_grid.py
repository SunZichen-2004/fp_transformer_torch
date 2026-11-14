"""Grid + cubic B-spline MetaD bias vs exact sum of Gaussians."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch_sim")
import torch

torch_sim = pytest.importorskip("torch_sim")

from fp_transformer_torch.integrator.enhanced_sampling import (  # noqa: E402
    WellTemperedMetadynamicsBias,
)

TWO_PI = 2.0 * math.pi


def _cv(coord: torch.Tensor) -> torch.Tensor:
    return coord[:, 0, :]


def _make_bias(*, grid_n, device="cpu", dtype=torch.float64):
    return WellTemperedMetadynamicsBias(
        _cv,
        n_cv=3,
        sigma=(0.35, 0.35, 0.35),
        height=0.02,
        bias_factor=15.0,
        temperature=300.0,
        max_hills=0,
        period=TWO_PI,
        grid_n=grid_n,
        device=device,
        dtype=dtype,
        n_atoms_per_system=1,
    )


def test_grid_bias_matches_exact_gaussians():
    device = torch.device("cpu")
    rng = torch.Generator(device="cpu").manual_seed(1)
    s = (torch.rand(48, 3, generator=rng) - 0.5) * TWO_PI
    coord = s.unsqueeze(1)
    exact = _make_bias(grid_n=0)
    grid = _make_bias(grid_n=81)
    exact.deposit(coord)
    grid.deposit(coord)
    query = (torch.rand(200, 3, generator=rng) - 0.5) * TWO_PI
    v_ex = exact.bias_potential_cv(query)
    v_gr = grid.bias_potential_cv(query)
    err = (v_ex - v_gr).abs()
    # σ/dx ≈ 4.5; interpolating cubic spline of the sampled Gaussians.
    assert float(err.max()) < 5e-4
    assert float(err.mean()) < 5e-5
    torch.testing.assert_close(v_ex, v_gr, atol=5e-4, rtol=5e-3)


def test_grid_bias_grad_matches_exact():
    rng = torch.Generator().manual_seed(2)
    s = (torch.rand(16, 3, generator=rng) - 0.5) * TWO_PI
    coord = s.unsqueeze(1)
    exact = _make_bias(grid_n=0)
    grid = _make_bias(grid_n=81)
    exact.deposit(coord)
    grid.deposit(coord)
    q = (torch.rand(32, 3, generator=rng) - 0.5) * TWO_PI
    qe = q.clone().requires_grad_(True)
    qg = q.clone().requires_grad_(True)
    (ge,) = torch.autograd.grad(exact.bias_potential_cv(qe).sum(), qe)
    (gg,) = torch.autograd.grad(grid.bias_potential_cv(qg).sum(), qg)
    err = (ge - gg).abs()
    assert float(err.max()) < 5e-3
    assert float(err.mean()) < 5e-4


def test_nonperiodic_grid_bias_matches_exact():
    """Distance-like CVs: grid_min/grid_max without period."""

    def cv2(coord: torch.Tensor) -> torch.Tensor:
        return coord[:, 0, :2]

    kw = dict(
        n_cv=2,
        sigma=(0.2, 0.2),
        height=0.02,
        bias_factor=30.0,
        temperature=300.0,
        max_hills=0,
        device="cpu",
        dtype=torch.float64,
        n_atoms_per_system=1,
    )
    exact = WellTemperedMetadynamicsBias(cv2, **kw)
    grid = WellTemperedMetadynamicsBias(
        cv2, grid_n=81, grid_min=(0.0, 0.0), grid_max=(10.0, 10.0), **kw
    )
    assert grid.use_grid and grid.period is None
    rng = torch.Generator().manual_seed(3)
    s = 1.0 + 5.0 * torch.rand(24, 2, generator=rng, dtype=torch.float64)
    coord = torch.zeros(24, 1, 3, dtype=torch.float64)
    coord[:, 0, :2] = s
    exact.deposit(coord)
    grid.deposit(coord)
    q = 1.0 + 5.0 * torch.rand(64, 2, generator=rng, dtype=torch.float64)
    err = (exact.bias_potential_cv(q) - grid.bias_potential_cv(q)).abs()
    assert float(err.max()) < 5e-4
    assert float(err.mean()) < 5e-5
