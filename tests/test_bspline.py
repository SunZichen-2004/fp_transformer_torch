"""Periodic cubic B-spline interpolation used by grid MetaD."""

from __future__ import annotations

import math

import torch

from fp_transformer_torch.math.bspline import (
    cubic_bspline_weights,
    periodic_bspline_eval,
    periodic_cubic_bspline_coeffs,
)

TWO_PI = 2.0 * math.pi


def test_cubic_bspline_weights_partition_of_unity():
    u = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)[:-1]
    w = cubic_bspline_weights(u)
    assert w.shape == (20, 4)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(20, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_bspline_interpolates_nodal_samples_1d():
    n = 48
    origin = torch.tensor([-math.pi], dtype=torch.float64)
    period = torch.tensor([TWO_PI], dtype=torch.float64)
    ax = origin[0] + torch.arange(n, dtype=torch.float64) * (period[0] / n)
    samples = torch.cos(2.0 * ax) + 0.3 * torch.sin(ax)
    coeffs = periodic_cubic_bspline_coeffs(samples)
    got = periodic_bspline_eval(
        coeffs, ax.unsqueeze(-1), origin=origin, period=period
    )
    torch.testing.assert_close(got, samples, atol=1e-9, rtol=1e-9)


def test_bspline_interpolates_nodal_samples_3d():
    n = 16
    origin = torch.full((3,), -math.pi, dtype=torch.float64)
    period = torch.full((3,), TWO_PI, dtype=torch.float64)
    ax = origin[0] + torch.arange(n, dtype=torch.float64) * (period[0] / n)
    x, y, z = torch.meshgrid(ax, ax, ax, indexing="ij")
    samples = torch.cos(x) * torch.sin(y) + 0.2 * torch.cos(2.0 * z)
    coeffs = periodic_cubic_bspline_coeffs(samples)
    pts = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)
    got = periodic_bspline_eval(coeffs, pts, origin=origin, period=period)
    torch.testing.assert_close(got.reshape(n, n, n), samples, atol=1e-9, rtol=1e-9)


def test_bspline_grad_finite_difference():
    n = 24
    origin = torch.full((2,), -math.pi, dtype=torch.float64)
    period = torch.full((2,), TWO_PI, dtype=torch.float64)
    ax = origin[0] + torch.arange(n, dtype=torch.float64) * (period[0] / n)
    x, y = torch.meshgrid(ax, ax, indexing="ij")
    samples = torch.cos(x) + torch.sin(2.0 * y)
    coeffs = periodic_cubic_bspline_coeffs(samples)
    s = torch.tensor([[0.4, -1.1]], dtype=torch.float64, requires_grad=True)
    v = periodic_bspline_eval(coeffs, s, origin=origin, period=period)
    (g,) = torch.autograd.grad(v.sum(), s)
    eps = 1e-6
    s0 = s.detach()
    vp = periodic_bspline_eval(
        coeffs, s0 + torch.tensor([[eps, 0.0]]), origin=origin, period=period
    )
    vm = periodic_bspline_eval(
        coeffs, s0 - torch.tensor([[eps, 0.0]]), origin=origin, period=period
    )
    fd0 = (vp - vm) / (2.0 * eps)
    torch.testing.assert_close(g[0, 0], fd0[0], atol=1e-6, rtol=1e-5)
