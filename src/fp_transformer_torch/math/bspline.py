"""Periodic cubic B-spline interpolation (Unser prefilter + local 4-point eval).

Used to evaluate a tabulated metadynamics bias ``V(s)`` in O(1) per sample
instead of summing Gaussians.
"""

from __future__ import annotations

import math

import torch


def cubic_bspline_weights(u: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline basis at offset ``u ∈ [0, 1)``. Returns ``(..., 4)``."""
    um = 1.0 - u
    w0 = (um * um * um) / 6.0
    w1 = (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0
    w2 = (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0
    w3 = (u * u * u) / 6.0
    return torch.stack((w0, w1, w2, w3), dim=-1)


def periodic_cubic_bspline_coeffs(samples: torch.Tensor) -> torch.Tensor:
    """Convert periodic nodal samples to interpolating cubic B-spline coeffs.

    Separable Unser prefilter via FFT: the cubic B-spline sampled on integers
    has DFT ``H(k) = 2/3 + (1/3) cos(2π k / n)``.
    """
    spec = torch.fft.fftn(samples)
    real_dtype = samples.dtype
    for d, n in enumerate(samples.shape):
        freq = torch.fft.fftfreq(int(n), device=samples.device).to(dtype=real_dtype)
        transfer = (2.0 / 3.0) + (1.0 / 3.0) * torch.cos(2.0 * math.pi * freq)
        shape = [1] * samples.ndim
        shape[d] = int(n)
        spec = spec / transfer.to(dtype=spec.dtype).reshape(shape)
    return torch.fft.ifftn(spec).real.to(dtype=samples.dtype)


def periodic_bspline_eval(
    coeffs: torch.Tensor,
    s: torch.Tensor,
    *,
    origin: torch.Tensor,
    period: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a periodic cubic B-spline on a 1–3D coefficient grid.

    ``s`` is ``(B, n_cv)``; ``coeffs`` has shape ``n_cv``-dimensional.
    """
    nd = int(coeffs.ndim)
    if s.ndim != 2 or int(s.shape[-1]) != nd:
        raise ValueError(f"s shape {tuple(s.shape)} incompatible with coeffs {tuple(coeffs.shape)}")
    if nd < 1 or nd > 3:
        raise ValueError(f"B-spline eval supports 1–3 D, got nd={nd}")
    four = torch.arange(-1, 3, device=s.device)
    weights: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for d in range(nd):
        n = int(coeffs.shape[d])
        scale = float(n) / period[d]
        x = torch.remainder((s[:, d] - origin[d]) * scale, n)
        i_floor = torch.floor(x)
        weights.append(cubic_bspline_weights(x - i_floor))
        indices.append((i_floor.to(dtype=torch.long).unsqueeze(1) + four) % n)
    if nd == 1:
        return (coeffs[indices[0]] * weights[0]).sum(dim=-1)
    if nd == 2:
        ii = indices[0][:, :, None].expand(-1, 4, 4)
        jj = indices[1][:, None, :].expand(-1, 4, 4)
        vals = coeffs[ii, jj]
        return (vals * weights[0][:, :, None] * weights[1][:, None, :]).sum(dim=(1, 2))
    ii = indices[0][:, :, None, None].expand(-1, 4, 4, 4)
    jj = indices[1][:, None, :, None].expand(-1, 4, 4, 4)
    kk = indices[2][:, None, None, :].expand(-1, 4, 4, 4)
    vals = coeffs[ii, jj, kk]
    w0 = weights[0][:, :, None, None]
    w1 = weights[1][:, None, :, None]
    w2 = weights[2][:, None, None, :]
    return (vals * w0 * w1 * w2).sum(dim=(1, 2, 3))
