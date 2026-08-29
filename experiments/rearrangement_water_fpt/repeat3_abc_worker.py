#!/usr/bin/env python3
"""Slim KDE worker for plot_repeat3_abc.py (spawn-safe, no matplotlib)."""

from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

LAM = 0.025
T_K = 300.0
P_FLOOR = 1e-6
VMAX_RMSE = 60.0
VMAX30 = 30.0
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
BETA0 = 1.0 / (KB_EV_PER_K * T_K)
KDE_CHUNK = 4096


def bandwidth_column_squared(bandwidth: torch.Tensor) -> torch.Tensor:
    vector = bandwidth.reshape(-1)
    return (vector * vector).unsqueeze(0)


def gaussian_kernel_log_normalization(bandwidth: torch.Tensor, feature_dim: float) -> torch.Tensor:
    sigma_squared = bandwidth_column_squared(bandwidth)
    return -0.5 * feature_dim * torch.log(2.0 * torch.pi * sigma_squared)

_W: dict = {}


def relative_fes(f: np.ndarray) -> np.ndarray:
    out = np.asarray(f, dtype=np.float64)
    return out - np.nanmin(out[np.isfinite(out)])


def project_1d_fixed_edges(d1, d2, f_kcal, edges, *, temperature: float):
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    xi = (g2 - g1).ravel()
    f = np.asarray(f_kcal, dtype=np.float64).ravel()
    centers = 0.5 * (edges[:-1] + edges[1:])
    kt = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kt
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            sel = (xi >= edges[i]) & (xi < edges[i + 1])
        else:
            sel = (xi >= edges[i]) & (xi <= edges[i + 1])
        sel = sel & np.isfinite(f)
        if not np.any(sel):
            continue
        vals = -beta * f[sel]
        m = float(np.max(vals))
        out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
    finite = np.isfinite(out)
    if np.any(finite):
        out = out - np.nanmin(out[finite])
    return centers, out


def rmse_1d(f_a, f_b, *, vmax, xi, xi_lo=None, xi_hi=None) -> float:
    a = np.asarray(f_a, dtype=np.float64)
    b = np.asarray(f_b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if vmax is not None and float(vmax) > 0:
        m = m & (b <= float(vmax))
    if not np.any(m):
        return float("nan")
    aa = a[m] - np.min(a[m])
    bb = b[m] - np.min(b[m])
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def delta_g_from_2d(d1, d2, f_kcal, *, temperature: float, xi_cut: float = 0.0) -> float:
    d1_a = np.asarray(d1, dtype=np.float64)
    d2_a = np.asarray(d2, dtype=np.float64)
    f = np.asarray(f_kcal, dtype=np.float64)
    g1, g2 = np.meshgrid(d1_a, d2_a, indexing="xy")
    kt = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kt
    xi = g2 - g1
    m = np.isfinite(f) & np.isfinite(xi)
    b1 = m & (xi < float(xi_cut))
    b2 = m & (xi > float(xi_cut))
    if not np.any(b1) or not np.any(b2):
        return float("nan")

    def _logZ(sel):
        vals = -beta * f[sel]
        mval = float(np.max(vals))
        return mval + float(np.log(np.sum(np.exp(vals - mval))))

    return float(kt * (_logZ(b1) - _logZ(b2)))


def density_f32(grid_xy: np.ndarray, sample_cv: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    samples = torch.as_tensor(np.asarray(sample_cv, dtype=np.float32).reshape(-1, 2))
    bw = torch.as_tensor(np.asarray(bandwidth, dtype=np.float32).reshape(-1))
    n = float(samples.shape[0])
    feature_dim = 2.0
    s2 = torch.sum(samples * samples, dim=1)
    m = int(grid_xy.shape[0])
    out = np.empty((m,), dtype=np.float64)
    log_norm = gaussian_kernel_log_normalization(bw, feature_dim)
    bw2 = bandwidth_column_squared(bw)
    for start in range(0, m, KDE_CHUNK):
        stop = min(start + KDE_CHUNK, m)
        grid = torch.as_tensor(np.asarray(grid_xy[start:stop], dtype=np.float32).reshape(-1, 2))
        g2 = torch.sum(grid * grid, dim=1, keepdim=True)
        dist = torch.clamp(g2 + s2.unsqueeze(0) - 2.0 * (grid @ samples.T), min=0.0) / feature_dim
        log_k = (-0.5 * dist) / bw2 + log_norm
        log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
        out[start:stop] = torch.exp(log_p).numpy()
    return out


def kde_fes_from_stages(stages, *, d1, d2, grid_xy, g1_shape) -> np.ndarray:
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    lam = LAM
    for cv_p, bw_p, lam_i in stages:
        lam = float(lam_i) if float(lam_i) > 0 else LAM
        p_acc += density_f32(grid_xy, np.load(cv_p), np.load(bw_p))
    p = np.clip(p_acc / float(len(stages)), 1e-300, None)
    f2 = relative_fes((-np.log(p) / (BETA0 * lam) * EV_TO_KCAL).reshape(g1_shape))
    p2 = p.reshape(g1_shape)
    f2 = f2.copy()
    f2[p2 < P_FLOOR * float(np.max(p2))] = np.nan
    return f2


def worker_init(shared: dict) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _W.clear()
    _W.update(shared)


def kde_job(payload: dict):
    d1 = _W["d1"]
    d2 = _W["d2"]
    f2 = kde_fes_from_stages(
        payload["stages"],
        d1=d1,
        d2=d2,
        grid_xy=_W["grid_xy"],
        g1_shape=_W["g1_shape"],
    )
    _, f1d = project_1d_fixed_edges(d1, d2, f2, _W["edges"], temperature=T_K)
    xi = _W["xi"]
    f1d_ref = _W["f1d_ref"]
    rmse = rmse_1d(f1d, f1d_ref, vmax=VMAX_RMSE, xi=xi, xi_lo=None, xi_hi=None)
    rmse30 = rmse_1d(f1d, f1d_ref, vmax=VMAX30, xi=xi, xi_lo=None, xi_hi=None)
    dG = delta_g_from_2d(d1, d2, f2, temperature=T_K, xi_cut=0.0)
    kind = payload.get("kind", "group")
    if kind == "last100":
        return "last100", payload["ir"], f2, f1d, rmse, rmse30, dG
    return (
        "group",
        payload["ir"],
        payload["ig"],
        f1d,
        rmse,
        rmse30,
        dG,
        f2 if payload.get("keep_f2") else None,
    )
