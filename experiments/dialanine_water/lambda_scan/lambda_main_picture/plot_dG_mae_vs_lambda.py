#!/usr/bin/env python3
"""ΔG MAE vs λ, same protocol as replica_number_scan/dG_mae_vs_R.png.

For each λ and each seed, MAE = time-average of |ΔG(t) − ΔG_US| over 80–140 ps.
Then mean ± std across two seeds (s0 and s1).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "dialanine_water"
DIAL = ROOT / "experiments" / "dialanine"
SCAN = EXP / "lambda_scan"
OUT = SCAN / "lambda_main_picture"
CACHE = OUT / "dG_mae_vs_lambda.npz"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(DIAL))
sys.path.insert(0, str(EXP))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    DIAL / "plot_convergence_analysis.py",
)
_dca = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dca)

from plot_fes_compare import nature_rcparams, relative_fes  # noqa: E402
from plot_fes_rmse_vs_time import delta_g_two_basins  # noqa: E402
from plot_lambda_scan_fes_rmse import LAM_COLORS  # noqa: E402

US_MBAR = (
    ROOT
    / "experiments"
    / "dialanine_water_us"
    / "results"
    / "dialanine_water_us2d_r576_k100_s0_87367"
    / "mbar"
    / "fes_mbar.npz"
)
LAM_RE = re.compile(r"lam([0-9.]+)_", re.I)
SEED_RE = re.compile(r"_s(\d+)_")
T_LO = 80.0
T_HI = 140.0
T_K = 300.0
GROUP_SIZE = 50
DT_GROUP = 5.0
GRID_N = 73
N_BINS = 72
LAMBDA_ATOL = 1e-3


def seed_of(path: Path) -> int | None:
    m = SEED_RE.search(path.name)
    return int(m.group(1)) if m else None


def pick_two(dirs: list[Path]) -> list[Path]:
    seeds = [seed_of(d) for d in dirs]
    preferred = [0, 1]
    chosen: list[Path] = []
    for s in preferred:
        for d, si in zip(dirs, seeds):
            if si == s and d not in chosen:
                chosen.append(d)
                break
        if len(chosen) == 2:
            return chosen
    for d in dirs:
        if d not in chosen:
            chosen.append(d)
        if len(chosen) == 2:
            return chosen
    return chosen


def discover_runs(scan_dir: Path) -> dict[float, list[Path]]:
    out: dict[float, list[Path]] = {}
    for d in sorted(scan_dir.glob("dialanine_water_sfpt_lam*_r*_s*_*/")):
        if not d.is_dir():
            continue
        m = LAM_RE.search(d.name)
        if not m:
            continue
        lam = float(m.group(1))
        out.setdefault(lam, []).append(d)
    return out


def load_us(xi: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.load(US_MBAR)
    xi_u = np.asarray(z["xi_phi"], dtype=np.float64)
    f = relative_fes(np.asarray(z["F_phi"], dtype=np.float64), None)
    if len(xi_u) != len(xi) or not np.allclose(xi_u, xi):
        f = relative_fes(np.interp(xi, xi_u, f, left=np.nan, right=np.nan), None)
    dG = delta_g_two_basins(
        xi, f, temperature=T_K, xi_cut=0.0, xi_lo=-np.pi, xi_hi=np.pi
    )
    return f, float(dG)


def compute_seed_dG(
    d: Path,
    *,
    lam: float,
    phi,
    psi,
    f1d_ref,
    xi,
    device,
    max_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_s, _, dG_s, _, _ = _dca.kde_groups_phi(
        d,
        phi=phi,
        psi=psi,
        f1d_ref=f1d_ref,
        xi=xi,
        group_size=GROUP_SIZE,
        dt_group=DT_GROUP,
        temperature=T_K,
        vmax=None,
        device=device,
        chunk=2048,
        n_bins=N_BINS,
        xi_cut=0.0,
        xi_lo=-np.pi,
        xi_hi=np.pi,
        max_groups=max_groups,
        lambda_target=float(lam),
        lambda_atol=LAMBDA_ATOL,
    )
    return np.asarray(t_s, dtype=np.float64), np.asarray(dG_s, dtype=np.float64)


def plot_mae(lams, y, e) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.errorbar(
        lams,
        y,
        yerr=e,
        fmt="o-",
        color="0.25",
        ecolor="0.45",
        elinewidth=1.0,
        capsize=3,
        lw=1.4,
        ms=5.5,
        zorder=2,
    )
    for x, yy in zip(lams, y):
        c = LAM_COLORS.get(float(x), LAM_COLORS.get(round(float(x), 2), "C0"))
        ax.plot(x, yy, "o", color=c, ms=6.5, zorder=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta G$ MAE vs US / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    png = OUT / "dG_mae_vs_lambda.png"
    pdf = OUT / "dG_mae_vs_lambda.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    phi = np.linspace(-np.pi, np.pi, GRID_N)
    psi = np.linspace(-np.pi, np.pi, GRID_N)
    xi = np.linspace(-np.pi, np.pi, N_BINS, endpoint=False)
    xi = xi + 0.5 * (xi[1] - xi[0])
    f1d_ref, dG_us = load_us(xi)
    print(f"[us] ΔG={dG_us:.4f}", flush=True)

    max_groups = int(np.ceil(T_HI / DT_GROUP))
    runs = discover_runs(SCAN)
    lams = sorted(runs)
    mae_mean = []
    mae_std = []
    save: dict = {
        "lambdas": np.asarray(lams, dtype=np.float64),
        "dG_us": np.float64(dG_us),
        "t_lo": T_LO,
        "t_hi": T_HI,
        "n_seeds": 2,
        "group_size": GROUP_SIZE,
        "dt_group": DT_GROUP,
    }

    for lam in lams:
        dirs = pick_two(runs[lam])
        maes = []
        seeds = []
        for d in dirs:
            print(f"[kde] λ={lam:g} {d.name}", flush=True)
            t, dG = compute_seed_dG(
                d,
                lam=lam,
                phi=phi,
                psi=psi,
                f1d_ref=f1d_ref,
                xi=xi,
                device=device,
                max_groups=max_groups,
            )
            sel = (t >= T_LO) & (t <= T_HI)
            if not np.any(sel):
                raise RuntimeError(f"λ={lam}: no samples in [{T_LO}, {T_HI}]")
            mae = float(np.mean(np.abs(dG[sel] - dG_us)))
            s = seed_of(d)
            maes.append(mae)
            seeds.append(-1 if s is None else s)
            tag = f"{lam:g}".replace(".", "p")
            save[f"t_lam{tag}_s{s}"] = t
            save[f"dG_lam{tag}_s{s}"] = dG
            print(
                f"    s{s} t={t[sel][0]:.0f}–{t[sel][-1]:.0f} ps  n={int(np.sum(sel))}  MAE={mae:.3f}",
                flush=True,
            )
        maes_a = np.asarray(maes, dtype=np.float64)
        mu = float(np.mean(maes_a))
        sd = float(np.std(maes_a, ddof=1)) if maes_a.size > 1 else 0.0
        mae_mean.append(mu)
        mae_std.append(sd)
        save[f"mae_lam{str(lam).replace('.', 'p')}"] = maes_a
        save[f"seeds_lam{str(lam).replace('.', 'p')}"] = np.asarray(seeds, dtype=np.int64)
        print(f"[λ={lam:g}] mean={mu:.3f}  std={sd:.3f}", flush=True)

    y = np.asarray(mae_mean, dtype=np.float64)
    e = np.asarray(mae_std, dtype=np.float64)
    save["mae_mean"] = y
    save["mae_std"] = e
    np.savez_compressed(CACHE, **save)
    plot_mae(np.asarray(lams, dtype=np.float64), y, e)


if __name__ == "__main__":
    main()
