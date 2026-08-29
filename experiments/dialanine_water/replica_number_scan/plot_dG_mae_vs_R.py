#!/usr/bin/env python3
"""ΔG MAE vs replica number R, matching the style of rmse_vs_lambda.png.

For each R and each seed, MAE = time-average of |ΔG(t) − ΔG_US| over 80–140 ps.
Then mean ± std across two seeds (prefer s0 and s2). Averaging |error| per seed
avoids cancellation of opposite-signed ΔG errors.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from plot_fes_compare import nature_rcparams  # noqa: E402
from plot_fes_rmse_vs_time import delta_g_two_basins  # noqa: E402

SFPT = ROOT / "experiments" / "dialanine_water" / "sFPT"
OUT = ROOT / "experiments" / "dialanine_water" / "replica_number_scan"

R_NPZ = {
    64: SFPT / "fair_compete_result" / "convergence_64" / "fes_1d_all_groups.npz",
    128: SFPT / "fair_compete_result" / "convergence_128" / "fes_1d_all_groups.npz",
    320: SFPT / "convergence_320" / "fes_1d_all_groups.npz",
    512: SFPT / "convergence_512" / "fes_1d_all_groups.npz",
    1024: SFPT / "convergence_1024" / "fes_1d_all_groups.npz",
}
R_META = {
    64: SFPT / "fair_compete_result" / "convergence_64" / "fes_rmse_vs_time.npz",
    128: SFPT / "fair_compete_result" / "convergence_128" / "fes_rmse_vs_time.npz",
    320: SFPT / "convergence_320" / "fes_rmse_vs_time.npz",
    512: SFPT / "convergence_512" / "fes_rmse_vs_time.npz",
    1024: SFPT / "convergence_1024" / "fes_rmse_vs_time.npz",
}
COLORS = {
    64: "#66a61e",
    128: "#e7298a",
    320: "#1b9e77",
    512: "#d95f02",
    1024: "#7570b3",
}
PANEL_A_R = (320, 512, 1024)
T_LO = 80.0
T_HI = 140.0
T_K = 300.0
SEED_RE = re.compile(r"_s(\d+)_")


def seed_of(path: str) -> int | None:
    m = SEED_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def pick_two(seeds: list[int | None]) -> list[int]:
    preferred = [0, 2]
    idx = [i for s in preferred for i, si in enumerate(seeds) if si == s]
    seen: list[int] = []
    for i in idx:
        if i not in seen:
            seen.append(i)
        if len(seen) == 2:
            return seen
    for i in range(len(seeds)):
        if i not in seen:
            seen.append(i)
        if len(seen) == 2:
            return seen
    return seen


def dG_series(xi: np.ndarray, f_rows: np.ndarray) -> np.ndarray:
    out = np.empty(f_rows.shape[0], dtype=np.float64)
    for i, f in enumerate(f_rows):
        out[i] = delta_g_two_basins(
            xi, f, temperature=T_K, xi_cut=0.0, xi_lo=-np.pi, xi_hi=np.pi
        )
    return out


def mae_from_npz(
    r_npz: dict[int, Path],
    r_meta: dict[int, Path],
) -> tuple[list[int], np.ndarray, np.ndarray, dict[int, dict]]:
    Rs = [
        R
        for R in sorted(r_npz)
        if Path(r_npz[R]).is_file() and Path(r_meta[R]).is_file()
    ]
    if not Rs:
        raise RuntimeError("no replica-scan npz found")
    mae_mean: list[float] = []
    mae_std: list[float] = []
    per_r: dict[int, dict] = {}
    for R in Rs:
        g1d = np.load(r_npz[R])
        meta = np.load(r_meta[R])
        xi = np.asarray(g1d["xi"], dtype=np.float64)
        t = np.asarray(g1d["t_ps"], dtype=np.float64)
        runs = np.asarray(g1d["F1d_kde_runs"], dtype=np.float64)
        dG_us = float(meta["dG_us_ref"])
        dirs = [str(x) for x in meta["fpt_dirs"]]
        seeds = [seed_of(d) for d in dirs]
        take = pick_two(seeds)
        sel = (t >= T_LO) & (t <= T_HI)
        if not np.any(sel):
            raise RuntimeError(f"R={R}: no samples in [{T_LO}, {T_HI}] ps")
        maes = []
        labels = []
        for i in take:
            dG = dG_series(xi, runs[i])
            mae = float(np.mean(np.abs(dG[sel] - dG_us)))
            maes.append(mae)
            labels.append((seeds[i], Path(dirs[i]).name, mae))
        maes_a = np.asarray(maes, dtype=np.float64)
        mu = float(np.mean(maes_a))
        sd = float(np.std(maes_a, ddof=1)) if maes_a.size > 1 else 0.0
        mae_mean.append(mu)
        mae_std.append(sd)
        per_r[R] = {
            "seeds": [s for s, _, _ in labels],
            "mae_seeds": maes_a,
            "n_t": int(np.sum(sel)),
            "t_lo": float(t[sel][0]),
            "t_hi": float(t[sel][-1]),
            "dG_us": dG_us,
        }
        print(
            f"[R={R}] window {t[sel][0]:.0f}–{t[sel][-1]:.0f} ps  n={int(np.sum(sel))}  "
            + "  ".join(f"s{s} MAE={m:.3f}" for s, _, m in labels)
            + f"  mean={mu:.3f}  std={sd:.3f}",
            flush=True,
        )
        g1d.close()
        meta.close()
    return (
        Rs,
        np.asarray(mae_mean, dtype=np.float64),
        np.asarray(mae_std, dtype=np.float64),
        per_r,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Rs, y, e, per_r = mae_from_npz(R_NPZ, R_META)
    Rs_a = np.asarray(Rs, dtype=np.float64)

    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.errorbar(
        Rs_a,
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
    for x, yy in zip(Rs_a, y):
        ax.plot(x, yy, "o", color=COLORS.get(int(x), "C0"), ms=6.5, zorder=3)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$\Delta G$ MAE vs US / kcal mol$^{-1}$")
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(list(Rs))
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    png = OUT / "dG_mae_vs_R.png"
    pdf = OUT / "dG_mae_vs_R.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)

    kw = {
        "R": Rs_a,
        "mae_mean": y,
        "mae_std": e,
        "t_lo": T_LO,
        "t_hi": T_HI,
        "n_seeds": 2,
    }
    for R in Rs:
        kw[f"mae_R{R}"] = per_r[R]["mae_seeds"]
        kw[f"seeds_R{R}"] = np.asarray(per_r[R]["seeds"], dtype=np.int64)
    np.savez_compressed(OUT / "dG_mae_vs_R.npz", **kw)
    print(f"[ok] {png}", flush=True)


if __name__ == "__main__":
    main()
