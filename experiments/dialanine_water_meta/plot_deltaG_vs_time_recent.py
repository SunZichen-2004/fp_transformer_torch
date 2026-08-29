#!/usr/bin/env python3
"""Overlay WT-MetaD ΔG(φ) vs simulation time for recent dialanine-water jobs.

Hills are binned onto a torus grid and convolved with a periodic Gaussian via
FFT (equivalent to summing WT hills, vectorized). ΔG uses the same two-basin
definition as the convergence scripts: kT ln(P_{φ<0}/P_{φ>0}).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

JOBS = [
    {
        "job": 89387,
        "R": 64,
        "seed": 0,
        "dir": RESULTS / "dialanine_water_metad_r64_s0_89387",
        "color": "#1b9e77",
        "ls": "-",
        "note": "",
    },
    {
        "job": 89392,
        "R": 64,
        "seed": 2,
        "dir": RESULTS / "dialanine_water_metad_r64_s2_89392",
        "color": "#66c2a5",
        "ls": "--",
        "note": "",
    },
    {
        "job": 89388,
        "R": 128,
        "seed": 0,
        "dir": RESULTS / "dialanine_water_metad_r128_s0_89388",
        "color": "#d95f02",
        "ls": "-",
        "note": "",
    },
    {
        "job": 89389,
        "R": 256,
        "seed": 0,
        "dir": RESULTS / "dialanine_water_metad_r256_s0_89389",
        "color": "#7570b3",
        "ls": "-",
        "note": "FAILED ~100 ps",
    },
]

PACE = 50
DT_PS = 0.001
SIGMA = (0.35, 0.35)
GAMMA = 15.0
T_K = 300.0
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
KT_KCAL = KB_EV_PER_K * T_K * EV_TO_KCAL
TWO_PI = 2.0 * np.pi
GRID_N = 256
STRIDE_DEP = 20  # 1 ps


def label_of(job: dict) -> str:
    lab = f"R={job['R']}  seed={job['seed']}  ({job['job']})"
    if job["note"]:
        lab += f"  [{job['note']}]"
    return lab


def periodic_gaussian_kernel(n: int, sigma: tuple[float, float]) -> np.ndarray:
    dx = TWO_PI / float(n)
    idx = np.arange(n, dtype=np.float64)
    idx[idx > n / 2] -= n
    kx = idx * dx
    ky = idx * dx
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.exp(-0.5 * ((KX / sigma[0]) ** 2 + (KY / sigma[1]) ** 2))


def delta_g_from_V(V: np.ndarray, phi: np.ndarray) -> float:
    """ΔG = kT ln(P_{φ<0}/P_{φ>0}) from bias V (eV) on (nφ, nψ) grid."""
    scale = GAMMA / (GAMMA - 1.0)
    F = (-scale * V) * EV_TO_KCAL
    left = phi < 0.0
    right = phi > 0.0
    w = -F / KT_KCAL
    w -= np.max(w)
    e = np.exp(w)
    z1 = float(e[left, :].sum())
    z2 = float(e[right, :].sum())
    if z1 <= 0.0 or z2 <= 0.0:
        return float("nan")
    return float(KT_KCAL * np.log(z1 / z2))


def incremental_dg(hills_path: Path, *, n_replicas: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(hills_path)
    n_hills = int(z["n_hills"]) if "n_hills" in z.files else int(z["centers"].shape[0])
    centers = np.asarray(z["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(z["heights"][:n_hills], dtype=np.float64)
    n_dep = n_hills // int(n_replicas)
    print(
        f"  hills={n_hills}  n_dep={n_dep}  t={n_dep * PACE * DT_PS:.1f} ps",
        flush=True,
    )

    n = GRID_N
    dx = TWO_PI / float(n)
    phi = -np.pi + (np.arange(n) + 0.5) * dx
    ix = np.floor((centers[:, 0] + np.pi) / dx).astype(np.int64) % n
    iy = np.floor((centers[:, 1] + np.pi) / dx).astype(np.int64) % n
    lin = ix * n + iy

    ker_fft = np.fft.rfft2(periodic_gaussian_kernel(n, SIGMA))
    H = np.zeros(n * n, dtype=np.float64)

    ks = list(range(STRIDE_DEP, n_dep + 1, STRIDE_DEP))
    if not ks or ks[-1] != n_dep:
        ks.append(n_dep)

    t_list = np.empty(len(ks), dtype=np.float64)
    g_list = np.empty(len(ks), dtype=np.float64)
    i_prev = 0
    for j, k in enumerate(ks):
        i1 = k * int(n_replicas)
        H += np.bincount(lin[i_prev:i1], weights=heights[i_prev:i1], minlength=n * n)
        i_prev = i1
        V = np.fft.irfft2(np.fft.rfft2(H.reshape(n, n)) * ker_fft, s=(n, n)).real
        t_list[j] = float(k) * PACE * DT_PS
        g_list[j] = delta_g_from_V(V, phi)
        if j == 0 or j + 1 == len(ks) or (j + 1) % 50 == 0:
            print(f"    t={t_list[j]:.1f} ps  ΔG={g_list[j]:.3f}", flush=True)
    return t_list, g_list


def _rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }


COLOR_US = "#2ca02c"
US_MBAR = (
    Path(__file__).resolve().parents[1]
    / "dialanine_water_us"
    / "results"
    / "dialanine_water_us2d_r576_k100_s0_87367"
    / "mbar"
    / "fes_mbar.npz"
)


def delta_g_from_f1d(xi: np.ndarray, f_kcal: np.ndarray) -> float:
    m = np.isfinite(xi) & np.isfinite(f_kcal)
    xi_a = np.asarray(xi, dtype=np.float64)[m]
    f = np.asarray(f_kcal, dtype=np.float64)[m]

    def _logZ(sel: np.ndarray) -> float:
        vals = -f[sel] / KT_KCAL
        mval = float(np.max(vals))
        return mval + float(np.log(np.sum(np.exp(vals - mval))))

    return float(KT_KCAL * (_logZ(xi_a < 0.0) - _logZ(xi_a > 0.0)))


def load_us_dg(path: Path) -> float:
    z = np.load(path)
    return delta_g_from_f1d(z["xi_phi"], z["F_phi"])


def plot_overlay(series, *, ylim, png: Path, pdf: Path, dG_ref: float | None = None) -> None:
    plt.rcParams.update(_rc())
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.axhline(0.0, color="0.55", ls="--", lw=0.8, zorder=0)
    if dG_ref is not None:
        ax.axhspan(dG_ref - 1.0, dG_ref + 1.0, color=COLOR_US, alpha=0.15, lw=0, zorder=0)
        ax.axhline(
            dG_ref,
            color=COLOR_US,
            ls=":",
            lw=1.3,
            zorder=1,
            label=rf"US-MBAR $\Delta G$={dG_ref:.2f}",
        )
    for job, t, dG in series:
        ax.plot(t, dG, color=job["color"], ls=job["ls"], lw=1.6, label=label_of(job), zorder=2)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$\Delta G=k_BT\ln(P_{\phi<0}/P_{\phi>0})$ / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)


def series_from_npz(path: Path):
    z = np.load(path)
    series = []
    by_job = {int(j["job"]): j for j in JOBS}
    for job in JOBS:
        tag = f"j{job['job']}"
        if f"t_{tag}" not in z.files:
            raise KeyError(f"missing t_{tag} in {path}")
        series.append((by_job[int(job["job"])], z[f"t_{tag}"], z[f"dG_{tag}"]))
    return series


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="reuse cached ΔG(t) instead of rebuilding from hills",
    )
    args = p.parse_args()

    dG_us = load_us_dg(US_MBAR)
    print(f"US-MBAR ΔG (1D F_phi) = {dG_us:.3f} kcal/mol  [{US_MBAR}]", flush=True)

    cache = RESULTS / "deltaG_vs_time_89387_89388_89389_89392.npz"
    if args.from_npz is not None:
        series = series_from_npz(args.from_npz)
    else:
        series = []
        for job in JOBS:
            hills = job["dir"] / "live" / "hills.npz"
            if not hills.is_file():
                raise FileNotFoundError(hills)
            print(f"[{job['job']}] {hills}", flush=True)
            t, dG = incremental_dg(hills, n_replicas=int(job["R"]))
            series.append((job, t, dG))

    png = RESULTS / "deltaG_vs_time_89387_89388_89389_89392.png"
    pdf = RESULTS / "deltaG_vs_time_89387_89388_89389_89392.pdf"
    plot_overlay(series, ylim=None, png=png, pdf=pdf, dG_ref=dG_us)
    png_z = RESULTS / "deltaG_vs_time_89387_89388_89389_89392_divein.png"
    plot_overlay(
        series,
        ylim=(2.0, 8.0),
        png=png_z,
        pdf=RESULTS / "deltaG_vs_time_89387_89388_89389_89392_divein.pdf",
        dG_ref=dG_us,
    )

    kw = {"dG_us": np.float64(dG_us)}
    for job, t, dG in series:
        tag = f"j{job['job']}"
        kw[f"t_{tag}"] = t
        kw[f"dG_{tag}"] = dG
        kw[f"R_{tag}"] = job["R"]
        kw[f"seed_{tag}"] = job["seed"]
    np.savez_compressed(cache, **kw)
    print("wrote", png)
    print("wrote", png_z)
    for job, t, dG in series:
        print(
            f"  {job['job']}: t_end={t[-1]:.1f} ps  ΔG_end={dG[-1]:.3f}  "
            f"ΔG_last10mean={dG[-max(1, len(dG)//10):].mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
