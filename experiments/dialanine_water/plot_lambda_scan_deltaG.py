#!/usr/bin/env python3
"""λ-scan ΔG and |ΔG−ΔG_US| vs time for dialanine_water sFPT.

For each λ, average two seeds (mean±std) and overlay curves.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DIAL = ROOT / "experiments" / "dialanine"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(DIAL))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    DIAL / "plot_convergence_analysis.py",
)
_dca = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dca)

from plot_fes_compare import nature_rcparams, relative_fes  # noqa: E402
from plot_fes_rmse_vs_time import delta_g_two_basins  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import COLOR_US  # noqa: E402

EV_TO_KCAL = _dca.EV_TO_KCAL
LAM_RE = re.compile(r"lam([0-9.]+)_", re.I)

# Colorblind-friendly palette for λ curves
LAM_COLORS = {
    0.02: "#1b9e77",
    0.04: "#d95f02",
    0.06: "#7570b3",
    0.08: "#e7298a",
    0.10: "#66a61e",
    0.1: "#66a61e",
}


def mean_std(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=0)
    if stack.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=0, ddof=1)


def load_us_f_phi(path: Path, xi_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if "xi_phi" in z.files and "F_phi" in z.files:
        xi = np.asarray(z["xi_phi"], dtype=np.float64)
        f = relative_fes(np.asarray(z["F_phi"], dtype=np.float64), None)
    else:
        xi = np.asarray(z["xi"], dtype=np.float64)
        f = relative_fes(np.asarray(z["F_kcal"], dtype=np.float64), None)
    if len(xi) != len(xi_query) or not np.allclose(xi, xi_query):
        f = np.interp(xi_query, xi, f, left=np.nan, right=np.nan)
        f = relative_fes(f, None)
        xi = np.asarray(xi_query, dtype=np.float64)
    return xi, f


def discover_runs(scan_dir: Path) -> dict[float, list[Path]]:
    """Map λ → list of result dirs."""
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-dir",
        type=Path,
        default=ROOT / "experiments/dialanine_water/lambda_scan",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water/lambda_scan/convergence_analysis",
    )
    p.add_argument(
        "--us-mbar",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water_us/results"
        / "dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz",
    )
    p.add_argument("--group-size", type=int, default=50)
    p.add_argument("--dt-group", type=float, default=5.0)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-np.pi)
    p.add_argument("--xi-hi", type=float, default=np.pi)
    p.add_argument("--xlim", type=str, default="0,0", help="x limits; 0,0 = auto")
    p.add_argument(
        "--ylim",
        type=str,
        default="",
        help="y limits for ΔG plots; empty = auto (no set_ylim)",
    )
    p.add_argument(
        "--ylim-abs",
        type=str,
        default="",
        help="y limits for |ΔG−ΔG_US| plots; empty = auto",
    )
    args = p.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    us_path = Path(args.us_mbar).resolve()
    if not us_path.is_file():
        raise FileNotFoundError(us_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda → cpu", flush=True)

    phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    # dummy 2d to get xi from project (or just linspace bins)
    xi = np.linspace(-np.pi, np.pi, int(args.n_bins_1d), endpoint=False)
    xi = xi + 0.5 * (xi[1] - xi[0])
    _, f1d_ref = load_us_f_phi(us_path, xi)
    dG_ref = delta_g_two_basins(
        xi,
        f1d_ref,
        temperature=float(args.temperature),
        xi_cut=float(args.xi_cut),
        xi_lo=float(args.xi_lo),
        xi_hi=float(args.xi_hi),
    )
    print(f"[us] ΔG_ref={dG_ref:.4f} kcal/mol", flush=True)

    runs = discover_runs(Path(args.scan_dir))
    if not runs:
        raise RuntimeError(f"no lam runs under {args.scan_dir}")
    print(f"[scan] λs={sorted(runs)}", flush=True)

    series: dict[float, dict] = {}
    for lam in sorted(runs):
        dirs = runs[lam]
        abs_list = []
        dG_list = []
        t_ref = None
        n_min = None
        for d in dirs:
            n_stages = len(
                _dca.discover_dihedral_stages(
                    d,
                    lambda_target=float(lam),
                    lambda_atol=float(args.lambda_atol),
                )
            )
            n_groups = n_stages // int(args.group_size)
            if n_groups < 1:
                print(f"[skip] {d.name}: stages={n_stages} < group", flush=True)
                continue
            if n_min is None:
                n_min = n_groups
            else:
                n_min = min(n_min, n_groups)
        if n_min is None or n_min < 1:
            print(f"[skip] λ={lam}: no complete groups", flush=True)
            continue

        for d in dirs:
            t_s, _, dG_s, _, _ = _dca.kde_groups_phi(
                d,
                phi=phi,
                psi=psi,
                f1d_ref=f1d_ref,
                xi=xi,
                group_size=int(args.group_size),
                dt_group=float(args.dt_group),
                temperature=float(args.temperature),
                vmax=None,
                device=device,
                chunk=int(args.chunk),
                n_bins=int(args.n_bins_1d),
                xi_cut=float(args.xi_cut),
                xi_lo=float(args.xi_lo),
                xi_hi=float(args.xi_hi),
                max_groups=int(n_min),
                lambda_target=float(lam),
                lambda_atol=float(args.lambda_atol),
            )
            abs_err = np.abs(np.asarray(dG_s, dtype=np.float64) - dG_ref)
            abs_list.append(abs_err)
            dG_list.append(np.asarray(dG_s, dtype=np.float64))
            if t_ref is None:
                t_ref = t_s
            print(
                f"[kde] λ={lam:g} {d.name} groups={len(t_s)} "
                f"|ΔG−US|_end={abs_err[-1]:.3f} ΔG_end={dG_s[-1]:.3f}",
                flush=True,
            )
        stack = np.stack(abs_list, axis=0)
        mean, std = mean_std(stack)
        dG_mean, dG_std = mean_std(np.stack(dG_list, axis=0))
        series[lam] = {
            "t": t_ref,
            "abs": mean,
            "abs_std": std,
            "dG": dG_mean,
            "dG_std": dG_std,
            "runs": len(abs_list),
        }
        print(
            f"[λ={lam:g}] n_seeds={len(abs_list)} groups={len(t_ref)} "
            f"|ΔG−US|_end={mean[-1]:.3f}±{std[-1]:.3f}",
            flush=True,
        )

    save_kw: dict = {
        "dG_us_ref": dG_ref,
        "group_size": np.int64(args.group_size),
        "dt_group": np.float64(args.dt_group),
        "lambdas": np.asarray(sorted(series.keys()), dtype=np.float64),
    }
    for lam in series:
        tag = f"{lam:g}".replace(".", "p")
        save_kw[f"t_lam{tag}"] = series[lam]["t"]
        save_kw[f"abs_lam{tag}"] = series[lam]["abs"]
        save_kw[f"abs_std_lam{tag}"] = series[lam]["abs_std"]
        save_kw[f"dG_lam{tag}"] = series[lam]["dG"]
        save_kw[f"dG_std_lam{tag}"] = series[lam]["dG_std"]
    np.savez_compressed(out / "deltaG_vs_time_by_lambda.npz", **save_kw)

    xlim_raw = [float(x) for x in args.xlim.replace(";", ",").split(",") if x.strip()]
    if len(xlim_raw) == 2 and xlim_raw[0] == 0.0 and xlim_raw[1] == 0.0:
        t_max = max(float(np.max(s["t"])) for s in series.values()) if series else 50.0
        xlim = (0.0, t_max)
    else:
        xlim = tuple(xlim_raw)

    def _parse_ylim(s: str) -> tuple[float, float] | None:
        s = (s or "").strip()
        if not s:
            return None
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
        return (vals[0], vals[1]) if len(vals) == 2 else None

    ylim_dg = _parse_ylim(args.ylim)
    ylim_abs = _parse_ylim(args.ylim_abs)

    ylabel_dg = (
        rf"$\Delta G=k_BT\ln(P_{{\phi<0}}/P_{{\phi>0}})$ / kcal mol$^{{-1}}$"
    )
    ylabel_abs = r"$|\Delta G-\Delta G_{\mathrm{US}}|$ / kcal mol$^{-1}$"

    def _plot_dg(ax, *, markers: bool) -> None:
        ax.axhspan(
            dG_ref - 1.0,
            dG_ref + 1.0,
            color=COLOR_US,
            alpha=0.18,
            lw=0,
            zorder=0,
            label=r"final $\Delta G\pm 1$",
        )
        ax.axhline(
            dG_ref,
            color=COLOR_US,
            ls=":",
            lw=1.2,
            zorder=1,
            label=rf"US $\Delta G$={dG_ref:.2f}",
        )
        for lam in sorted(series):
            s = series[lam]
            c = LAM_COLORS.get(lam, LAM_COLORS.get(round(lam, 2), "C0"))
            kw = dict(color=c, lw=1.6 if markers else 1.5, zorder=3)
            if markers:
                kw.update(marker="o", ms=3.5)
                lab = rf"$\lambda$={lam:g} (n={s['runs']})"
            else:
                lab = rf"$\lambda$={lam:g}"
            ax.plot(s["t"], s["dG"], label=lab, **kw)
            if np.any(s["dG_std"] > 0):
                ax.fill_between(
                    s["t"],
                    s["dG"] - s["dG_std"],
                    s["dG"] + s["dG_std"],
                    color=c,
                    alpha=0.20 if markers else 0.18,
                    lw=0,
                    zorder=2,
                )

    def _plot_abs(ax, *, markers: bool) -> None:
        ax.axhline(0.0, color="0.5", ls=":", lw=0.9, zorder=1)
        for lam in sorted(series):
            s = series[lam]
            c = LAM_COLORS.get(lam, LAM_COLORS.get(round(lam, 2), "C0"))
            kw = dict(color=c, lw=1.6 if markers else 1.5, zorder=3)
            if markers:
                kw.update(marker="o", ms=3.5)
                lab = rf"$\lambda$={lam:g} (n={s['runs']})"
            else:
                lab = rf"$\lambda$={lam:g}"
            ax.plot(s["t"], s["abs"], label=lab, **kw)
            if np.any(s["abs_std"] > 0):
                ax.fill_between(
                    s["t"],
                    s["abs"] - s["abs_std"],
                    s["abs"] + s["abs_std"],
                    color=c,
                    alpha=0.20 if markers else 0.18,
                    lw=0,
                    zorder=2,
                )

    plt.rcParams.update(nature_rcparams())

    # --- ΔG dive-in ---
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _plot_dg(ax, markers=True)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(ylabel_dg)
    ax.set_xlim(*xlim)
    if ylim_dg is not None:
        ax.set_ylim(*ylim_dg)
    ax.legend(frameon=False, fontsize=7.5, loc="best", ncol=1)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    for stem in ("deltaG_vs_time_divein", "deltaG_vs_time_by_lambda_divein"):
        fig.savefig(out / f"{stem}.png", dpi=200)
        fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)

    # --- ΔG full ---
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _plot_dg(ax, markers=False)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(ylabel_dg)
    ax.set_xlim(left=0.0)
    if ylim_dg is not None:
        ax.set_ylim(*ylim_dg)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "deltaG_vs_time_by_lambda.png", dpi=200)
    fig.savefig(out / "deltaG_vs_time_by_lambda.pdf")
    plt.close(fig)

    # --- |ΔG−ΔG_US| dive-in ---
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _plot_abs(ax, markers=True)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(ylabel_abs)
    ax.set_xlim(*xlim)
    if ylim_abs is not None:
        ax.set_ylim(*ylim_abs)
    ax.legend(frameon=False, fontsize=7.5, loc="best", ncol=1)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    for stem in (
        "abs_deltaG_vs_time_divein",
        "abs_deltaG_vs_time_by_lambda_divein",
    ):
        fig.savefig(out / f"{stem}.png", dpi=200)
        fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)

    # --- |ΔG−ΔG_US| full ---
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _plot_abs(ax, markers=False)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(ylabel_abs)
    ax.set_xlim(left=0.0)
    if ylim_abs is not None:
        ax.set_ylim(*ylim_abs)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "abs_deltaG_vs_time_by_lambda.png", dpi=200)
    fig.savefig(out / "abs_deltaG_vs_time_by_lambda.pdf")
    plt.close(fig)

    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
