#!/usr/bin/env python3
"""Flattening-law check: F_eff,λ = −kT ln P vs F_US on the native (φ,ψ) grid.

Last-50% torus-KDE is the same construction as ``fes_2d_lam*.png``.
At each grid point with F_US ≤ ``f-cut``:

    y = F_eff,λ = −k_B T ln P(φ,ψ)     (mean ± std over seeds)
    x = F_US(φ,ψ)

Theory: y = [λ/(1+λ)] x + C.  Linear fit is compared to that slope.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DIAL = EXP.parent / "dialanine"
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

from plot_fes_compare import nature_rcparams  # noqa: E402

from plot_lambda_scan_fes_rmse import (  # noqa: E402
    KB_EV_PER_K,
    EV_TO_KCAL,
    LAM_COLORS,
    discover_runs,
    load_us_bundle,
    mean_std,
)

XLABEL = r"$F_{\mathrm{US}}(x)$ / kcal/mol"
YLABEL = r"$F_{\mathrm{eff},\lambda}=-k_BT\ln P$ / kcal/mol"


def lam_tag(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def linregress_xy(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
    """Return slope, intercept, R², slope stderr (unweighted)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = float(len(x))
    xm = np.mean(x)
    ym = np.mean(y)
    dx = x - xm
    dy = y - ym
    sxx = float(np.dot(dx, dx))
    sxy = float(np.dot(dx, dy))
    syy = float(np.dot(dy, dy))
    slope = sxy / sxx if sxx > 0 else float("nan")
    intercept = ym - slope * xm
    r2 = (sxy * sxy) / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    resid = y - (slope * x + intercept)
    dof = max(n - 2.0, 1.0)
    se = float(np.sqrt(np.sum(resid**2) / dof / sxx)) if sxx > 0 else float("nan")
    return float(slope), float(intercept), float(r2), se


def theory_intercept(x: np.ndarray, y: np.ndarray, slope_th: float) -> float:
    """Least-squares intercept with slope fixed to theory."""
    return float(np.mean(y) - float(slope_th) * np.mean(x))


def plot_one_lambda(
    out: Path,
    *,
    stem: str,
    lam: float,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    slope: float,
    intercept: float,
    r2: float,
    se: float,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    c = LAM_COLORS.get(float(lam), LAM_COLORS.get(round(float(lam), 2), "C0"))
    slope_th = float(lam) / (1.0 + float(lam))
    c_th = theory_intercept(x, y, slope_th)
    xmin = 0.0
    xmax = max(float(np.nanmax(x)) * 1.04, 1.0)
    xline = np.linspace(xmin, xmax, 200)
    ax.errorbar(
        x,
        y,
        yerr=np.clip(yerr, 0.0, None),
        fmt="o",
        ms=3.2,
        color=c,
        ecolor=c,
        elinewidth=0.6,
        capsize=0,
        alpha=0.75,
        zorder=3,
        label=rf"$n={len(x)}$",
    )
    ax.plot(
        xline,
        slope * xline + intercept,
        color="0.15",
        lw=1.6,
        zorder=4,
        label=rf"fit $a={slope:.4f}\pm{se:.4f}$",
    )
    ax.plot(
        xline,
        slope_th * xline + c_th,
        color="0.15",
        lw=1.4,
        ls="--",
        zorder=4,
        label=rf"theory $\lambda/(1+\lambda)={slope_th:.4f}$",
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xlim(xmin, xmax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(rf"$\lambda={lam:g}$  $R^2={r2:.3f}$", fontsize=11)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def plot_all_overlay(
    out: Path,
    *,
    results: list[dict],
) -> None:
    """Overlay every λ scatter + fit + theory on one axes."""
    from matplotlib.lines import Line2D

    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    xmax = max(float(np.nanmax(r["x"])) for r in results)
    xmin, xmax = 0.0, max(xmax * 1.04, 1.0)
    xline = np.linspace(xmin, xmax, 200)
    for res in results:
        lam = float(res["lam"])
        x, y = res["x"], res["y"]
        c = LAM_COLORS.get(lam, LAM_COLORS.get(round(lam, 2), "C0"))
        slope_th = lam / (1.0 + lam)
        c_th = theory_intercept(x, y, slope_th)
        ax.scatter(
            x,
            y,
            s=5,
            color=c,
            alpha=0.22,
            linewidths=0,
            zorder=2,
        )
        ax.plot(
            xline,
            res["slope"] * xline + res["intercept"],
            color=c,
            lw=1.6,
            zorder=4,
            label=rf"$\lambda={lam:g}$",
        )
        ax.plot(
            xline,
            slope_th * xline + c_th,
            color=c,
            lw=1.2,
            ls="--",
            zorder=3,
        )
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="0.25", lw=1.6, ls="-", label="fit"),
            Line2D(
                [0],
                [0],
                color="0.25",
                lw=1.2,
                ls="--",
                label=r"theory $\lambda/(1+\lambda)$",
            ),
        ]
    )
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=1)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "flatten_xy_all.png", dpi=200)
    fig.savefig(out / "flatten_xy_all.pdf")
    plt.close(fig)
    print("[ok] flatten_xy_all", flush=True)


def plot_slope_vs_lambda(out: Path, results: list[dict]) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    lams = np.array([r["lam"] for r in results], dtype=np.float64)
    sl = np.array([r["slope"] for r in results], dtype=np.float64)
    se = np.array([r["se"] for r in results], dtype=np.float64)
    lam_c = np.linspace(0.0, max(float(lams.max()) * 1.15, 0.12), 200)
    ax.plot(
        lam_c,
        lam_c / (1.0 + lam_c),
        color="0.25",
        ls="--",
        lw=1.5,
        label=r"theory $\lambda/(1+\lambda)$",
        zorder=2,
    )
    ax.errorbar(
        lams,
        sl,
        yerr=se,
        fmt="o",
        color="0.15",
        ecolor="0.4",
        elinewidth=1.0,
        capsize=3,
        ms=5.5,
        zorder=3,
        label="fit",
    )
    for lam, yy in zip(lams, sl):
        c = LAM_COLORS.get(float(lam), LAM_COLORS.get(round(float(lam), 2), "C0"))
        ax.plot(lam, yy, "o", color=c, ms=7.0, zorder=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"fitted slope $a$  ($y=ax+C$)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "flatten_slope_vs_lambda.png", dpi=200)
    fig.savefig(out / "flatten_slope_vs_lambda.pdf")
    plt.close(fig)
    print("[ok] flatten_slope_vs_lambda", flush=True)


def y_from_precomputed_f(
    f_kcal: np.ndarray, lam: float
) -> np.ndarray:
    """Last-50% KDE FES was stored as F = −kT/λ ln P (relative).

    The flattening-law observable is y = −kT ln P = λ F (+ seed-wise constant
    absorbed by the intercept C).
    """
    return float(lam) * np.asarray(f_kcal, dtype=np.float64)


def kde_negkTlnP_last_frac(
    fpt_dir: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    last_frac: float,
    temperature: float,
    device,
    chunk: int,
    lambda_target: float,
    lambda_atol: float,
) -> tuple[np.ndarray, int, int, int]:
    """Return −kT ln P on the (φ,ψ) mesh, last ``last_frac`` stages."""
    import torch

    rows = _dca.discover_dihedral_stages(
        fpt_dir,
        lambda_target=float(lambda_target),
        lambda_atol=float(lambda_atol),
    )
    if not rows:
        raise RuntimeError(f"no λ≈{lambda_target:g} stages in {fpt_dir}")
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    kT = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    for _, dih_p, bw_p, _ in use:
        p_acc += _dca.density_on_grid_torus(
            grid_xy, np.load(dih_p), np.load(bw_p), chunk=chunk, device=device
        )
    p = np.clip(p_acc / float(len(use)), 1e-300, None)
    y = (-kT * np.log(p)).reshape(g1.shape)
    return y, int(use[0][0]), int(use[-1][0]), int(n_keep)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water/lambda_scan/flatten_law_validation",
    )
    p.add_argument(
        "--fes-npz",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water/lambda_scan/convergence_analysis"
        / "fes_last50/fes_rmse_by_lambda.npz",
        help="Precomputed last-50% KDE FES (same as fes_2d_lam*.png).",
    )
    p.add_argument(
        "--scan-dir",
        type=Path,
        default=ROOT / "experiments/dialanine_water/lambda_scan",
    )
    p.add_argument(
        "--us-mbar",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water_us/results"
        / "dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz",
    )
    p.add_argument("--f-cut", type=float, default=16.0)
    p.add_argument("--last-frac", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument(
        "--recompute-kde",
        action="store_true",
        help="Evaluate −kT ln P on the native US (φ,ψ) grid from last-50% "
        "stages (otherwise convert precomputed FES).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    args = p.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    f_cut = float(args.f_cut)

    if args.recompute_kde:
        import torch

        us = load_us_bundle(Path(args.us_mbar).resolve())
        phi, psi, x_grid = us["phi"], us["psi"], us["F"]
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            print("[warn] cuda → cpu", flush=True)
        runs = discover_runs(Path(args.scan_dir))
        if not runs:
            raise RuntimeError(f"no lam runs under {args.scan_dir}")
        lams = sorted(runs)
        print(
            f"[kde] λs={lams}  grid={len(phi)}×{len(psi)}  F_cut={f_cut}",
            flush=True,
        )
        y_by_lam: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for lam in lams:
            ys = []
            for d in runs[lam]:
                y2, s0, s1, nk = kde_negkTlnP_last_frac(
                    d,
                    phi=phi,
                    psi=psi,
                    last_frac=float(args.last_frac),
                    temperature=float(args.temperature),
                    device=device,
                    chunk=int(args.chunk),
                    lambda_target=float(lam),
                    lambda_atol=float(args.lambda_atol),
                )
                ys.append(y2)
                print(
                    f"[kde] λ={lam:g} {d.name} stages={nk} ({s0}-{s1})",
                    flush=True,
                )
            ym, ys_ = mean_std(np.stack(ys, axis=0))
            y_by_lam[lam] = (ym, ys_)
    else:
        npz_path = Path(args.fes_npz).resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        z = np.load(npz_path)
        phi = np.asarray(z["phi"], dtype=np.float64)
        psi = np.asarray(z["psi"], dtype=np.float64)
        x_grid = np.asarray(z["F_us"], dtype=np.float64)
        lams = [float(v) for v in np.asarray(z["lambdas"], dtype=np.float64)]
        y_by_lam = {}
        for lam in lams:
            tag = lam_tag(lam)
            f2 = np.asarray(z[f"F2_lam{tag}"], dtype=np.float64)
            f2s = np.asarray(z[f"F2_std_lam{tag}"], dtype=np.float64)
            y_by_lam[lam] = (
                y_from_precomputed_f(f2, lam),
                y_from_precomputed_f(f2s, lam),
            )
        print(
            f"[npz] {npz_path.name}  λs={lams}  grid {len(phi)}×{len(psi)}  "
            f"F_cut={f_cut}",
            flush=True,
        )

    gphi, gpsi = np.meshgrid(phi, psi, indexing="xy")
    mask = np.isfinite(x_grid) & (x_grid <= f_cut)
    n_keep = int(np.sum(mask))
    print(
        f"[grid] {len(phi)}×{len(psi)}={x_grid.size}  "
        f"keep F_US≤{f_cut:g}: {n_keep}",
        flush=True,
    )

    results: list[dict] = []
    save: dict = {
        "phi": phi,
        "psi": psi,
        "gphi": gphi,
        "gpsi": gpsi,
        "F_us": x_grid,
        "mask": mask.astype(np.uint8),
        "n_grid": np.int64(len(phi)),
        "f_cut": np.float64(f_cut),
        "last_frac": np.float64(args.last_frac),
        "lambdas": np.asarray(lams, dtype=np.float64),
    }

    for lam in lams:
        y_m, y_s = y_by_lam[lam]
        m = mask & np.isfinite(y_m)
        x = x_grid[m]
        y = y_m[m]
        e = np.clip(y_s[m], 0.0, None)
        slope, intercept, r2, se = linregress_xy(x, y)
        slope_th = float(lam) / (1.0 + float(lam))
        c_th = theory_intercept(x, y, slope_th)
        print(
            f"[λ={lam:g}] n={len(x)}  a_fit={slope:.5f}±{se:.5f}  "
            f"a_th={slope_th:.5f}  C_fit={intercept:.4f}  "
            f"C_th={c_th:.4f}  R²={r2:.4f}",
            flush=True,
        )
        tag = lam_tag(lam)
        save[f"y_lam{tag}"] = y_m
        save[f"y_std_lam{tag}"] = y_s
        save[f"x_fit_lam{tag}"] = x
        save[f"y_fit_lam{tag}"] = y
        save[f"yerr_fit_lam{tag}"] = e
        save[f"slope_lam{tag}"] = np.float64(slope)
        save[f"slope_se_lam{tag}"] = np.float64(se)
        save[f"intercept_lam{tag}"] = np.float64(intercept)
        save[f"r2_lam{tag}"] = np.float64(r2)
        save[f"slope_theory_lam{tag}"] = np.float64(slope_th)
        res = {
            "lam": float(lam),
            "x": x,
            "y": y,
            "yerr": e,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "se": se,
        }
        results.append(res)
        plot_one_lambda(
            out,
            stem=f"flatten_xy_lam{tag}",
            lam=float(lam),
            **{k: res[k] for k in ("x", "y", "yerr", "slope", "intercept", "r2", "se")},
        )

    slopes = np.array([r["slope"] for r in results])
    ses = np.array([r["se"] for r in results])
    r2s = np.array([r["r2"] for r in results])
    intercepts = np.array([r["intercept"] for r in results])
    save["slope"] = slopes
    save["slope_se"] = ses
    save["r2"] = r2s
    save["intercept"] = intercepts
    save["slope_theory"] = np.array(
        [r["lam"] / (1.0 + r["lam"]) for r in results], dtype=np.float64
    )
    np.savez_compressed(out / "flatten_law_fit.npz", **save)

    plot_all_overlay(out, results=results)
    plot_slope_vs_lambda(out, results)

    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
