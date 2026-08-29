#!/usr/bin/env python3
"""λ-scan last-50% FES vs US-MBAR: 2D + 1D(φ/ψ) panels and RMSE(λ).

For each λ (two seeds): torus-KDE on the last ``last_frac`` stages → F(φ,ψ),
Boltzmann-project F(φ), F(ψ); RMSE vs US-MBAR; mean±std over seeds.
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

from plot_fes_compare import FE_LABEL, nature_rcparams, relative_fes  # noqa: E402
from plot_fes_rmse_vs_time import rmse_1d  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import COLOR_US  # noqa: E402

KB_EV_PER_K = _dca.KB_EV_PER_K
EV_TO_KCAL = _dca.EV_TO_KCAL
LAM_RE = re.compile(r"lam([0-9.]+)_", re.I)

LAM_COLORS = {
    0.02: "#1b9e77",
    0.04: "#d95f02",
    0.06: "#7570b3",
    0.08: "#e7298a",
    0.10: "#66a61e",
    0.1: "#66a61e",
}


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    if stack.shape[axis] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


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


def load_us_bundle(path: Path) -> dict:
    z = np.load(path)
    phi = np.asarray(z["phi"], dtype=np.float64)
    psi = np.asarray(z["psi"], dtype=np.float64)
    F = relative_fes(np.asarray(z["F"], dtype=np.float64), None)
    if F.shape == (len(phi), len(psi)) and F.shape != (len(psi), len(phi)):
        F = F.T
    if "xi_phi" in z.files and "F_phi" in z.files:
        xi_phi = np.asarray(z["xi_phi"], dtype=np.float64)
        F_phi = relative_fes(np.asarray(z["F_phi"], dtype=np.float64), None)
    else:
        xi_phi = np.asarray(z["xi"], dtype=np.float64)
        F_phi = relative_fes(np.asarray(z["F_kcal"], dtype=np.float64), None)
    if "xi_psi" in z.files and "F_psi" in z.files:
        xi_psi = np.asarray(z["xi_psi"], dtype=np.float64)
        F_psi = relative_fes(np.asarray(z["F_psi"], dtype=np.float64), None)
    else:
        xi_psi, F_psi = _dca.project_1d_axis(
            phi, psi, F, axis="psi", n_bins=len(xi_phi), temperature=300.0
        )
    return {
        "phi": phi,
        "psi": psi,
        "F": F,
        "xi_phi": xi_phi,
        "F_phi": F_phi,
        "xi_psi": xi_psi,
        "F_psi": F_psi,
    }


def kde_2d_last_frac(
    fpt_dir: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    last_frac: float,
    temperature: float,
    device: torch.device,
    chunk: int,
    lambda_target: float,
    lambda_atol: float,
) -> tuple[np.ndarray, int, int, int]:
    rows = _dca.discover_dihedral_stages(
        fpt_dir,
        lambda_target=float(lambda_target),
        lambda_atol=float(lambda_atol),
    )
    if not rows:
        raise RuntimeError(f"no λ≈{lambda_target:g} stages in {fpt_dir}")
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    lam = float(use[0][3])
    for _, dih_p, bw_p, _ in use:
        p_acc += _dca.density_on_grid_torus(
            grid_xy, np.load(dih_p), np.load(bw_p), chunk=chunk, device=device
        )
    p = np.clip(p_acc / float(len(use)), 1e-300, None)
    F = relative_fes(
        (-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None
    )
    return F, int(use[0][0]), int(use[-1][0]), int(n_keep)


def rmse_2d(
    f_a: np.ndarray,
    f_b: np.ndarray,
    *,
    vmax: float | None,
) -> float:
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


def align_1d(
    xi_src: np.ndarray, f_src: np.ndarray, xi_dst: np.ndarray
) -> np.ndarray:
    if len(xi_src) == len(xi_dst) and np.allclose(xi_src, xi_dst):
        return relative_fes(f_src, None)
    f = np.interp(xi_dst, xi_src, f_src, left=np.nan, right=np.nan)
    return relative_fes(f, None)


def plot_2d_panel(
    out: Path,
    *,
    stem: str,
    phi: np.ndarray,
    psi: np.ndarray,
    F_sfpt: np.ndarray,
    F_us: np.ndarray,
    vmax: float,
    title: str,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    levels = np.linspace(0.0, float(vmax), 25)
    for ax, F, lab in [
        (axes[0], F_sfpt, "sFPT KDE"),
        (axes[1], F_us, "US-MBAR"),
    ]:
        Z = np.minimum(relative_fes(F, None), float(vmax))
        im = ax.contourf(phi, psi, Z, levels=levels, cmap="viridis")
        ax.contour(
            phi, psi, Z, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.35
        )
        ax.set_title(lab, fontsize=9)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"$\phi$ / rad")
        ax.set_ylabel(r"$\psi$ / rad")
        ax.set_aspect("equal")
        ax.tick_params(which="both", top=False, right=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(FE_LABEL, fontsize=8)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_1d_panel(
    out: Path,
    *,
    stem: str,
    xi_phi: np.ndarray,
    F_phi: np.ndarray,
    F_phi_std: np.ndarray | None,
    F_phi_us: np.ndarray,
    xi_psi: np.ndarray,
    F_psi: np.ndarray,
    F_psi_std: np.ndarray | None,
    F_psi_us: np.ndarray,
    vmax: float,
    title: str,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, xi, f, fstd, fus, xlab in [
        (axes[0], xi_phi, F_phi, F_phi_std, F_phi_us, r"$\phi$ / rad"),
        (axes[1], xi_psi, F_psi, F_psi_std, F_psi_us, r"$\psi$ / rad"),
    ]:
        y_us = relative_fes(fus, None)
        y = relative_fes(f, None)
        if vmax > 0:
            y_us = np.minimum(y_us, float(vmax))
            y = np.minimum(y, float(vmax))
        ax.plot(xi, y_us, color=COLOR_US, lw=1.6, ls="--", label="US-MBAR", zorder=2)
        ax.plot(xi, y, color="#1f77b4", lw=1.5, label="sFPT", zorder=3)
        if fstd is not None and np.any(fstd > 0):
            lo = y - fstd
            hi = y + fstd
            if vmax > 0:
                lo = np.minimum(lo, float(vmax))
                hi = np.minimum(hi, float(vmax))
            ax.fill_between(xi, lo, hi, color="#1f77b4", alpha=0.22, lw=0, zorder=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel(FE_LABEL)
        ax.set_xlim(-np.pi, np.pi)
        if vmax > 0:
            ax.set_ylim(0.0, float(vmax) * 1.02)
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(which="both", top=False, right=False)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_lambda(
    out: Path,
    *,
    stem: str,
    lams: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    ylabel: str,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    colors = [LAM_COLORS.get(float(l), LAM_COLORS.get(round(float(l), 2), "C0")) for l in lams]
    ax.errorbar(
        lams,
        y,
        yerr=yerr,
        fmt="o-",
        color="0.25",
        ecolor="0.45",
        elinewidth=1.0,
        capsize=3,
        lw=1.4,
        ms=5.5,
        zorder=2,
    )
    for x, yy, c in zip(lams, y, colors):
        ax.plot(x, yy, "o", color=c, ms=6.5, zorder=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


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
        / "experiments/dialanine_water/lambda_scan/convergence_analysis/fes_last50",
    )
    p.add_argument(
        "--us-mbar",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water_us/results"
        / "dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz",
    )
    p.add_argument("--last-frac", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--fes2d-vmax", type=float, default=20.0)
    p.add_argument("--fes1d-vmax", type=float, default=20.0)
    p.add_argument(
        "--rmse-vmax",
        type=float,
        default=0.0,
        help="If >0, RMSE only where F_US ≤ this (kcal/mol)",
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

    us = load_us_bundle(us_path)
    phi, psi = us["phi"], us["psi"]
    F_us = us["F"]
    xi_phi, F_phi_us = us["xi_phi"], us["F_phi"]
    xi_psi, F_psi_us = us["xi_psi"], us["F_psi"]
    rmse_vmax = float(args.rmse_vmax) if float(args.rmse_vmax) > 0 else None
    print(
        f"[us] grid φ={len(phi)} ψ={len(psi)}  "
        f"1D φ={len(xi_phi)} ψ={len(xi_psi)}  rmse_vmax={rmse_vmax}",
        flush=True,
    )

    runs = discover_runs(Path(args.scan_dir))
    if not runs:
        raise RuntimeError(f"no lam runs under {args.scan_dir}")
    print(f"[scan] λs={sorted(runs)}", flush=True)

    lams: list[float] = []
    rmse2_m, rmse2_s = [], []
    rmse_phi_m, rmse_phi_s = [], []
    rmse_psi_m, rmse_psi_s = [], []
    save: dict = {
        "phi": phi,
        "psi": psi,
        "F_us": F_us,
        "xi_phi": xi_phi,
        "F_phi_us": F_phi_us,
        "xi_psi": xi_psi,
        "F_psi_us": F_psi_us,
        "last_frac": np.float64(args.last_frac),
        "rmse_vmax": np.float64(args.rmse_vmax),
    }

    for lam in sorted(runs):
        dirs = runs[lam]
        F2_list, Fphi_list, Fpsi_list = [], [], []
        r2_list, rphi_list, rpsi_list = [], [], []
        stage_lo = stage_hi = n_keep = None
        for d in dirs:
            F2, s0, s1, nk = kde_2d_last_frac(
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
            xp, Fp = _dca.project_1d_axis(
                phi,
                psi,
                F2,
                axis="phi",
                n_bins=len(xi_phi),
                temperature=float(args.temperature),
            )
            xs, Fs = _dca.project_1d_axis(
                phi,
                psi,
                F2,
                axis="psi",
                n_bins=len(xi_psi),
                temperature=float(args.temperature),
            )
            Fp = align_1d(xp, Fp, xi_phi)
            Fs = align_1d(xs, Fs, xi_psi)
            r2 = rmse_2d(F2, F_us, vmax=rmse_vmax)
            rp = rmse_1d(Fp, F_phi_us, vmax=rmse_vmax, xi=xi_phi, xi_lo=-np.pi, xi_hi=np.pi)
            rs = rmse_1d(Fs, F_psi_us, vmax=rmse_vmax, xi=xi_psi, xi_lo=-np.pi, xi_hi=np.pi)
            F2_list.append(F2)
            Fphi_list.append(Fp)
            Fpsi_list.append(Fs)
            r2_list.append(r2)
            rphi_list.append(rp)
            rpsi_list.append(rs)
            stage_lo, stage_hi, n_keep = s0, s1, nk
            print(
                f"[kde] λ={lam:g} {d.name} stages={nk} ({s0}-{s1})  "
                f"RMSE2d={r2:.3f}  RMSEφ={rp:.3f}  RMSEψ={rs:.3f}",
                flush=True,
            )

        F2_m, F2_s = mean_std(np.stack(F2_list, axis=0))
        Fphi_m, Fphi_s = mean_std(np.stack(Fphi_list, axis=0))
        Fpsi_m, Fpsi_s = mean_std(np.stack(Fpsi_list, axis=0))
        F2_m = relative_fes(F2_m, None)
        Fphi_m = relative_fes(Fphi_m, None)
        Fpsi_m = relative_fes(Fpsi_m, None)

        r2a = np.asarray(r2_list, dtype=np.float64)
        rpa = np.asarray(rphi_list, dtype=np.float64)
        rsa = np.asarray(rpsi_list, dtype=np.float64)
        lams.append(lam)
        rmse2_m.append(float(np.nanmean(r2a)))
        rmse2_s.append(float(np.nanstd(r2a, ddof=1)) if len(r2a) > 1 else 0.0)
        rmse_phi_m.append(float(np.nanmean(rpa)))
        rmse_phi_s.append(float(np.nanstd(rpa, ddof=1)) if len(rpa) > 1 else 0.0)
        rmse_psi_m.append(float(np.nanmean(rsa)))
        rmse_psi_s.append(float(np.nanstd(rsa, ddof=1)) if len(rsa) > 1 else 0.0)

        tag = f"{lam:g}".replace(".", "p")
        save[f"F2_lam{tag}"] = F2_m
        save[f"F2_std_lam{tag}"] = F2_s
        save[f"F_phi_lam{tag}"] = Fphi_m
        save[f"F_phi_std_lam{tag}"] = Fphi_s
        save[f"F_psi_lam{tag}"] = Fpsi_m
        save[f"F_psi_std_lam{tag}"] = Fpsi_s
        save[f"rmse2_seeds_lam{tag}"] = r2a
        save[f"rmse_phi_seeds_lam{tag}"] = rpa
        save[f"rmse_psi_seeds_lam{tag}"] = rsa

        title = (
            rf"$\lambda$={lam:g}  last {float(args.last_frac):.0%}  "
            rf"(stage {stage_lo}–{stage_hi}, n={n_keep}, seeds={len(dirs)})"
        )
        plot_2d_panel(
            out,
            stem=f"fes_2d_lam{tag}",
            phi=phi,
            psi=psi,
            F_sfpt=F2_m,
            F_us=F_us,
            vmax=float(args.fes2d_vmax),
            title=title,
        )
        plot_1d_panel(
            out,
            stem=f"fes_1d_lam{tag}",
            xi_phi=xi_phi,
            F_phi=Fphi_m,
            F_phi_std=Fphi_s,
            F_phi_us=F_phi_us,
            xi_psi=xi_psi,
            F_psi=Fpsi_m,
            F_psi_std=Fpsi_s,
            F_psi_us=F_psi_us,
            vmax=float(args.fes1d_vmax),
            title=title,
        )
        print(
            f"[λ={lam:g}] RMSE2d={rmse2_m[-1]:.3f}±{rmse2_s[-1]:.3f}  "
            f"φ={rmse_phi_m[-1]:.3f}±{rmse_phi_s[-1]:.3f}  "
            f"ψ={rmse_psi_m[-1]:.3f}±{rmse_psi_s[-1]:.3f}",
            flush=True,
        )

    lams_a = np.asarray(lams, dtype=np.float64)
    r2m = np.asarray(rmse2_m)
    r2s = np.asarray(rmse2_s)
    rpm = np.asarray(rmse_phi_m)
    rps = np.asarray(rmse_phi_s)
    rsm = np.asarray(rmse_psi_m)
    rss = np.asarray(rmse_psi_s)
    save["lambdas"] = lams_a
    save["rmse_2d"] = r2m
    save["rmse_2d_std"] = r2s
    save["rmse_phi"] = rpm
    save["rmse_phi_std"] = rps
    save["rmse_psi"] = rsm
    save["rmse_psi_std"] = rss
    np.savez_compressed(out / "fes_rmse_by_lambda.npz", **save)

    ylab = r"RMSE vs US / kcal mol$^{-1}$"
    if rmse_vmax is not None:
        ylab = rf"RMSE ($F_{{\mathrm{{US}}}}\leq {rmse_vmax:g}$) / kcal mol$^{{-1}}$"
    plot_rmse_vs_lambda(
        out, stem="rmse2d_vs_lambda", lams=lams_a, y=r2m, yerr=r2s, ylabel=ylab
    )
    plot_rmse_vs_lambda(
        out, stem="rmse_phi_vs_lambda", lams=lams_a, y=rpm, yerr=rps, ylabel=ylab
    )
    plot_rmse_vs_lambda(
        out, stem="rmse_psi_vs_lambda", lams=lams_a, y=rsm, yerr=rss, ylabel=ylab
    )

    # combined 3-panel RMSE(λ)
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True)
    for ax, y, e, tit in [
        (axes[0], r2m, r2s, r"2D $F(\phi,\psi)$"),
        (axes[1], rpm, rps, r"1D $F(\phi)$"),
        (axes[2], rsm, rss, r"1D $F(\psi)$"),
    ]:
        ax.errorbar(
            lams_a, y, yerr=e, fmt="o-", color="0.25", ecolor="0.45",
            elinewidth=1.0, capsize=3, lw=1.4, ms=5.0, zorder=2,
        )
        for x, yy in zip(lams_a, y):
            c = LAM_COLORS.get(float(x), LAM_COLORS.get(round(float(x), 2), "C0"))
            ax.plot(x, yy, "o", color=c, ms=6.0, zorder=3)
        ax.set_title(tit, fontsize=10)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(ylab)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "rmse_vs_lambda.png", dpi=200)
    fig.savefig(out / "rmse_vs_lambda.pdf")
    plt.close(fig)
    print("[ok] rmse_vs_lambda", flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
