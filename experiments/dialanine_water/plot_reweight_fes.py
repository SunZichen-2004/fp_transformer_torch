#!/usr/bin/env python3
"""sFPT reweight FES: last-frac samples weighted by P(s)^{1/λ}.

For each replica in the last ``--last-frac`` of λ-target stages, evaluate the
stage torus-KDE P(s) at the sample and assign

    w(s) = P(s)^{1/λ} = exp( log P(s) / λ )

Pooled weighted histograms give 2D F(φ,ψ) and 1D F(φ), F(ψ)
(no extra /λ: the 1/λ is already in the weights). Mean±std over seeds.

Defaults match ``sFPT/convergence_320`` (R=320, λ=0.0625, last 20%).
"""

from __future__ import annotations

import argparse
import importlib.util
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
sys.path.insert(0, str(EXP))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    DIAL / "plot_convergence_analysis.py",
)
_dca = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dca)

_spec_m = importlib.util.spec_from_file_location(
    "dialanine_water_plot_convergence_multi",
    EXP / "plot_convergence_multi.py",
)
_pcm = importlib.util.module_from_spec(_spec_m)
assert _spec_m.loader is not None
_spec_m.loader.exec_module(_pcm)
load_metad_2d = _pcm.load_metad_2d
load_us_f1d = _pcm.load_us_f1d
mean_std = _pcm.mean_std
from plot_fes_compare import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    FE_LABEL,
    kT_kcal,
    nature_rcparams,
    relative_fes,
)
from plot_g5_1d_grid_and_dg_zoom import COLOR_US  # noqa: E402

KB_EV_PER_K = _dca.KB_EV_PER_K
EV_TO_KCAL = _dca.EV_TO_KCAL

DEFAULT_FPT = [
    EXP / "sFPT" / "results" / "dialanine_water_sfpt_r320_g1_s0_87371",
    EXP / "sFPT" / "results" / "dialanine_water_sfpt_r320_g1_s1_87372",
    EXP / "sFPT" / "results" / "dialanine_water_sfpt_r320_g1_s2_87373",
]
DEFAULT_METAD = [
    EXP.parent
    / "dialanine_water_meta"
    / "results"
    / "dialanine_water_metad_r64_s0_89387"
    / "live",
    EXP.parent
    / "dialanine_water_meta"
    / "results"
    / "dialanine_water_metad_r64_s2_89392"
    / "live",
]
DEFAULT_US = (
    EXP.parent
    / "dialanine_water_us"
    / "results"
    / "dialanine_water_us2d_r576_k100_s0_87367"
    / "mbar"
    / "fes_mbar.npz"
)
DEFAULT_CONV = EXP / "sFPT" / "convergence_320"
DEFAULT_OUT = EXP / "sFPT" / "reweight_FES"


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    tot = float(np.sum(w))
    ss = float(np.sum(w * w))
    if tot <= 0.0 or ss <= 0.0:
        return 0.0
    return (tot * tot) / ss


def fes_from_weighted_hist(
    counts: np.ndarray,
    *,
    kT: float,
) -> np.ndarray:
    p = np.asarray(counts, dtype=np.float64)
    tot = float(np.sum(np.clip(p, 0.0, None)))
    f = np.full(p.shape, np.nan, dtype=np.float64)
    if tot <= 0.0:
        return f
    pos = p > 0.0
    f[pos] = -float(kT) * np.log(p[pos] / tot)
    return relative_fes(f, None)


def collect_weighted_samples(
    fpt_dir: Path,
    *,
    last_frac: float,
    lambda_target: float,
    lambda_atol: float,
    device: torch.device,
    chunk: int,
) -> dict:
    rows = _dca.discover_dihedral_stages(
        fpt_dir,
        lambda_target=float(lambda_target),
        lambda_atol=float(lambda_atol),
    )
    if not rows:
        raise RuntimeError(f"no λ={lambda_target:g} stages in {fpt_dir}")
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    phi_parts: list[np.ndarray] = []
    psi_parts: list[np.ndarray] = []
    log_w_parts: list[np.ndarray] = []
    lams: list[float] = []
    for _, dih_p, bw_p, lam in use:
        dih = np.asarray(np.load(dih_p), dtype=np.float64)
        if dih.ndim != 2 or dih.shape[-1] != 2:
            raise ValueError(f"expected (N,2) dihedrals in {dih_p}, got {dih.shape}")
        bw = np.asarray(np.load(bw_p), dtype=np.float64)
        p = _dca.density_on_grid_torus(
            dih, dih, bw, chunk=int(chunk), device=device
        )
        log_p = np.log(np.clip(p, 1e-300, None))
        log_w_parts.append(log_p / float(lam))
        phi_parts.append(wrap_pi(dih[:, 0]))
        psi_parts.append(wrap_pi(dih[:, 1]))
        lams.append(float(lam))
    phi = np.concatenate(phi_parts, axis=0)
    psi = np.concatenate(psi_parts, axis=0)
    log_w = np.concatenate(log_w_parts, axis=0)
    log_w = log_w - float(np.max(log_w))
    w = np.exp(log_w)
    return {
        "phi": phi,
        "psi": psi,
        "w": w,
        "n_stages_all": int(len(rows)),
        "n_stages": int(len(use)),
        "stage_lo": int(use[0][0]),
        "stage_hi": int(use[-1][0]),
        "lam": float(np.mean(lams)),
        "n_samples": int(phi.shape[0]),
        "ess": float(ess(w)),
    }


def weighted_fes_1d(
    x: np.ndarray,
    w: np.ndarray,
    *,
    n_bins: int,
    kT: float,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    counts, _ = np.histogram(x, bins=edges, weights=w)
    xi = 0.5 * (edges[:-1] + edges[1:])
    return xi, fes_from_weighted_hist(counts, kT=kT)


def weighted_fes_2d(
    phi: np.ndarray,
    psi: np.ndarray,
    w: np.ndarray,
    *,
    n_bins: int,
    kT: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    counts, _, _ = np.histogram2d(phi, psi, bins=[edges, edges], weights=w)
    f = fes_from_weighted_hist(counts, kT=kT).T
    ax = 0.5 * (edges[:-1] + edges[1:])
    return ax, ax, f


def load_us_2d(us_path: Path, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    z = np.load(us_path)
    Fu = relative_fes(np.asarray(z["F"], dtype=np.float64), None)
    pu, su = np.asarray(z["phi"]), np.asarray(z["psi"])
    if Fu.shape == (len(pu), len(su)):
        Fu = Fu.T
    interp = RegularGridInterpolator(
        (su, pu), Fu, bounds_error=False, fill_value=None
    )
    g_phi, g_psi = np.meshgrid(phi, psi, indexing="xy")
    return relative_fes(
        interp(np.stack([g_psi.ravel(), g_phi.ravel()], axis=-1)).reshape(g_phi.shape),
        None,
    )


def plot_2d(
    out: Path,
    *,
    panels: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    vmax: float,
    last_frac: float,
) -> None:
    """``panels``: list of (phi, psi, F, title). Each panel may use its own grid."""
    del last_frac  # title already encodes this
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), sharex=True, sharey=True)
    levels = np.linspace(0.0, float(vmax), 25)
    for ax, (phi, psi, F, title) in zip(axes, panels):
        Z = np.minimum(relative_fes(F, None), float(vmax))
        im = ax.contourf(phi, psi, Z, levels=levels, cmap="viridis")
        ax.contour(
            phi, psi, Z, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.35
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"$\phi$ / rad")
        ax.set_ylabel(r"$\psi$ / rad")
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(FE_LABEL, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fes_2d_last_frac.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fes_2d_last_frac.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fes_2d_last_frac", flush=True)


def plot_1d_overlay(
    out: Path,
    *,
    stem: str,
    xlabel: str,
    xi,
    f_sfpt,
    e_sfpt,
    f_meta,
    e_meta,
    f_us,
    vmax: float,
    last_frac: float,
    n_samples: int,
    ess_mean: float,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    y = relative_fes(f_sfpt, None)
    ylo = relative_fes(f_sfpt - e_sfpt, None)
    yhi = relative_fes(f_sfpt + e_sfpt, None)
    y_meta = relative_fes(f_meta, None)
    y_us = relative_fes(f_us, None)
    if vmax > 0:
        y = np.minimum(y, float(vmax))
        ylo = np.minimum(ylo, float(vmax))
        yhi = np.minimum(yhi, float(vmax))
        y_meta = np.minimum(y_meta, float(vmax))
        y_us = np.minimum(y_us, float(vmax))
    if e_meta is not None:
        y_mlo = relative_fes(f_meta - e_meta, None)
        y_mhi = relative_fes(f_meta + e_meta, None)
        if vmax > 0:
            y_mlo = np.minimum(y_mlo, float(vmax))
            y_mhi = np.minimum(y_mhi, float(vmax))
        ax.fill_between(xi, y_mlo, y_mhi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
    ax.plot(xi, y_meta, color=COLOR_META, lw=1.2, label="WT-MetaD mean", zorder=2)
    ax.fill_between(xi, ylo, yhi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
    ax.plot(xi, y, color=COLOR_SFPT, lw=1.5, label=r"sFPT $P^{1/\lambda}$ reweight", zorder=4)
    ax.plot(xi, y_us, color=COLOR_US, lw=1.55, ls="--", label="US-MBAR", zorder=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(FE_LABEL)
    ax.set_xlim(-np.pi, np.pi)
    if vmax > 0:
        ax.set_ylim(0.0, float(vmax) * 1.02)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    ax.set_title(
        rf"last {100.0 * float(last_frac):.0f}%  $n={n_samples}$  "
        rf"ESS$={ess_mean:.0f}$",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def interp_1d(xi_src, f_src, xi_dst, *, as_fes: bool = True) -> np.ndarray:
    xi_src = np.asarray(xi_src, dtype=np.float64)
    f_src = np.asarray(f_src, dtype=np.float64)
    xi_dst = np.asarray(xi_dst, dtype=np.float64)
    if len(xi_src) == len(xi_dst) and np.allclose(xi_src, xi_dst):
        out = f_src.copy()
    else:
        out = np.interp(xi_dst, xi_src, f_src, left=np.nan, right=np.nan)
    return relative_fes(out, None) if as_fes else out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fpt-dirs",
        type=str,
        default=",".join(str(d) for d in DEFAULT_FPT),
    )
    p.add_argument(
        "--metad-dirs",
        type=str,
        default=",".join(str(d) for d in DEFAULT_METAD),
    )
    p.add_argument("--us-mbar", type=Path, default=DEFAULT_US)
    p.add_argument("--conv-dir", type=Path, default=DEFAULT_CONV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--last-frac", type=float, default=0.2)
    p.add_argument("--lambda-target", type=float, default=0.0625)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--n-bins", type=int, default=72)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--fes2d-vmax", type=float, default=20.0)
    p.add_argument("--fes1d-vmax", type=float, default=20.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fpt_dirs = [Path(x.strip()).resolve() for x in args.fpt_dirs.split(",") if x.strip()]
    metad_dirs = [
        Path(x.strip()).resolve() for x in args.metad_dirs.split(",") if x.strip()
    ]
    us_path = Path(args.us_mbar).resolve()
    conv = Path(args.conv_dir).resolve()
    kT = kT_kcal(float(args.temperature))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    # Prefer cached MetaD/US 2D from convergence_320 so reference panels match.
    f2_path = conv / "fes_2d_last_frac.npz"
    f1_path = conv / "fes_1d_all_groups.npz"
    if f2_path.is_file():
        z2 = np.load(f2_path)
        phi_ref = np.asarray(z2["phi"], dtype=np.float64)
        psi_ref = np.asarray(z2["psi"], dtype=np.float64)
        F_meta = relative_fes(np.asarray(z2["F_metad"], dtype=np.float64), None)
        F_us = relative_fes(np.asarray(z2["F_us"], dtype=np.float64), None)
        print(f"[ref2d] {f2_path}", flush=True)
    else:
        f2_meta_list = []
        phi_ref = psi_ref = None
        for md in metad_dirs:
            phi_i, psi_i, f2 = load_metad_2d(md / "fes.npz", int(args.grid_n))
            if phi_ref is None:
                phi_ref, psi_ref = phi_i, psi_i
            f2_meta_list.append(f2)
        F_meta, _ = mean_std(np.stack(f2_meta_list, 0), 0)
        F_meta = relative_fes(F_meta, None)
        F_us = load_us_2d(us_path, phi_ref, psi_ref)

    if f1_path.is_file():
        z1 = np.load(f1_path)
        xi_ref = np.asarray(z1["xi"], dtype=np.float64)
        f1d_meta = relative_fes(np.asarray(z1["F1d_metad_mean"], dtype=np.float64), None)
        f1d_meta_std = np.asarray(z1["F1d_metad_std"], dtype=np.float64)
        f1d_us = relative_fes(np.asarray(z1["F1d_us"], dtype=np.float64), None)
        print(f"[ref1d] {f1_path}", flush=True)
    else:
        xi_ref, f1d_us = load_us_f1d(us_path)
        f1d_meta = f1d_us
        f1d_meta_std = np.zeros_like(f1d_meta)

    xi_psi_meta, f1d_psi_meta = _dca.project_1d_axis(
        phi_ref,
        psi_ref,
        F_meta,
        axis="psi",
        n_bins=int(args.n_bins),
        temperature=float(args.temperature),
    )
    _, f1d_psi_us = _dca.project_1d_axis(
        phi_ref,
        psi_ref,
        F_us,
        axis="psi",
        n_bins=int(args.n_bins),
        temperature=float(args.temperature),
    )

    f2_runs = []
    f1_phi_runs = []
    f1_psi_runs = []
    stats = []
    xi_phi = xi_psi = phi_h = None
    for d in fpt_dirs:
        print(f"[reweight] {d.name}", flush=True)
        pack = collect_weighted_samples(
            d,
            last_frac=float(args.last_frac),
            lambda_target=float(args.lambda_target),
            lambda_atol=float(args.lambda_atol),
            device=device,
            chunk=int(args.chunk),
        )
        stats.append(pack)
        print(
            f"  stages {pack['stage_lo']:04d}–{pack['stage_hi']:04d}  "
            f"{pack['n_stages']}/{pack['n_stages_all']}  "
            f"n={pack['n_samples']}  ESS={pack['ess']:.1f}  "
            f"λ={pack['lam']:g}",
            flush=True,
        )
        ax, _, f2 = weighted_fes_2d(
            pack["phi"],
            pack["psi"],
            pack["w"],
            n_bins=int(args.n_bins),
            kT=kT,
        )
        xi_p, f_phi = weighted_fes_1d(
            pack["phi"], pack["w"], n_bins=int(args.n_bins), kT=kT
        )
        xi_s, f_psi = weighted_fes_1d(
            pack["psi"], pack["w"], n_bins=int(args.n_bins), kT=kT
        )
        if phi_h is None:
            phi_h = ax
            xi_phi = xi_p
            xi_psi = xi_s
        f2_runs.append(f2)
        f1_phi_runs.append(f_phi)
        f1_psi_runs.append(f_psi)

    f2_stack = np.stack(f2_runs, 0)
    f2_mean, f2_std = mean_std(f2_stack, 0)
    f2_mean = relative_fes(f2_mean, None)
    f_phi_mean, f_phi_std = mean_std(np.stack(f1_phi_runs, 0), 0)
    f_psi_mean, f_psi_std = mean_std(np.stack(f1_psi_runs, 0), 0)
    f_phi_mean = relative_fes(f_phi_mean, None)
    f_psi_mean = relative_fes(f_psi_mean, None)

    n_samp = int(np.mean([s["n_samples"] for s in stats]))
    ess_mean = float(np.mean([s["ess"] for s in stats]))
    s0 = int(stats[0]["stage_lo"])
    s1 = int(stats[0]["stage_hi"])

    np.savez_compressed(
        out / "fes_2d_last_frac.npz",
        phi=phi_h,
        psi=phi_h,
        F_sfpt=f2_mean,
        F_sfpt_std=f2_std,
        F_sfpt_runs=f2_stack,
        phi_ref=phi_ref,
        psi_ref=psi_ref,
        F_metad=F_meta,
        F_us=F_us,
        last_frac=np.float64(args.last_frac),
        lambda_target=np.float64(args.lambda_target),
        stage_lo=s0,
        stage_hi=s1,
        n_samples=n_samp,
        ess_mean=np.float64(ess_mean),
        n_bins=int(args.n_bins),
        fpt_dirs=np.asarray([str(d) for d in fpt_dirs]),
    )
    np.savez_compressed(
        out / "fes_1d_last_frac.npz",
        xi_phi=xi_phi,
        F_phi=f_phi_mean,
        F_phi_std=f_phi_std,
        F_phi_metad=interp_1d(xi_ref, f1d_meta, xi_phi),
        F_phi_metad_std=interp_1d(
            xi_ref, f1d_meta_std, xi_phi, as_fes=False
        ),
        F_phi_us=interp_1d(xi_ref, f1d_us, xi_phi),
        xi_psi=xi_psi,
        F_psi=f_psi_mean,
        F_psi_std=f_psi_std,
        F_psi_metad=interp_1d(xi_psi_meta, f1d_psi_meta, xi_psi),
        F_psi_us=interp_1d(xi_psi_meta, f1d_psi_us, xi_psi),
        last_frac=np.float64(args.last_frac),
        n_samples=n_samp,
        ess_mean=np.float64(ess_mean),
        ess_runs=np.asarray([s["ess"] for s in stats], dtype=np.float64),
        n_samples_runs=np.asarray([s["n_samples"] for s in stats], dtype=np.int64),
        temperature=np.float64(args.temperature),
    )

    plot_2d(
        out,
        panels=[
            (
                phi_h,
                phi_h,
                f2_mean,
                rf"sFPT reweight (last {100.0 * float(args.last_frac):.0f}%)",
            ),
            (phi_ref, psi_ref, F_meta, "WT-MetaD mean"),
            (phi_ref, psi_ref, F_us, "US-MBAR"),
        ],
        vmax=float(args.fes2d_vmax),
        last_frac=float(args.last_frac),
    )
    plot_1d_overlay(
        out,
        stem="fes_1d_phi",
        xlabel=r"$\phi$ / rad",
        xi=xi_phi,
        f_sfpt=f_phi_mean,
        e_sfpt=f_phi_std,
        f_meta=interp_1d(xi_ref, f1d_meta, xi_phi),
        e_meta=interp_1d(xi_ref, f1d_meta_std, xi_phi, as_fes=False),
        f_us=interp_1d(xi_ref, f1d_us, xi_phi),
        vmax=float(args.fes1d_vmax),
        last_frac=float(args.last_frac),
        n_samples=n_samp,
        ess_mean=ess_mean,
    )
    plot_1d_overlay(
        out,
        stem="fes_1d_psi",
        xlabel=r"$\psi$ / rad",
        xi=xi_psi,
        f_sfpt=f_psi_mean,
        e_sfpt=f_psi_std,
        f_meta=interp_1d(xi_psi_meta, f1d_psi_meta, xi_psi),
        e_meta=None,
        f_us=interp_1d(xi_psi_meta, f1d_psi_us, xi_psi),
        vmax=float(args.fes1d_vmax),
        last_frac=float(args.last_frac),
        n_samples=n_samp,
        ess_mean=ess_mean,
    )
    print(
        f"[done] out={out}  stages {s0}–{s1}  n≈{n_samp}  ESS≈{ess_mean:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
