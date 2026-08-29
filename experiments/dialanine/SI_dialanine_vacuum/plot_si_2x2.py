#!/usr/bin/env python3
"""SI 2×2 for dialanine vacuum.

a/b: ΔG from raw MetaD hills + sFPT stages (no t_max crop).
c/d: 1D F(φ) RMSE with F_ref ≤ 8 from convergence_analysis/fes_rmse_F8.npz.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.lines import Line2D

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = Path(__file__).resolve().parent
DIAL = EXP.parent
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

from plot_fes_compare import COLOR_META, COLOR_SFPT, nature_rcparams, relative_fes  # noqa: E402
from plot_fes_rmse_vs_time import delta_g_two_basins  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import fpt_wall_hours_from_throughput  # noqa: E402

incremental_metad_phi = _dca.incremental_metad_phi
kde_groups_phi = _dca.kde_groups_phi
metad_ref_from_last_ps = _dca.metad_ref_from_last_ps

COLOR_REF = "#2ca02c"
DOUBLE_COL_IN = 180.0 / 25.4
CACHE = EXP / "si_raw_series.npz"
F8_NPZ = DIAL / "convergence_analysis" / "fes_rmse_F8.npz"

FPT_DIRS = [
    DIAL / "results" / "dialanine_gaff_s0_87220",
    DIAL / "results" / "dialanine_gaff_s2_90437",
    DIAL / "results" / "dialanine_gaff_s4_90438",
]
METAD_DIRS = [
    DIAL.parent / "dialanine_meta" / "results" / "dialanine_metad_r64_s0_87211" / "live",
    DIAL.parent / "dialanine_meta" / "results" / "dialanine_metad_r64_s2_90425" / "live",
    DIAL.parent / "dialanine_meta" / "results" / "dialanine_metad_r64_s4_90426" / "live",
]

GROUP_SIZE = 5
DT_GROUP = 0.5
N_REP = 64
DEPOSIT_PACE = 50
TIMESTEP = 0.001
BIAS = 15.0
SIGMA = (0.35, 0.35)
STRIDE = 10
GRID_N = 73
N_BINS = 72
T_K = 300.0
FPT_RATE = 64.4
METAD_RATE = 139.0


def mean_std_nan(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0)
    std = np.nanstd(stack, axis=0, ddof=1)
    std = np.where(n <= 1, 0.0, std)
    return mean, std


def overlay_on_t(t_list: list[np.ndarray], y_list: list[np.ndarray]):
    t_all = np.unique(np.concatenate([np.asarray(t, dtype=np.float64) for t in t_list]))
    t_all.sort()
    mat = np.full((len(y_list), t_all.size), np.nan, dtype=np.float64)
    for i, (t, y) in enumerate(zip(t_list, y_list)):
        t = np.asarray(t, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        idx = np.searchsorted(t_all, t)
        mat[i, idx] = y
    mu, sd = mean_std_nan(mat)
    return t_all, mu, sd


def tag(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.025,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def series(ax, x_m, y_m, e_m, x_s, y_s, e_s) -> None:
    ax.plot(x_m, y_m, color=COLOR_META, lw=1.7, zorder=3)
    ax.fill_between(x_m, y_m - e_m, y_m + e_m, color=COLOR_META, alpha=0.22, lw=0, zorder=1)
    ax.plot(x_s, y_s, color=COLOR_SFPT, lw=2.0, zorder=4)
    ax.fill_between(x_s, y_s - e_s, y_s + e_s, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=2)


def style_ax(ax) -> None:
    ax.set_frame_on(True)
    ax.tick_params(which="both", top=False, right=False, labelsize=11, direction="out")
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color("0.15")


def compute(device: torch.device) -> dict[str, np.ndarray]:
    phi = np.linspace(-np.pi, np.pi, GRID_N)
    psi = np.linspace(-np.pi, np.pi, GRID_N)
    fes0 = METAD_DIRS[0] / "fes.npz"
    if fes0.is_file():
        zfes = np.load(fes0)
        phi_f = np.asarray(zfes["phi"], dtype=np.float64)
        psi_f = np.asarray(zfes["psi"], dtype=np.float64)
        if len(phi_f) == GRID_N and len(psi_f) == GRID_N:
            phi, psi = phi_f, psi_f

    f1d_ref_list = []
    xi = None
    for md in METAD_DIRS:
        hills = md / "hills.npz"
        xi_i, f1d_i, t_tot = metad_ref_from_last_ps(
            hills,
            phi=phi,
            psi=psi,
            n_replicas=N_REP,
            deposit_pace=DEPOSIT_PACE,
            timestep=TIMESTEP,
            bias_factor=BIAS,
            sigma=SIGMA,
            last_ps=0.0,
            n_bins=N_BINS,
            temperature=T_K,
            device=device,
        )
        if xi is None:
            xi = xi_i
        f1d_ref_list.append(relative_fes(f1d_i, None))
        print(f"[ref] {md.parent.name} traj={t_tot:.1f} ps", flush=True)
    f1d_ref, _ = mean_std_nan(np.stack(f1d_ref_list, 0))
    f1d_ref = relative_fes(f1d_ref, None)
    dG_ref = delta_g_two_basins(
        xi, f1d_ref, temperature=T_K, xi_cut=0.0, xi_lo=-np.pi, xi_hi=np.pi
    )
    print(f"[ref] ΔG={dG_ref:.4f}  n_seed={len(METAD_DIRS)}", flush=True)

    t_m_list, dG_m_list, rmse_m_list = [], [], []
    for md in METAD_DIRS:
        t_m, rmse_m, dG_m, _ = incremental_metad_phi(
            md / "hills.npz",
            phi=phi,
            psi=psi,
            f1d_ref=f1d_ref,
            xi=xi,
            n_replicas=N_REP,
            deposit_pace=DEPOSIT_PACE,
            timestep=TIMESTEP,
            bias_factor=BIAS,
            sigma=SIGMA,
            temperature=T_K,
            n_bins=N_BINS,
            vmax_kcal=None,
            stride_deposits=STRIDE,
            xi_cut=0.0,
            xi_lo=-np.pi,
            xi_hi=np.pi,
            device=device,
        )
        t_m_list.append(t_m)
        dG_m_list.append(dG_m)
        rmse_m_list.append(rmse_m)
        print(f"[metad] {md.parent.name} t_end={t_m[-1]:.1f} ps", flush=True)
    t_meta, dG_meta, dG_meta_std = overlay_on_t(t_m_list, dG_m_list)
    _, rmse_meta, rmse_meta_std = overlay_on_t(t_m_list, rmse_m_list)

    t_s_list, dG_s_list, rmse_s_list = [], [], []
    for d in FPT_DIRS:
        t_s, r_s, g_s, _, _ = kde_groups_phi(
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
            max_groups=None,
            lambda_target=0.1,
        )
        t_s_list.append(t_s)
        dG_s_list.append(g_s)
        rmse_s_list.append(r_s)
        print(f"[sFPT] {d.name} t_end={t_s[-1]:.1f} ps  n={len(t_s)}", flush=True)
    t_sfpt, dG_sfpt, dG_sfpt_std = overlay_on_t(t_s_list, dG_s_list)
    _, rmse_sfpt, rmse_sfpt_std = overlay_on_t(t_s_list, rmse_s_list)

    tw_sfpt = fpt_wall_hours_from_throughput(t_sfpt, timestep_ps=TIMESTEP, rate_it_s=FPT_RATE)
    tw_meta = fpt_wall_hours_from_throughput(t_meta, timestep_ps=TIMESTEP, rate_it_s=METAD_RATE)
    return dict(
        t_metad_ps=t_meta,
        t_kde_ps=t_sfpt,
        tw_metad_h=tw_meta,
        tw_kde_h=tw_sfpt,
        cost_metad_gpuh=tw_meta,
        cost_kde_gpuh=tw_sfpt,
        dG_metad=dG_meta,
        dG_metad_std=dG_meta_std,
        dG_kde=dG_sfpt,
        dG_kde_std=dG_sfpt_std,
        rmse_metad=rmse_meta,
        rmse_metad_std=rmse_meta_std,
        rmse_kde=rmse_sfpt,
        rmse_kde_std=rmse_sfpt_std,
        dG_metad_ref=np.float64(dG_ref),
    )


def plot_figure(z: dict[str, np.ndarray]) -> None:
    t_m = np.asarray(z["t_metad_ps"])
    t_s = np.asarray(z["t_kde_ps"])
    tw_m = np.asarray(z["tw_metad_h"])
    tw_s = np.asarray(z["tw_kde_h"])
    dG_m = np.asarray(z["dG_metad"])
    dG_ms = np.asarray(z["dG_metad_std"])
    dG_s = np.asarray(z["dG_kde"])
    dG_ss = np.asarray(z["dG_kde_std"])
    dG_ref = float(z["dG_metad_ref"])

    z8 = np.load(F8_NPZ)
    t8_m = np.asarray(z8["t_metad_ps"])
    t8_s = np.asarray(z8["t_kde_ps"])
    tw8_m = np.asarray(z8["tw_metad_h"])
    tw8_s = np.asarray(z8["tw_kde_h"])
    r_m = np.asarray(z8["rmse_metad"])
    r_ms = np.asarray(z8["rmse_metad_std"])
    r_s = np.asarray(z8["rmse_kde"])
    r_ss = np.asarray(z8["rmse_kde_std"])
    vmax8 = float(z8["vmax_kcal"])
    z8.close()
    print(
        f"[F8] {F8_NPZ.name}  vmax={vmax8:g}  "
        f"t=0–{float(t8_s[-1]):.1f} ps  tw=0–{float(tw8_s[-1]):.3f} h",
        flush=True,
    )

    plt.rcParams.update(nature_rcparams())
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.05,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
        }
    )

    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.96))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        wspace=0.40,
        hspace=0.38,
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.91,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    tw_m_min = np.asarray(tw_m) * 60.0
    tw_s_min = np.asarray(tw_s) * 60.0
    tw8_m_min = np.asarray(tw8_m) * 60.0
    tw8_s_min = np.asarray(tw8_s) * 60.0

    ax_a.axhline(dG_ref, color=COLOR_REF, ls=":", lw=1.4, zorder=1)
    series(ax_a, tw_m_min, dG_m, dG_ms, tw_s_min, dG_s, dG_ss)
    ax_a.set_xlim(0.0, 12.0)
    ymin = min(float(np.nanmin(dG_m - dG_ms)), float(np.nanmin(dG_s - dG_ss)))
    pad = 0.08 * max(5.0 - ymin, 1.0)
    ax_a.set_ylim(ymin - pad, 5.0)
    ax_a.set_xlabel(r"wall-clock time / min")
    ax_a.set_ylabel(r"$\Delta G$ (kcal/mol)")
    tag(ax_a, "a")
    style_ax(ax_a)

    ax_b.axhline(dG_ref, color=COLOR_REF, ls=":", lw=1.4, zorder=1)
    series(ax_b, t_m, dG_m, dG_ms, t_s, dG_s, dG_ss)
    ax_b.set_xlim(0.0, 100.0)
    ax_b.set_ylim(ymin - pad, 5.0)
    ax_b.set_xlabel(r"$t$ / ps")
    ax_b.set_ylabel(r"$\Delta G$ (kcal/mol)")
    tag(ax_b, "b")
    style_ax(ax_b)

    series(ax_c, tw8_m_min, r_m, r_ms, tw8_s_min, r_s, r_ss)
    ax_c.set_xlim(0.0, float(max(tw8_m_min[-1], tw8_s_min[-1])))
    ax_c.set_ylim(0.0, 1.5)
    ax_c.set_xlabel(r"wall-clock time / min")
    ax_c.set_ylabel(r"1D $F(\phi)$ RMSE ($F_{\mathrm{ref}}\leq 8$) (kcal/mol)")
    tag(ax_c, "c")
    style_ax(ax_c)

    series(ax_d, t8_m, r_m, r_ms, t8_s, r_s, r_ss)
    ax_d.set_xlim(0.0, float(max(t8_m[-1], t8_s[-1])))
    ax_d.set_ylim(0.0, 1.5)
    ax_d.set_xlabel(r"$t$ / ps")
    ax_d.set_ylabel(r"1D $F(\phi)$ RMSE ($F_{\mathrm{ref}}\leq 8$) (kcal/mol)")
    tag(ax_d, "d")
    style_ax(ax_d)

    handles = [
        Line2D([0], [0], color=COLOR_META, lw=1.8, label="WT-MetaD"),
        Line2D([0], [0], color=COLOR_SFPT, lw=2.0, label="sFPT"),
        Line2D(
            [0],
            [0],
            color=COLOR_REF,
            ls=":",
            lw=1.5,
            label=rf"MetaD $\Delta G$={dG_ref:.2f}",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.958),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    fig.canvas.draw()
    for ax in (ax_c, ax_d):
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 + 0.016, p.width, p.height])

    EXP.mkdir(parents=True, exist_ok=True)
    png = EXP / "si_vacuum_2x2.png"
    pdf = EXP / "si_vacuum_2x2.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    if CACHE.is_file() and "--recompute" not in sys.argv:
        z = dict(np.load(CACHE))
        print(f"[cache] {CACHE}", flush=True)
    else:
        z = compute(device)
        np.savez_compressed(CACHE, **z)
        print(f"[cache] wrote {CACHE}", flush=True)
    plot_figure(z)


if __name__ == "__main__":
    main()
