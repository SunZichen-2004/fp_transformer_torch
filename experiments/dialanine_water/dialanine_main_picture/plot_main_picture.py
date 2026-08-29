#!/usr/bin/env python3
"""2×4 main figure: alanine dipeptide in vacuum (a–d) and explicit water (e–h)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments"
OUT = EXP / "dialanine_water" / "dialanine_main_picture"

VAC_KDE_2D = (
    EXP
    / "dialanine"
    / "results"
    / "dialanine_gaff_s0_87188"
    / "kde_fes"
    / "kde_fes_last50pct_lam0p1.npz"
)
VAC_METAD_2D = (
    EXP
    / "dialanine_meta"
    / "results"
    / "dialanine_metad_r64_s0_87211"
    / "live"
    / "fes.npz"
)
VAC_1D = EXP / "dialanine" / "convergence_analysis" / "fes_1d_all_groups.npz"
VAC_DG = EXP / "dialanine" / "convergence_analysis" / "fes_rmse_vs_time.npz"

WAT_2D = EXP / "dialanine_water" / "sFPT" / "convergence_1024" / "fes_2d_last_frac.npz"
WAT_1D = EXP / "dialanine_water" / "sFPT" / "convergence_1024" / "fes_1d_all_groups.npz"
WAT_DG = EXP / "dialanine_water" / "sFPT" / "convergence_1024" / "fes_rmse_vs_time.npz"
WAT_2D_LAST50 = (
    EXP / "dialanine_water" / "sFPT" / "convergence_1024" / "fes_2d_last50.npz"
)
WAT_LAST_FRAC = 0.5
REF_BAND = 0.5

FE_LABEL = r"free energy / kcal mol$^{-1}$"
COLOR_META = "#2166AC"   # blue, consistent across all panels
COLOR_SFPT = "#E67E22"   # orange, consistent across all panels
COLOR_US = "#2CA02C"     # green reference
FES_CMAP = "RdBu_r"      # low F = blue, intermediate = white, high F = red
EV_TO_KCAL = 23.06054783061903
KB_EV_PER_K = 8.617330337217213e-5
KT_KCAL = KB_EV_PER_K * 300.0 * EV_TO_KCAL
DG_YLABEL = r"$\Delta G$ / kcal mol$^{-1}$"
PI_TICKS = (-np.pi, 0.0, np.pi)
PI_LABELS = (r"$-\pi$", r"$0$", r"$\pi$")

VAC_VMAX = 12.0
WAT_VMAX = 16.0
VAC_1D_YLIM = 15.0
VAC_N_LAST = 10  # 10× original group_size=5  → 50 stages
WAT_N_LAST = 10  # 10× original group_size=20 → 200 stages


def nature_rcparams() -> dict[str, object]:
    # JCTC/ACS double-column is 7.0 in. This canvas is 14 in wide, so
    # lettering is set ~2× published size: ≥8 pt after scaling (ACS min 4.5 pt).
    # Line weights ≥1.0 pt here → ≥0.5 pt at published size.
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }


def relative_fes(f: np.ndarray) -> np.ndarray:
    out = np.asarray(f, dtype=np.float64)
    return out - np.nanmin(out)


def boltzmann_combine(f_rows: np.ndarray, kT: float = KT_KCAL) -> np.ndarray:
    f_rows = np.asarray(f_rows, dtype=np.float64)
    p = np.mean(np.exp(-np.clip(f_rows, None, 80.0) / kT), axis=0)
    return relative_fes(-kT * np.log(np.clip(p, 1e-300, None)))


def fes_cmap():
    cmap = plt.get_cmap(FES_CMAP).copy()
    cmap.set_over("white")
    cmap.set_bad("white")
    return cmap


def tag_title(
    ax, letter: str, title: str, *, letter_x: float = -0.08, title_fontsize: float = 17
) -> None:
    ax.text(
        letter_x, 1.04, f"{letter}", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=18, fontweight="bold", clip_on=False
    )
    ax.set_title(
        title if title else " ",
        loc="center",
        pad=8.0,
        fontsize=title_fontsize,
        fontweight="normal",
    )


def smooth_fes_torus(phi, psi, F, *, sigma: float = 1.1, n_fine: int = 361):
    """Periodic Gaussian smooth + upsample on the (φ, ψ) torus."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    Z = relative_fes(F)
    Zs = gaussian_filter(np.asarray(Z, dtype=np.float64), sigma=float(sigma), mode="wrap")
    phi_f = np.linspace(float(phi.min()), float(phi.max()), int(n_fine))
    psi_f = np.linspace(float(psi.min()), float(psi.max()), int(n_fine))
    interp = RegularGridInterpolator(
        (psi, phi), Zs, bounds_error=False, fill_value=None
    )
    g_phi, g_psi = np.meshgrid(phi_f, psi_f, indexing="xy")
    Zf = interp(np.stack([g_psi.ravel(), g_phi.ravel()], axis=-1)).reshape(g_phi.shape)
    return phi_f, psi_f, relative_fes(Zf)


def draw_fes2d(
    ax, phi, psi, F, *, vmax: float, title: str, letter: str,
    show_ylabel: bool = True, smooth: bool = False,
):
    if smooth:
        phi, psi, Z = smooth_fes_torus(phi, psi, F)
    else:
        Z = relative_fes(F)
    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    im = ax.contourf(
        phi, psi, Z, levels=levels, cmap=cmap, norm=norm, extend="max"
    )
    ax.contour(
        phi,
        psi,
        np.clip(Z, 0.0, float(vmax)),
        levels=levels[::3],
        colors="0.45",
        linewidths=0.35,
        alpha=0.45,
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_yticks(PI_TICKS)
    if show_ylabel:
        ax.set_yticklabels(PI_LABELS)
        ax.set_ylabel(r"$\psi$ / rad")
        tag_title(ax, letter, title)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        tag_title(ax, letter, title, letter_x=0.02)
    ax.set_xlabel(r"$\phi$ / rad")
    ax.tick_params(which="both", top=False, right=False)
    return im


def draw_fes1d_vac(
    ax, xi, f_meta, e_meta, f_sfpt, e_sfpt, title: str, letter: str
) -> None:
    y_m = relative_fes(f_meta)
    y_s = relative_fes(f_sfpt)
    ax.fill_between(
        xi, y_m - e_meta, y_m + e_meta, color=COLOR_META, alpha=0.18, lw=0, zorder=1
    )
    ax.plot(xi, y_m, color=COLOR_META, lw=1.8, label="WT-MetaD", zorder=2)
    ax.fill_between(
        xi, y_s - e_sfpt, y_s + e_sfpt, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3
    )
    ax.plot(xi, y_s, color=COLOR_SFPT, lw=2.1, label="sFPT", zorder=4)
    ax.axvline(0.0, color="0.55", ls=":", lw=1.0, zorder=0)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_ylim(0.0, VAC_1D_YLIM)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel(r"$\phi$ / rad")
    ax.set_ylabel(FE_LABEL)
    tag_title(ax, letter, title)
    ax.tick_params(which="both", top=False, right=False)


def draw_fes1d_wat(
    ax, xi, f_meta, e_meta, f_sfpt, e_sfpt, title: str, letter: str
) -> None:
    y_m = relative_fes(f_meta)
    y_s = relative_fes(f_sfpt)
    y_mlo = relative_fes(f_meta - e_meta)
    y_mhi = relative_fes(f_meta + e_meta)
    y_slo = relative_fes(f_sfpt - e_sfpt)
    y_shi = relative_fes(f_sfpt + e_sfpt)
    ax.fill_between(xi, y_mlo, y_mhi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
    ax.plot(xi, y_m, color=COLOR_META, lw=1.8, label="WT-MetaD", zorder=2)
    ax.fill_between(xi, y_slo, y_shi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
    ax.plot(xi, y_s, color=COLOR_SFPT, lw=2.1, label="sFPT", zorder=4)
    ax.axvline(0.0, color="0.55", ls=":", lw=1.0, zorder=0)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_ylim(0.0, WAT_VMAX)
    ax.set_xlabel(r"$\phi$ / rad")
    ax.set_ylabel(FE_LABEL)
    tag_title(ax, letter, title)
    ax.tick_params(which="both", top=False, right=False)


def draw_dg(
    ax,
    *,
    t_meta,
    y_meta,
    e_meta,
    t_sfpt,
    y_sfpt,
    e_sfpt,
    dG_ref: float,
    ref_label: str,
    letter: str,
    meta_label: str,
    sfpt_label: str,
    xlabel: str = r"wall-clock time / h",
    ylim=None,
    xlim=None,
    ref_band: float = REF_BAND,
) -> None:
    if ref_band is not None and float(ref_band) > 0:
        ax.axhspan(
            dG_ref - float(ref_band),
            dG_ref + float(ref_band),
            color=COLOR_US,
            alpha=0.18,
            lw=0,
            zorder=0,
        )
    ax.axhline(
        dG_ref,
        color=COLOR_US,
        ls=":",
        lw=1.6,
        zorder=1,
        label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
    )
    ax.plot(t_meta, y_meta, color=COLOR_META, lw=1.8, label=meta_label, zorder=2)
    if e_meta is not None:
        ax.fill_between(
            t_meta,
            y_meta - e_meta,
            y_meta + e_meta,
            color=COLOR_META,
            alpha=0.22,
            lw=0,
            zorder=1,
        )
    ax.plot(t_sfpt, y_sfpt, color=COLOR_SFPT, lw=2.1, label=sfpt_label, zorder=4)
    if e_sfpt is not None:
        ax.fill_between(
            t_sfpt,
            y_sfpt - e_sfpt,
            y_sfpt + e_sfpt,
            color=COLOR_SFPT,
            alpha=0.25,
            lw=0,
            zorder=3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(DG_YLABEL)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vals = [y_meta, y_sfpt, np.asarray([dG_ref])]
        if e_meta is not None:
            vals.extend([y_meta - e_meta, y_meta + e_meta])
        if e_sfpt is not None:
            vals.extend([y_sfpt - e_sfpt, y_sfpt + e_sfpt])
        stack = np.concatenate([np.ravel(v) for v in vals])
        stack = stack[np.isfinite(stack)]
        ymin, ymax = float(np.min(stack)), float(np.max(stack))
        pad = 0.08 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.tick_params(which="both", top=False, right=False)
    tag_title(
        ax,
        letter,
        r"$\Delta G=k_BT\ln(P_{\phi<0}/P_{\phi>0})$",
        title_fontsize=14,
    )


def load_vac_metad_2d():
    z = np.load(VAC_METAD_2D)
    phi = np.asarray(z["phi"], dtype=np.float64)
    psi = np.asarray(z["psi"], dtype=np.float64)
    f = relative_fes(np.asarray(z["F"], dtype=np.float64) * EV_TO_KCAL)
    return phi, psi, f


def metad_2d_avg_last_frac(
    hills_path: Path,
    phi,
    psi,
    *,
    last_frac: float,
    n_replicas: int = 64,
    bias_factor: float = 15.0,
    sigma=(0.35, 0.35),
    device,
) -> np.ndarray:
    """Mean F(φ,ψ) over the last ``last_frac`` of WT-MetaD deposits."""
    import torch

    two_pi = 2.0 * np.pi
    hills = np.load(hills_path)
    n_hills = (
        int(hills["n_hills"])
        if "n_hills" in hills.files
        else int(hills["centers"].shape[0])
    )
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float64)
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    n_dep = n_hills // int(n_replicas)
    k0 = max(1, int(np.ceil((1.0 - float(last_frac)) * n_dep)))
    scale = float(bias_factor) / (float(bias_factor) - 1.0)
    sig = np.asarray(sigma, dtype=np.float64).reshape(1, 1, 2)
    v = torch.zeros(grid.shape[0], dtype=torch.float64, device=device)
    grid_t = torch.as_tensor(grid, dtype=torch.float64, device=device)
    sig_t = torch.as_tensor(sig, dtype=torch.float64, device=device)
    acc = torch.zeros_like(v)
    n_snap = 0
    for k in range(1, n_dep + 1):
        i0 = (k - 1) * int(n_replicas)
        i1 = k * int(n_replicas)
        c = torch.as_tensor(centers[i0:i1], dtype=torch.float64, device=device)
        w = torch.as_tensor(heights[i0:i1], dtype=torch.float64, device=device)
        diff = grid_t[:, None, :] - c[None, :, :]
        diff = diff - two_pi * torch.round(diff / two_pi)
        gauss = torch.exp(-0.5 * ((diff / sig_t) ** 2).sum(-1))
        v = v + (gauss @ w)
        if k >= k0:
            acc = acc + v
            n_snap += 1
        if k == 1 or k % max(1, n_dep // 5) == 0 or k == n_dep:
            print(
                f"    [metad2d] {hills_path.parent.name} deposit {k}/{n_dep}",
                flush=True,
            )
    f = (-scale * (acc / max(n_snap, 1)).detach().cpu().numpy()).reshape(g1.shape)
    return relative_fes(f * EV_TO_KCAL)


def water_2d_last_frac(zdw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """sFPT + WT-MetaD 2D FES from the last 50% of each trajectory, mean of 2 seeds."""
    if WAT_2D_LAST50.is_file():
        z = np.load(WAT_2D_LAST50)
        if float(z["last_frac"]) == float(WAT_LAST_FRAC):
            print(f"[2d] load cache {WAT_2D_LAST50.name}", flush=True)
            return z["phi"], z["psi"], z["F_sfpt"], z["F_metad"]
        z.close()

    import torch

    sys.path.insert(0, str(EXP / "dialanine_water"))
    from plot_convergence_multi import kde_2d_last_frac, mean_std  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2d] last {100 * WAT_LAST_FRAC:.0f}%  device={device}", flush=True)
    z2 = np.load(WAT_2D)
    phi = np.asarray(z2["phi"], dtype=np.float64)
    psi = np.asarray(z2["psi"], dtype=np.float64)
    z2.close()
    fpt_dirs = [Path(str(x)) for x in zdw["fpt_dirs"]]
    metad_dirs = [Path(str(x)) for x in zdw["metad_dirs"]]
    lam = float(zdw["lambda_target"])

    f_s_list = []
    for d in fpt_dirs:
        print(f"[2d] sFPT {d.name}", flush=True)
        f_s_list.append(
            kde_2d_last_frac(
                d,
                phi=phi,
                psi=psi,
                last_frac=WAT_LAST_FRAC,
                temperature=300.0,
                device=device,
                chunk=2048,
                lambda_target=lam,
            )
        )
    f_sfpt, _ = mean_std(np.stack(f_s_list, 0), 0)
    f_sfpt = relative_fes(f_sfpt)

    f_m_list = []
    for md in metad_dirs:
        hills = md / "hills.npz"
        print(f"[2d] MetaD {md}", flush=True)
        f_m_list.append(
            metad_2d_avg_last_frac(
                hills,
                phi,
                psi,
                last_frac=WAT_LAST_FRAC,
                device=device,
            )
        )
    f_meta, _ = mean_std(np.stack(f_m_list, 0), 0)
    f_meta = relative_fes(f_meta)
    np.savez_compressed(
        WAT_2D_LAST50,
        phi=phi,
        psi=psi,
        F_sfpt=f_sfpt,
        F_metad=f_meta,
        last_frac=np.float64(WAT_LAST_FRAC),
        fpt_dirs=np.asarray([str(d) for d in fpt_dirs]),
        metad_dirs=np.asarray([str(d) for d in metad_dirs]),
    )
    print(f"[2d] wrote {WAT_2D_LAST50}", flush=True)
    return phi, psi, f_sfpt, f_meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(nature_rcparams())

    z_kde = np.load(VAC_KDE_2D)
    phi_s, psi_s = z_kde["phi"], z_kde["psi"]
    F_vac_sfpt = z_kde["F_kcal"]
    phi_m, psi_m, F_vac_meta = load_vac_metad_2d()

    z1 = np.load(VAC_1D)
    xi_v = z1["xi"]
    f_meta_v = np.asarray(z1["F1d_metad_mean"], dtype=np.float64)
    e_meta_v = np.asarray(z1["F1d_metad_std"], dtype=np.float64)
    f_kde_mean = np.asarray(z1["F1d_kde_mean"], dtype=np.float64)
    e_kde = np.asarray(z1["F1d_kde_std"], dtype=np.float64)
    f_sfpt_v = boltzmann_combine(f_kde_mean[-VAC_N_LAST:])
    e_sfpt_v = np.mean(e_kde[-VAC_N_LAST:], axis=0)
    t_lo_v, t_hi_v = float(z1["t_ps"][-VAC_N_LAST]), float(z1["t_ps"][-1])

    zd = np.load(VAC_DG)
    tw_meta_v = np.asarray(zd["tw_metad_h"], dtype=np.float64) * 60.0
    tw_sfpt_v = np.asarray(zd["tw_kde_h"], dtype=np.float64) * 60.0
    dG_meta_v = np.asarray(zd["dG_metad"], dtype=np.float64)
    dG_meta_e_v = np.asarray(zd["dG_metad_std"], dtype=np.float64)
    dG_sfpt_v = np.asarray(zd["dG_kde"], dtype=np.float64)
    dG_sfpt_e_v = np.asarray(zd["dG_kde_std"], dtype=np.float64)
    dG_ref_v = float(zd["dG_metad_ref"])
    xlim_d = (0.0, float(tw_meta_v[-1]))
    print(f"[d] xlim=0–{xlim_d[1]:.2f} min  (MetaD wall-clock max)", flush=True)

    zdw = np.load(WAT_DG)
    phi_w, psi_w, F_wat_sfpt, F_wat_meta = water_2d_last_frac(zdw)

    z1w = np.load(WAT_1D)
    xi_w = z1w["xi"]
    f_meta_w = z1w["F1d_metad_mean"]
    e_meta_w = z1w["F1d_metad_std"]
    runs = np.asarray(z1w["F1d_kde_runs"], dtype=np.float64)
    t_1d = np.asarray(z1w["t_ps"], dtype=np.float64)
    # Keep panel g on the same 122–140 ps window as the current main picture.
    g_sel = (t_1d >= 122.0) & (t_1d <= 140.0)
    if int(np.sum(g_sel)) < 2:
        g_sel = np.zeros(t_1d.size, dtype=bool)
        g_sel[-WAT_N_LAST:] = True
    last = [boltzmann_combine(runs[i, g_sel]) for i in range(runs.shape[0])]
    last = np.stack(last, axis=0)
    f_sfpt_w = np.mean(last, axis=0)
    e_sfpt_w = np.std(last, axis=0, ddof=1)
    t_lo_w, t_hi_w = float(t_1d[g_sel][0]), float(t_1d[g_sel][-1])

    tw_meta_w = np.asarray(zdw["tw_metad_h"], dtype=np.float64)
    tw_sfpt_w = np.asarray(zdw["tw_kde_h"], dtype=np.float64)
    dG_meta_w = zdw["dG_metad"]
    dG_meta_e = zdw["dG_metad_std"]
    dG_sfpt_w = zdw["dG_kde"]
    dG_sfpt_e = zdw["dG_kde_std"]
    dG_ref_w = float(zdw["dG_us_ref"])
    xlim_h = (0.0, float(tw_meta_w[-1]))
    print(f"[h] xlim=0–{xlim_h[1]:.4f} h  (MetaD wall-clock max)", flush=True)

    # Tall enough that bottom x-labels of e–h stay inside the canvas.
    fig_w, fig_h = 14.0, 9.20
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        wspace=0.38,
        hspace=0.62,
        left=0.055,
        right=0.985,
        top=0.88,
        bottom=0.10,
    )
    fig.text(
        0.52, 0.975,
        "Alanine dipeptide in vacuum",
        ha="center", va="top", fontsize=18, fontweight="bold",
    )
    water_title = fig.text(
        0.52, 0.48,
        "Alanine dipeptide in explicit water",
        ha="center", va="bottom", fontsize=18, fontweight="bold",
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2])
    ax_h = fig.add_subplot(gs[1, 3])

    im_a = draw_fes2d(
        ax_a, phi_s, psi_s, F_vac_sfpt, vmax=VAC_VMAX, title="sFPT", letter="a",
        smooth=True,
    )
    draw_fes2d(
        ax_b, phi_m, psi_m, F_vac_meta, vmax=VAC_VMAX, title="WT-MetaD", letter="b",
        show_ylabel=False,
    )
    draw_fes1d_vac(
        ax_c,
        xi_v,
        f_meta_v,
        e_meta_v,
        f_sfpt_v,
        e_sfpt_v,
        title=r"$F(\phi)$",
        letter="c",
    )
    draw_dg(
        ax_d,
        t_meta=tw_meta_v,
        y_meta=dG_meta_v,
        e_meta=dG_meta_e_v,
        t_sfpt=tw_sfpt_v,
        y_sfpt=dG_sfpt_v,
        e_sfpt=dG_sfpt_e_v,
        dG_ref=dG_ref_v,
        ref_label="Reference",
        letter="d",
        meta_label="WT-MetaD",
        sfpt_label="sFPT",
        xlabel=r"wall-clock time / min",
        xlim=xlim_d,
    )

    im_e = draw_fes2d(
        ax_e,
        phi_w,
        psi_w,
        F_wat_sfpt,
        vmax=WAT_VMAX,
        title="sFPT",
        letter="e",
    )
    draw_fes2d(
        ax_f,
        phi_w,
        psi_w,
        F_wat_meta,
        vmax=WAT_VMAX,
        title="WT-MetaD",
        letter="f",
        show_ylabel=False,
    )
    draw_fes1d_wat(
        ax_g,
        xi_w,
        f_meta_w,
        e_meta_w,
        f_sfpt_w,
        e_sfpt_w,
        title=r"$F(\phi)$",
        letter="g",
    )
    draw_dg(
        ax_h,
        t_meta=tw_meta_w,
        y_meta=dG_meta_w,
        e_meta=dG_meta_e,
        t_sfpt=tw_sfpt_w,
        y_sfpt=dG_sfpt_w,
        e_sfpt=dG_sfpt_e,
        dG_ref=dG_ref_w,
        ref_label="US",
        letter="h",
        meta_label="WT-MetaD",
        sfpt_label="sFPT",
        ylim=(2.5, 6.0),
        xlim=xlim_h,
    )

    for ax in (ax_d, ax_h):
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax_h.set_ylim(2.5, 6.0)
    ax_h.set_xlim(*xlim_h)
    ax_h.set_yticks([2.5, 3.0, 4.0, 5.0, 6.0])
    ax_c.set_ylim(0.0, VAC_1D_YLIM)
    ax_c.set_yticks([0, 5, 10, 15])

    fig.legend(
        handles=[
            Line2D([0], [0], color=COLOR_META, lw=2.4, label="WT-MetaD"),
            Line2D([0], [0], color=COLOR_SFPT, lw=2.6, label="sFPT"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=2,
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        borderaxespad=0.0,
        columnspacing=1.2,
    )

    fig.canvas.draw()

    def _stick_to_left(ax_left, ax_right, gap: float = 0.004) -> None:
        pl, pr = ax_left.get_position(), ax_right.get_position()
        ax_right.set_position([pl.x1 + gap, pl.y0, pr.width, pr.height])

    _stick_to_left(ax_a, ax_b)
    _stick_to_left(ax_e, ax_f)
    for ax in (ax_b, ax_f):
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    pe = ax_e.get_position()
    water_title.set_position((0.52, pe.y1 + 0.072))

    def _cbar_hug_left(im, ax_ref, vmax: float):
        p = ax_ref.get_position()
        cax = fig.add_axes([p.x1 + 0.004, p.y0, 0.008, p.height])
        cb = fig.colorbar(im, cax=cax, extend="max")
        cb.set_label(r"kcal mol$^{-1}$", fontsize=14, labelpad=3)
        cb.ax.tick_params(labelsize=13, length=3.2, width=1.0)
        cb.set_ticks(np.linspace(0.0, float(vmax), 5))
        return cb

    _cbar_hug_left(im_a, ax_b, VAC_VMAX)
    _cbar_hug_left(im_e, ax_f, WAT_VMAX)

    png = OUT / "dialanine_main_picture.png"
    pdf = OUT / "dialanine_main_picture.pdf"
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0.08)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0.08)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)
    print(
        f"[1d] vacuum last {VAC_N_LAST} groups  t={t_lo_v:g}–{t_hi_v:g} ps",
        flush=True,
    )
    print(
        f"[1d] water last {WAT_N_LAST} groups  t={t_lo_w:g}–{t_hi_w:g} ps  "
        f"n_seeds={runs.shape[0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
