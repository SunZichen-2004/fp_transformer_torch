#!/usr/bin/env python3
"""JCTC 1×4 sFPT CV-space transport scatter collages.

Claisen water, dialanine vacuum, dialanine water (R=320 and R=1024),
plus a WT-MetaD vacuum init scatter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
sys.path.insert(0, str(ROOT / "src"))

from fp_transformer_torch import Torsion  # noqa: E402
import torch  # noqa: E402

COLOR_SFPT = "#E67E22"
COLOR_META = "#2166AC"
PI_TICKS = (-np.pi, 0.0, np.pi)
PI_LABELS = (r"$-\pi$", r"$0$", r"$\pi$")

CLAISEN = (
    ROOT
    / "experiments"
    / "rearrangement_water_fpt"
    / "results"
    / "claisen_water_fpt_mh-0_g4_89110"
)
VAC = ROOT / "experiments" / "dialanine" / "results" / "dialanine_gaff_s4_90438"
VAC_INIT = ROOT / "experiments" / "dialanine" / "init_coords_2048.npy"
VAC_CV = ROOT / "experiments" / "dialanine" / "backbone_cv.txt"
WAT320 = (
    ROOT
    / "experiments"
    / "dialanine_water"
    / "sFPT"
    / "fair_compete_result"
    / "dialanine_water_sfpt_r320_g1_s1_90464"
)
WAT1024 = (
    ROOT
    / "experiments"
    / "dialanine_water"
    / "sFPT"
    / "fair_compete_result"
    / "dialanine_water_sfpt_r1024_g4_s2_91355"
)
WAT_INIT = (
    ROOT
    / "experiments"
    / "dialanine_water_meta"
    / "results"
    / "dialanine_water_metad_r256_s2_89603"
    / "init_coords_from_pre_md.npy"
)
WAT_CV = ROOT / "experiments" / "dialanine_water" / "mm_box" / "backbone_cv.txt"
METAD = (
    ROOT
    / "experiments"
    / "dialanine_meta"
    / "results"
    / "dialanine_metad_r64_s4_90426"
)

DT_PS = 0.1
STAGES = (5, 50, 500)
TITLES = (
    "initial distribution",
    r"$0.5$ ps",
    r"$5$ ps",
    r"$50$ ps",
)


def jctc_rcparams() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.15,
        "xtick.major.width": 1.15,
        "ytick.major.width": 1.15,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }


def tag_title(ax, letter: str, title: str, *, letter_x: float = -0.16) -> None:
    ax.set_title(title, loc="center", pad=8.0, fontsize=16, fontweight="normal")


def style_ax(ax) -> None:
    ax.tick_params(which="both", top=False, right=False)
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
        ax.spines[side].set_color("0.15")


def load_cv2(path: Path) -> np.ndarray:
    a = np.asarray(np.load(path), dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(a.shape[0], -1)
    if a.shape[1] < 2:
        raise ValueError(f"need 2 CVs in {path}, got {a.shape}")
    return a[:, :2]


def find_stage(run: Path, stage: int, suffix: str) -> Path:
    hits = sorted(run.glob(f"stage{stage:04d}_*{suffix}"))
    if not hits:
        raise FileNotFoundError(f"no stage {stage:04d} *{suffix} under {run}")
    return hits[0]


def last_stage_index(run: Path, suffix: str) -> int:
    idxs = []
    for p in run.glob(f"stage[0-9]*{suffix}"):
        try:
            idxs.append(int(p.name.split("_")[0].replace("stage", "")))
        except ValueError:
            continue
    if not idxs:
        raise FileNotFoundError(f"no stage*{suffix} under {run}")
    return max(idxs)


def torsion_indices(path: Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text().splitlines():
        nums = []
        for part in line.replace(",", " ").split():
            try:
                nums.append(int(part) - 1)
            except ValueError:
                continue
            if len(nums) == 4:
                rows.append(nums)
                nums = []
    return np.asarray(rows, dtype=np.int64)


def phi_psi_from_coords(coords: np.ndarray, idx_path: Path) -> np.ndarray:
    idx = torsion_indices(idx_path)
    x = torch.as_tensor(np.asarray(coords, dtype=np.float64))
    cv = Torsion(torch.as_tensor(idx, dtype=torch.long))(x).detach().cpu().numpy()
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    return cv[:, :2]


def subsample_init(path: Path, n: int) -> np.ndarray:
    loaded = np.load(path)
    idx = np.linspace(0, loaded.shape[0] - 1, int(n)).astype(np.int64)
    return loaded[idx]


def point_style(n: int) -> tuple[float, float]:
    n = max(int(n), 1)
    s = float(np.clip(280.0 / np.sqrt(n), 1.6, 14.0))
    alpha = float(np.clip(0.22 + 18.0 / np.sqrt(n), 0.28, 0.72))
    return s, alpha


def shared_lim(arrays: list[np.ndarray], *, pad: float = 0.06) -> tuple[float, float]:
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    span = max(hi - lo, 1e-3)
    lo -= pad * span
    hi += pad * span
    # round to 0.5
    lo = float(np.floor(lo * 2.0) / 2.0)
    hi = float(np.ceil(hi * 2.0) / 2.0)
    if hi <= lo:
        hi = lo + 0.5
    return lo, hi


def draw_scatter(
    ax,
    xy: np.ndarray,
    *,
    letter: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str | None,
    color: str,
    periodic: bool,
    letter_x: float = -0.16,
) -> None:
    s, alpha = point_style(xy.shape[0])
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=s,
        c=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if periodic:
        ax.set_xticks(PI_TICKS)
        ax.set_xticklabels(PI_LABELS)
        ax.set_yticks(PI_TICKS)
        if ylabel:
            ax.set_yticklabels(PI_LABELS)
        else:
            ax.set_yticklabels([])
    else:
        ticks = np.arange(0.0, max(xlim[1], ylim[1]) + 0.1, 2.0)
        ticks = ticks[(ticks >= xlim[0] - 1e-9) & (ticks <= xlim[1] + 1e-9)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if not ylabel:
            ax.set_yticklabels([])
    tag_title(ax, letter, title, letter_x=letter_x)
    style_ax(ax)


def collage(
    panels: list[tuple[np.ndarray, str]],
    *,
    out: Path,
    xlabel: str,
    ylabel: str,
    periodic: bool,
    color: str = COLOR_SFPT,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.rcParams.update(jctc_rcparams())
    xs = [p[0][:, 0] for p in panels]
    ys = [p[0][:, 1] for p in panels]
    if xlim is None:
        xlim = shared_lim(xs) if not periodic else (-np.pi, np.pi)
    if ylim is None:
        ylim = shared_lim(ys) if not periodic else (-np.pi, np.pi)
    fig = plt.figure(figsize=(14.8, 4.70))
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        wspace=0.02,
        left=0.058,
        right=0.998,
        bottom=0.145,
        top=0.86,
    )
    letters = "abcd"
    axes = []
    for i, (xy, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        draw_scatter(
            ax,
            xy,
            letter=letters[i],
            title=title,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel if i == 0 else None,
            color=color,
            periodic=periodic,
            letter_x=-0.22 if i == 0 else -0.08,
        )
        axes.append(ax)
    fig.canvas.draw()
    p0 = axes[0].get_position()
    gap = 0.005
    for i, ax in enumerate(axes):
        ax.set_position([p0.x0 + i * (p0.width + gap), p0.y0, p0.width, p0.height])
    fig.canvas.draw()
    for i, (ax, letter) in enumerate(zip(axes, letters)):
        p = ax.get_position()
        dx = 0.010 if i == 0 else 0.003
        fig.text(
            p.x0 - dx,
            p.y1 + 0.008,
            letter,
            ha="right",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            clip_on=False,
        )
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out.with_suffix('.png')}  xlim={xlim} ylim={ylim}", flush=True)


def claisen_panels() -> list[tuple[np.ndarray, str]]:
    init = load_cv2(CLAISEN / "stage_init_cv.npy")
    out = [(init, TITLES[0])]
    for st, title in zip(STAGES, TITLES[1:]):
        try:
            p = find_stage(CLAISEN, st, "_cv.npy")
            out.append((load_cv2(p), title))
        except FileNotFoundError:
            last = last_stage_index(CLAISEN, "_cv.npy")
            p = find_stage(CLAISEN, last, "_cv.npy")
            t_ps = float(last) * DT_PS
            print(f"[warn] claisen: stage {st} missing → {last} ({t_ps:g} ps)", flush=True)
            out.append((load_cv2(p), rf"${t_ps:g}$ ps"))
    return out


def sFPT_phi_panels(run: Path, init_xy: np.ndarray) -> list[tuple[np.ndarray, str]]:
    out = [(init_xy, TITLES[0])]
    for st, title in zip(STAGES, TITLES[1:]):
        try:
            p = find_stage(run, st, "_dihedral.npy")
            out.append((load_cv2(p), title))
        except FileNotFoundError:
            last = last_stage_index(run, "_dihedral.npy")
            p = find_stage(run, last, "_dihedral.npy")
            t_ps = float(last) * DT_PS
            print(f"[warn] {run.name}: stage {st} missing → {last} ({t_ps:g} ps)", flush=True)
            out.append((load_cv2(p), rf"${t_ps:g}$ ps"))
    return out


def plot_metad_init(out: Path) -> None:
    cv = load_cv2(METAD / "trajectory_cv.npy")[:64]
    plt.rcParams.update(jctc_rcparams())
    fig = plt.figure(figsize=(4.8, 4.75))
    ax = fig.add_axes((0.20, 0.15, 0.76, 0.72))
    draw_scatter(
        ax,
        cv,
        letter="a",
        title="initial distribution",
        xlim=(-np.pi, np.pi),
        ylim=(-np.pi, np.pi),
        xlabel=r"$\phi$ / rad",
        ylabel=r"$\psi$ / rad",
        color=COLOR_META,
        periodic=True,
        letter_x=-0.22,
    )
    fig.canvas.draw()
    p = ax.get_position()
    fig.text(
        p.x0 - 0.010,
        p.y1 + 0.008,
        "a",
        ha="right",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        clip_on=False,
    )
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out.with_suffix('.png')}  n={cv.shape[0]}", flush=True)


def main() -> None:
    claisen_out = ROOT / "experiments" / "rearrangement_water_fpt" / "SI"
    claisen_out.mkdir(parents=True, exist_ok=True)
    collage(
        claisen_panels(),
        out=claisen_out / "si_cv_transport_1x4",
        xlabel=r"$d_1$ (O7–C8) / Å",
        ylabel=r"$d_2$ (C1–C10) / Å",
        periodic=False,
    )

    vac_out = ROOT / "experiments" / "dialanine" / "SI_dialanine_vacuum"
    vac_out.mkdir(parents=True, exist_ok=True)
    vac_init = phi_psi_from_coords(np.load(VAC_INIT), VAC_CV)
    collage(
        sFPT_phi_panels(VAC, vac_init),
        out=vac_out / "si_cv_transport_1x4",
        xlabel=r"$\phi$ / rad",
        ylabel=r"$\psi$ / rad",
        periodic=True,
    )
    plot_metad_init(vac_out / "metad_init_scatter")

    wat_out = ROOT / "experiments" / "dialanine_water" / "dialanine_water_SI"
    wat_out.mkdir(parents=True, exist_ok=True)
    raw = np.load(WAT_INIT)
    init320 = phi_psi_from_coords(subsample_init(WAT_INIT, 320), WAT_CV)
    init1024 = phi_psi_from_coords(subsample_init(WAT_INIT, 1024), WAT_CV)
    print(f"[init] water npy {raw.shape} → R=320 / R=1024", flush=True)
    collage(
        sFPT_phi_panels(WAT320, init320),
        out=wat_out / "si_cv_transport_r320",
        xlabel=r"$\phi$ / rad",
        ylabel=r"$\psi$ / rad",
        periodic=True,
    )
    collage(
        sFPT_phi_panels(WAT1024, init1024),
        out=wat_out / "si_cv_transport_r1024",
        xlabel=r"$\phi$ / rad",
        ylabel=r"$\psi$ / rad",
        periodic=True,
    )


if __name__ == "__main__":
    main()
