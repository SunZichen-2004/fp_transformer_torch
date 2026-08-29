#!/usr/bin/env python3
"""Plot Claisen FPT CV panels (transitions or selected λ snapshots).

Default (no ``--config``): consecutive-stage transitions for all ``stage####_*_cv.npy``.

With ``--config`` YAML: plot an ordered list of selected heal/stage/init CV clouds
in a fixed grid (e.g. 2×4), JCTC double-column width. Inner shared axes omit
tick labels; panel tags show λ only (no stage index, no figure title).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

STAGE_CV_RE = re.compile(r"^stage(\d+)_.*_cv\.npy$")
LAMBDA_RE = re.compile(r"lambda(\d+p\d+)", re.IGNORECASE)
# JCTC double-column width (ACS): ~178 mm
JCTC_DOUBLE_COL_IN = 178.0 / 25.4


def resolve_result_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        path = path.parent
    if path.name == "d1_d2":
        return path.parent
    return path


def _load_cv_file(path: Path) -> np.ndarray:
    cv = np.asarray(np.load(path), dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    if cv.ndim != 2 or cv.shape[1] < 2:
        raise ValueError(f"expected [R, ≥2] CV in {path}, got {cv.shape}")
    return cv[:, :2]


def load_stage_cvs(result_dir: Path) -> list[tuple[int, str, np.ndarray]]:
    items: list[tuple[int, str, np.ndarray]] = []
    for p in result_dir.glob("stage*_cv.npy"):
        m = STAGE_CV_RE.match(p.name)
        if m is None:
            continue
        items.append((int(m.group(1)), p.stem.replace("_cv", ""), _load_cv_file(p)))
    items.sort(key=lambda t: t[0])
    if len(items) < 2:
        raise FileNotFoundError(
            f"need ≥2 stage####_*_cv.npy under {result_dir}, found {len(items)}"
        )
    return items


def _parse_lambda_token(token: str) -> float:
    return float(str(token).replace("p", "."))


def _lambda_from_name(name: str) -> float | None:
    m = LAMBDA_RE.search(name)
    if m is None:
        return None
    return _parse_lambda_token(m.group(1))


def resolve_panel_cv(
    result_dir: Path,
    spec: dict,
) -> tuple[str, np.ndarray]:
    """Return (label, cv[R,2]) for one config panel entry."""
    label = spec.get("label")
    if "match" in spec:
        path = result_dir / str(spec["match"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if label is None:
            lam = _lambda_from_name(path.name)
            label = "init" if lam is None else rf"$\lambda = {lam:g}$"
        return str(label), _load_cv_file(path)

    if "tag" in spec:
        path = result_dir / f"{spec['tag']}_cv.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        if label is None:
            lam = _lambda_from_name(path.name)
            label = "init" if lam is None else rf"$\lambda = {lam:g}$"
        return str(label), _load_cv_file(path)

    if "lambda" in spec:
        target = float(spec["lambda"])
        phase = str(spec.get("phase", "any")).lower()
        cands: list[Path] = []
        if phase in ("heal", "any"):
            cands.extend(sorted(result_dir.glob("heal*_cv.npy")))
        if phase in ("stage", "any"):
            cands.extend(sorted(result_dir.glob("stage*_cv.npy")))
        best: Path | None = None
        best_err = 1e9
        for p in cands:
            lam = _lambda_from_name(p.name)
            if lam is None:
                continue
            err = abs(lam - target)
            if err < best_err:
                best_err = err
                best = p
        if best is None or best_err > 1e-6 + 1e-9 * abs(target):
            raise FileNotFoundError(
                f"no {phase} *_cv.npy with λ≈{target} under {result_dir}"
            )
        if label is None:
            label = rf"$\lambda = {target:g}$"
        return str(label), _load_cv_file(best)

    raise ValueError(f"panel spec needs match/tag/lambda, got keys={list(spec)}")


def jctc_rcparams() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    }


def _savefig(fig, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches=None, pad_inches=0.02)
    fig.savefig(out, **kw)
    fig.savefig(out.with_suffix(".pdf"), **kw)
    print(f"[ok] wrote {out}", flush=True)
    print(f"[ok] wrote {out.with_suffix('.pdf')}", flush=True)


def _legend_lambda_text(lab: str) -> str:
    """Legend entry text: 'init' or 'λ = xx'."""
    s = str(lab).strip()
    if s.lower() == "init":
        return "init"
    s = s.replace(r"$\lambda = ", "").replace("$", "").replace(r"\lambda = ", "")
    s = s.replace("λ = ", "").replace("lambda = ", "").strip()
    return rf"$\lambda = {s}$"


def plot_selected_snapshots(
    panels: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    nrows: int = 2,
    ncols: int = 4,
    point_size: float = 4.0,
    point_alpha: float = 0.55,
) -> Path:
    """One CV cloud per panel; λ label at upper-right."""
    n = len(panels)
    if n != nrows * ncols:
        raise ValueError(f"need nrows*ncols={nrows * ncols} panels, got {n}")

    all_cv = np.concatenate([cv for _, cv in panels], axis=0)
    d1_lo, d1_hi = float(all_cv[:, 0].min()), float(all_cv[:, 0].max())
    d2_lo, d2_hi = float(all_cv[:, 1].min()), float(all_cv[:, 1].max())
    pad1 = 0.05 * max(d1_hi - d1_lo, 1e-6)
    pad2 = 0.05 * max(d2_hi - d2_lo, 1e-6)
    xlim = (d1_lo - pad1, d1_hi + pad1)
    ylim = (d2_lo - pad2, d2_hi + pad2)

    panel_w = JCTC_DOUBLE_COL_IN / ncols * 0.92
    fig_w = JCTC_DOUBLE_COL_IN
    fig_h = panel_w * nrows * 1.05 + 0.35

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.10,
        top=0.96,
        wspace=0.06,
        hspace=0.12,
    )

    for k, (lab, cv) in enumerate(panels):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        ax.scatter(
            cv[:, 0],
            cv[:, 1],
            s=point_size,
            c="#2166ac",
            alpha=point_alpha,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_box_aspect(1)
        ax.tick_params(which="both", top=False, right=False)
        ax.text(
            0.96,
            0.96,
            _legend_lambda_text(lab),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
        if c == 0:
            ax.set_ylabel(r"$d_2$ / Å")
        else:
            ax.tick_params(labelleft=False)
        if r == nrows - 1:
            ax.set_xlabel(r"$d_1$ / Å")
        else:
            ax.tick_params(labelbottom=False)

    _savefig(fig, out_path)
    plt.close(fig)
    return Path(out_path)


def plot_selected_transitions(
    panels: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    nrows: int = 2,
    ncols: int = 4,
    point_size: float = 2.2,
    point_alpha: float = 0.35,
    blue_alpha: float | None = None,
    red_alpha: float | None = None,
    line_alpha: float = 0.35,
    max_replicas: int | None = None,
) -> Path:
    """Consecutive λ_i → λ_{i+1} transitions (blue→red + light-red links)."""
    if len(panels) < 2:
        raise ValueError("need ≥2 selected frames for transitions")
    n_trans = len(panels) - 1
    n_slots = nrows * ncols
    if n_trans > n_slots:
        raise ValueError(
            f"transitions={n_trans} exceed grid {nrows}×{ncols}={n_slots}"
        )

    all_cv = np.concatenate([cv for _, cv in panels], axis=0)
    d1_lo, d1_hi = float(all_cv[:, 0].min()), float(all_cv[:, 0].max())
    d2_lo, d2_hi = float(all_cv[:, 1].min()), float(all_cv[:, 1].max())
    pad1 = 0.05 * max(d1_hi - d1_lo, 1e-6)
    pad2 = 0.05 * max(d2_hi - d2_lo, 1e-6)
    xlim = (d1_lo - pad1, d1_hi + pad1)
    ylim = (d2_lo - pad2, d2_hi + pad2)

    panel_w = JCTC_DOUBLE_COL_IN / ncols * 0.92
    fig_w = JCTC_DOUBLE_COL_IN
    fig_h = panel_w * nrows * 1.05 + 0.35

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.10,
        top=0.96,
        wspace=0.06,
        hspace=0.12,
    )

    for k in range(n_slots):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        if k >= n_trans:
            ax.axis("off")
            continue

        lab_i, cv_i = panels[k]
        lab_j, cv_j = panels[k + 1]
        if cv_i.shape[0] != cv_j.shape[0]:
            raise ValueError(
                f"replica mismatch {lab_i} R={cv_i.shape[0]} vs {lab_j} R={cv_j.shape[0]}"
            )
        n_r = cv_i.shape[0]
        if max_replicas is not None and 0 < max_replicas < n_r:
            sel = np.linspace(0, n_r - 1, max_replicas).astype(np.int64)
            a, b = cv_i[sel], cv_j[sel]
        else:
            a, b = cv_i, cv_j

        segs = np.stack([a, b], axis=1)
        ax.add_collection(
            LineCollection(
                segs,
                colors=(1.0, 0.45, 0.45, float(line_alpha)),
                linewidths=0.45,
                zorder=1,
            )
        )
        a_blue = float(point_alpha if blue_alpha is None else blue_alpha)
        a_red = float(point_alpha if red_alpha is None else red_alpha)
        sc_i = ax.scatter(
            a[:, 0],
            a[:, 1],
            s=point_size,
            c="#1f4e9c",
            alpha=a_blue,
            linewidths=0,
            zorder=2,
            rasterized=True,
            label=_legend_lambda_text(lab_i),
        )
        sc_j = ax.scatter(
            b[:, 0],
            b[:, 1],
            s=point_size,
            c="#c62828",
            alpha=a_red,
            linewidths=0,
            zorder=3,
            rasterized=True,
            label=_legend_lambda_text(lab_j),
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_box_aspect(1)
        ax.tick_params(which="both", top=False, right=False)
        # two-line legend (color + λ), upper right
        leg = ax.legend(
            handles=[sc_i, sc_j],
            loc="upper right",
            fontsize=6.5,
            markerscale=2.2,
            handletextpad=0.3,
            borderpad=0.25,
            labelspacing=0.25,
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            framealpha=0.85,
        )
        leg.get_frame().set_linewidth(0.4)
        if c == 0:
            ax.set_ylabel(r"$d_2$ / Å")
        else:
            ax.tick_params(labelleft=False)
        if r == nrows - 1:
            ax.set_xlabel(r"$d_1$ / Å")
        else:
            ax.tick_params(labelbottom=False)

    _savefig(fig, out_path)
    plt.close(fig)
    return Path(out_path)


def plot_transitions(
    stages: list[tuple[int, str, np.ndarray]],
    out_path: Path,
    *,
    max_replicas: int | None = None,
    line_alpha: float = 0.12,
    point_size: float = 6.0,
    ncols: int = 3,
) -> Path:
    n = len(stages)
    n_panels = n - 1
    ncols = max(1, min(ncols, n_panels))
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    all_cv = np.concatenate([cv for _, _, cv in stages], axis=0)
    d1_lo, d1_hi = float(all_cv[:, 0].min()), float(all_cv[:, 0].max())
    d2_lo, d2_hi = float(all_cv[:, 1].min()), float(all_cv[:, 1].max())
    pad1 = 0.05 * max(d1_hi - d1_lo, 1e-6)
    pad2 = 0.05 * max(d2_hi - d2_lo, 1e-6)

    for k in range(n_panels):
        ax = axes[k // ncols][k % ncols]
        idx_i, tag_i, cv_i = stages[k]
        idx_j, tag_j, cv_j = stages[k + 1]
        if cv_i.shape[0] != cv_j.shape[0]:
            raise ValueError(
                f"replica count mismatch: {tag_i} R={cv_i.shape[0]} vs "
                f"{tag_j} R={cv_j.shape[0]}"
            )
        r = cv_i.shape[0]
        if max_replicas is not None and max_replicas > 0 and max_replicas < r:
            sel = np.linspace(0, r - 1, max_replicas).astype(np.int64)
            a, b = cv_i[sel], cv_j[sel]
        else:
            a, b = cv_i, cv_j

        segs = np.stack([a, b], axis=1)
        ax.add_collection(
            LineCollection(
                segs,
                colors=(1.0, 0.45, 0.45, float(line_alpha)),
                linewidths=0.6,
                zorder=1,
            )
        )
        ax.scatter(
            a[:, 0],
            a[:, 1],
            s=point_size,
            c="#1f4e9c",
            alpha=0.75,
            linewidths=0,
            zorder=2,
            label=f"stage {idx_i:04d}",
        )
        ax.scatter(
            b[:, 0],
            b[:, 1],
            s=point_size,
            c="#c62828",
            alpha=0.75,
            linewidths=0,
            zorder=3,
            label=f"stage {idx_j:04d}",
        )
        ax.set_xlim(d1_lo - pad1, d1_hi + pad1)
        ax.set_ylim(d2_lo - pad2, d2_hi + pad2)
        ax.set_xlabel(r"$d_1$ / Å")
        ax.set_ylabel(r"$d_2$ / Å")
        ax.set_title(f"stage {idx_i:04d} → {idx_j:04d}", fontsize=10)
        ax.legend(loc="best", fontsize=7, markerscale=1.5, framealpha=0.9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    for k in range(n_panels, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(
        f"FPT consecutive-stage CV transitions  (n={n} stages, {n_panels} panels)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML required for --config (pip/conda install pyyaml)") from e
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping, got {type(data)}")
    return data


def run_from_config(cfg_path: Path) -> Path:
    cfg = load_yaml_config(cfg_path)
    result_dir = resolve_result_dir(Path(cfg["result_dir"]))
    panels_spec = list(cfg["panels"])
    panels = [resolve_panel_cv(result_dir, spec) for spec in panels_spec]
    for lab, cv in panels:
        print(f"[panel] {lab}  R={cv.shape[0]}", flush=True)

    out = Path(
        cfg.get(
            "output",
            result_dir / "d1_d2" / "stage_snapshots_selected.png",
        )
    )
    nrows = int(cfg.get("nrows", 2))
    ncols = int(cfg.get("ncols", 4))
    mode = str(cfg.get("mode", "transitions")).lower()

    with plt.rc_context(jctc_rcparams()):
        if mode in ("snapshot", "snapshots"):
            return plot_selected_snapshots(
                panels,
                out,
                nrows=nrows,
                ncols=ncols,
                point_size=float(cfg.get("point_size", 4.0)),
                point_alpha=float(cfg.get("point_alpha", 0.55)),
            )
        return plot_selected_transitions(
            panels,
            out,
            nrows=nrows,
            ncols=ncols,
            point_size=float(cfg.get("point_size", 2.2)),
            point_alpha=float(cfg.get("point_alpha", 0.35)),
            blue_alpha=cfg.get("blue_alpha"),
            red_alpha=cfg.get("red_alpha"),
            line_alpha=float(cfg.get("line_alpha", 0.35)),
            max_replicas=cfg.get("max_replicas"),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "result_dir",
        type=Path,
        nargs="?",
        default=None,
        help="results dir or its d1_d2/ subdir (legacy transition mode)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config selecting panels (λ snapshots, 2×4 JCTC layout).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG (default: <result>/d1_d2/stage_transitions.png)",
    )
    p.add_argument(
        "--max-replicas",
        type=int,
        default=None,
        help="optional downsample for denser plots (default: all replicas)",
    )
    p.add_argument("--line-alpha", type=float, default=0.45)
    p.add_argument("--point-size", type=float, default=6.0)
    p.add_argument("--ncols", type=int, default=3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.config is not None:
        path = run_from_config(Path(args.config))
        print(f"[done] {path}", flush=True)
        return

    if args.result_dir is None:
        args.result_dir = Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_mace/results/claisen_fpt_mh-0_g4_85889/d1_d2"
        )
    result_dir = resolve_result_dir(Path(args.result_dir))
    stages = load_stage_cvs(result_dir)
    out = Path(args.output) if args.output else (result_dir / "d1_d2" / "stage_transitions.png")
    print(
        f"[load] {result_dir}  stages={[s[0] for s in stages]}  "
        f"R={stages[0][2].shape[0]}  panels={len(stages)-1}",
        flush=True,
    )
    path = plot_transitions(
        stages,
        out,
        max_replicas=args.max_replicas,
        line_alpha=float(args.line_alpha),
        point_size=float(args.point_size),
        ncols=int(args.ncols),
    )
    print(f"[ok] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
