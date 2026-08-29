#!/usr/bin/env python3
"""WT-MetaD replica scan: ΔG vs time / wall-clock, mean ± std over seeds."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
META = ROOT / "experiments" / "dialanine_water_meta"
RESULTS = META / "results"
OUT = META / "convergence_replica"
DT_PS = 0.001

_spec = importlib.util.spec_from_file_location(
    "plot_deltaG_vs_time_recent",
    META / "plot_deltaG_vs_time_recent.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
from plot_fes_compare import nature_rcparams  # noqa: E402

COLOR_US = "#2CA02C"
YLIM = (1.0, 8.0)
COLORS = {
    1: "#252525",
    16: "#386cb0",
    64: "#1b9e77",
    128: "#d95f02",
}
RATE_FALLBACK = {
    89387: 44.29,
    89392: 44.02,
    89388: 30.98,
    90726: 27.83,
    90728: 88.40,
    90729: 74.45,
    90730: 72.95,
}
JOBS = [
    {"job": 90728, "R": 1, "seed": 2, "dir": RESULTS / "dialanine_water_metad_r1_s2_90728"},
    {"job": 90729, "R": 16, "seed": 0, "dir": RESULTS / "dialanine_water_metad_r16_s0_90729"},
    {"job": 90730, "R": 16, "seed": 2, "dir": RESULTS / "dialanine_water_metad_r16_s2_90730"},
    {"job": 89387, "R": 64, "seed": 0, "dir": RESULTS / "dialanine_water_metad_r64_s0_89387"},
    {"job": 89392, "R": 64, "seed": 2, "dir": RESULTS / "dialanine_water_metad_r64_s2_89392"},
    {"job": 89388, "R": 128, "seed": 0, "dir": RESULTS / "dialanine_water_metad_r128_s0_89388"},
    {"job": 90726, "R": 128, "seed": 2, "dir": RESULTS / "dialanine_water_metad_r128_s2_90726"},
]
STEP_RE = re.compile(
    r"wt-metad Step .*?,\s*n_hills=\d+,\s*([0-9.]+)\s*it/s",
    re.S,
)


def log_candidates(job: int) -> list[Path]:
    jid = int(job)
    return [
        ROOT / f"job.{jid}.out",
        ROOT / "scripts" / f"job.{jid}.out",
        RESULTS / f"job.{jid}.out",
        META / f"job.{jid}.out",
        OUT / f"job.{jid}.out",
    ]


def parse_rate(job: int) -> float:
    for path in log_candidates(job):
        if not path.is_file():
            continue
        try:
            text = path.read_bytes()[-131072:].decode("utf-8", errors="ignore")
        except OSError:
            continue
        hits = STEP_RE.findall(text)
        if hits:
            rate = float(hits[-1])
            print(f"  [{job}] rate={rate:.2f} it/s  from {path.name}", flush=True)
            return rate
    rate = float(RATE_FALLBACK[int(job)])
    print(f"  [{job}] rate={rate:.2f} it/s  (fallback)", flush=True)
    return rate


def wall_hours(t_ps: np.ndarray, rate_it_s: float) -> np.ndarray:
    t = np.asarray(t_ps, dtype=np.float64)
    return (t / DT_PS) / float(rate_it_s) / 3600.0


def mean_std(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hi = min(float(x[-1]) for x, _ in curves)
    x = np.asarray(curves[0][0], dtype=np.float64)
    x = x[x <= x_hi + 1e-12]
    stack = np.stack([np.interp(x, xi, yi) for xi, yi in curves], axis=0)
    mean = np.mean(stack, axis=0)
    if stack.shape[0] > 1:
        std = np.std(stack, axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return x, mean, std


def plot_overlay(
    out: Path,
    *,
    grouped: dict[int, dict],
    xkey: str,
    xlabel: str,
    stem: str,
    dG_ref: float,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.axhspan(dG_ref - 1.0, dG_ref + 1.0, color=COLOR_US, alpha=0.15, lw=0, zorder=0)
    ax.axhline(
        dG_ref,
        color=COLOR_US,
        ls=":",
        lw=1.2,
        zorder=1,
        label=rf"US $\Delta G$={dG_ref:.2f}",
    )
    for R in sorted(grouped):
        s = grouped[R]
        x = s[xkey]
        y = s["dG"]
        e = s["dG_std"]
        c = COLORS.get(int(R), "C0")
        ax.plot(x, y, color=c, lw=1.6, label=rf"$R$={int(R)}", zorder=3)
        if np.any(e > 0):
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.20, lw=0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        r"$\Delta G=k_BT\ln(P_{\phi<0}/P_{\phi>0})$ / kcal mol$^{-1}$"
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(*YLIM)
    ax.legend(loc="best", frameon=False)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dG_us = _mod.load_us_dg(_mod.US_MBAR)
    print(f"US-MBAR ΔG = {dG_us:.3f} kcal/mol", flush=True)

    by_r: dict[int, list[dict]] = defaultdict(list)
    save: dict = {"dG_us": np.float64(dG_us)}
    for job in JOBS:
        hills = job["dir"] / "live" / "hills.npz"
        if not hills.is_file():
            raise FileNotFoundError(hills)
        rate = parse_rate(int(job["job"]))
        print(f"[{job['job']}] R={job['R']} seed={job['seed']}  {hills}", flush=True)
        t, dG = _mod.incremental_dg(hills, n_replicas=int(job["R"]))
        tw = wall_hours(t, rate)
        rec = {
            "job": int(job["job"]),
            "seed": int(job["seed"]),
            "rate": float(rate),
            "t": t,
            "tw": tw,
            "dG": dG,
        }
        by_r[int(job["R"])].append(rec)
        tag = f"j{job['job']}"
        save[f"t_{tag}"] = t
        save[f"tw_{tag}"] = tw
        save[f"dG_{tag}"] = dG
        save[f"R_{tag}"] = np.int64(job["R"])
        save[f"seed_{tag}"] = np.int64(job["seed"])
        save[f"rate_{tag}"] = np.float64(rate)
        print(
            f"    t_end={t[-1]:.1f} ps  tw_end={tw[-1]:.2f} h  ΔG_end={dG[-1]:.3f}",
            flush=True,
        )

    grouped: dict[int, dict] = {}
    for R, recs in sorted(by_r.items()):
        xt, ym, ys = mean_std([(r["t"], r["dG"]) for r in recs])
        xw, wm, ws = mean_std([(r["tw"], r["dG"]) for r in recs])
        grouped[R] = {
            "t": xt,
            "tw": xw,
            "dG": ym,
            "dG_std": ys,
            "dG_w": wm,
            "dG_w_std": ws,
            "n_seeds": len(recs),
        }
        save[f"t_R{R}"] = xt
        save[f"dG_R{R}"] = ym
        save[f"dG_std_R{R}"] = ys
        save[f"tw_R{R}"] = xw
        save[f"dG_w_R{R}"] = wm
        save[f"dG_w_std_R{R}"] = ws
        print(
            f"[R={R}] n_seeds={len(recs)}  t≤{xt[-1]:.1f} ps  "
            f"tw≤{xw[-1]:.2f} h  ΔG={ym[-1]:.3f}±{ys[-1]:.3f}",
            flush=True,
        )

    plot_overlay(
        OUT,
        grouped={R: {"t": g["t"], "dG": g["dG"], "dG_std": g["dG_std"]} for R, g in grouped.items()},
        xkey="t",
        xlabel=r"$t$ / ps",
        stem="deltaG_vs_time_replica",
        dG_ref=dG_us,
    )
    plot_overlay(
        OUT,
        grouped={
            R: {"tw": g["tw"], "dG": g["dG_w"], "dG_std": g["dG_w_std"]}
            for R, g in grouped.items()
        },
        xkey="tw",
        xlabel=r"wall-clock time / h",
        stem="deltaG_vs_wallclock_replica",
        dG_ref=dG_us,
    )
    np.savez_compressed(OUT / "deltaG_vs_time_replica.npz", **save)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
