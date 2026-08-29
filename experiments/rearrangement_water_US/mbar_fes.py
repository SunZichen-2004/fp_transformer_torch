#!/usr/bin/env python3
"""Reconstruct 1D F(ξ) from umbrella sampling with pymbar MBAR.

Reads a US result directory written by ``sample.py``::

    trajectory_cv.npy          (n_frames, n_windows, 1)   ξ = d1 − d2 / Å
    trajectory_cv_times_ps.npy (n_frames,)
    umbrella_centers.npy       (n_windows, 1)
    umbrella_kappa.npy         (n_windows, 1)             eV / Å²
    run_info.txt               temperature etc.

Bias per window: ``U = ½ κ (ξ − ξ₀)²`` (same form as ``UmbrellaBias``).
Unbiased PMF is obtained by adding a fictitious unbiased state (N_unbiased=0)
and evaluating bin-indicator expectations in that state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.06054783061903
KB_EV_PER_K = 8.617330337217213e-5


def _parse_run_info(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not path.is_file():
        return info
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


def _subsample_window(xi: np.ndarray, *, subsample: bool) -> np.ndarray:
    """Return decorrelated samples for one window (1D array)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    xi = xi[np.isfinite(xi)]
    if xi.size < 3 or not subsample:
        return xi
    try:
        from pymbar import timeseries

        idx = timeseries.subsample_correlated_data(xi)
        return xi[np.asarray(idx, dtype=np.int64)]
    except Exception:
        return xi


def build_u_kn(
    samples: list[np.ndarray],
    centers: np.ndarray,
    kappa: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reduced potentials including an unbiased (u=0) state.

    Returns
    -------
    u_kn : (K+1, N)
    N_k : (K+1,)
    xi_n : (N,)
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    K = int(centers.shape[0])
    if kappa.shape[0] == 1:
        kappa = np.full(K, float(kappa[0]))
    if kappa.shape[0] != K:
        raise ValueError(f"kappa length {kappa.shape[0]} != n_windows {K}")

    xi_parts = []
    N_k = np.zeros(K + 1, dtype=np.int64)
    for k, xi in enumerate(samples):
        xi = np.asarray(xi, dtype=np.float64).reshape(-1)
        xi_parts.append(xi)
        N_k[k] = xi.size
    xi_n = np.concatenate(xi_parts, axis=0)
    N = int(xi_n.size)
    if N == 0:
        raise RuntimeError("no CV samples after discard/subsample")

    u_kn = np.zeros((K + 1, N), dtype=np.float64)
    for k in range(K):
        u_kn[k] = beta * 0.5 * kappa[k] * (xi_n - centers[k]) ** 2
    # state K: unbiased (no umbrella); N_k[K] stays 0
    return u_kn, N_k, xi_n


def mbar_pmf_1d(
    u_kn: np.ndarray,
    N_k: np.ndarray,
    xi_n: np.ndarray,
    *,
    centers: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    n_bins: int,
    xi_min: float,
    xi_max: float,
    kT_kcal: float,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """MBAR free energies of windows + WHAM histogram for F(ξ).

    Window free energies ``f_k`` from pymbar; unbiased density on a grid::

        ρ(s) = Σ_n 1[s_n∈bin(s)] / Σ_k N_k exp(f_k − β ½ κ (s − ξ_k)²)

    Evaluating the bias at the **bin center** (WHAM) avoids the underflow that
    occurs if one reweights every sample into a single global unbiased ensemble
    and then histograms (which collapses onto the reactant well for high barriers).
    """
    from pymbar import MBAR

    if N_k[-1] == 0 and u_kn.shape[0] == N_k.shape[0]:
        u_bias = np.asarray(u_kn[:-1], dtype=np.float64)
        N_bias = np.asarray(N_k[:-1], dtype=np.int64)
    else:
        u_bias = np.asarray(u_kn, dtype=np.float64)
        N_bias = np.asarray(N_k, dtype=np.int64)
    K = int(N_bias.shape[0])
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    if kappa.shape[0] == 1:
        kappa = np.full(K, float(kappa[0]))
    if centers.shape[0] != K or kappa.shape[0] != K:
        raise ValueError("centers/kappa length must match n_windows")

    mbar = MBAR(u_bias, N_bias, verbose=verbose)
    fe = mbar.compute_free_energy_differences()
    Delta_f = np.asarray(fe["Delta_f"], dtype=np.float64)
    dDelta_f = np.asarray(fe["dDelta_f"], dtype=np.float64)
    f_k = np.asarray(mbar.f_k, dtype=np.float64)

    edges = np.linspace(float(xi_min), float(xi_max), int(n_bins) + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    # raw counts per bin (all windows pooled)
    counts, _ = np.histogram(xi_n, bins=edges)
    p = np.zeros(int(n_bins), dtype=np.float64)
    for i, s in enumerate(bin_centers):
        if counts[i] <= 0:
            p[i] = 0.0
            continue
        # Σ_k N_k exp(f_k − β ½ κ (s − ξ_k)²)
        u_s = beta * 0.5 * kappa * (s - centers) ** 2
        log_den = np.log(np.maximum(N_bias.astype(np.float64), 1.0)) + f_k - u_s
        # logsumexp
        m = float(np.max(log_den))
        den = np.exp(m) * np.exp(log_den - m).sum()
        p[i] = float(counts[i]) / den

    # normalize on occupied bins; F = −kT ln ρ
    ok = p > 0
    if not np.any(ok):
        raise RuntimeError("MBAR/WHAM PMF: no occupied bins")
    p_norm = p.copy()
    p_norm[ok] = p[ok] / p[ok].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -kT_kcal * np.log(p_norm)
    F0 = float(np.nanmin(F[ok]))
    F = F - F0
    F[~ok] = np.nan
    # Poisson-ish uncertainty on counts → rough dF
    dF = np.full_like(F, np.nan)
    dF[ok] = kT_kcal / np.sqrt(counts[ok].astype(np.float64))

    try:
        overlap = mbar.compute_overlap()
        if isinstance(overlap, dict):
            O = np.asarray(
                overlap.get("matrix", overlap.get("overlap", [])), dtype=np.float64
            )
        else:
            O = np.asarray(overlap, dtype=np.float64)
    except Exception:
        O = np.full((K, K), np.nan)

    # stiff-restraint check: f_k * kT vs ξ_k
    F_windows = kT_kcal * (f_k - float(np.min(f_k)))

    return {
        "xi": bin_centers,
        "bin_edges": edges,
        "p": p_norm,
        "dp": np.sqrt(np.maximum(counts, 0.0)),
        "counts": counts.astype(np.float64),
        "F_kcal": F,
        "dF_kcal": dF,
        "Delta_f_kT": Delta_f,
        "dDelta_f_kT": dDelta_f,
        "overlap": O,
        "N_k": N_bias.astype(np.float64),
        "f_k_kT": f_k,
        "F_windows_kcal": F_windows,
        "centers": centers,
    }


def plot_results(
    out: Path,
    *,
    xi: np.ndarray,
    F: np.ndarray,
    dF: np.ndarray,
    samples: list[np.ndarray],
    centers: np.ndarray,
    overlap: np.ndarray,
    N_k: np.ndarray,
    F_windows: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    K = centers.size

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ok = np.isfinite(F)
    ax.plot(xi[ok], F[ok], c="C0", lw=2.0, label="MBAR/WHAM")
    if np.any(np.isfinite(dF)):
        ax.fill_between(
            xi[ok],
            (F - dF)[ok],
            (F + dF)[ok],
            color="C0",
            alpha=0.25,
            linewidth=0,
            label=r"$\pm 1\sigma$ (rough)",
        )
    if F_windows is not None:
        ax.plot(
            centers,
            np.asarray(F_windows, dtype=np.float64),
            "k.",
            ms=4,
            alpha=0.7,
            label=r"$f_k$ (windows)",
        )
    ax.set_xlabel(r"$\xi = d_1 - d_2$ / Å")
    ax.set_ylabel(r"$F(\xi)$ / kcal mol$^{-1}$")
    ax.set_title("1D US → MBAR FES")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fes_mbar.png", dpi=160)
    plt.close(fig)

    # per-window histograms
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    cmap = plt.cm.turbo
    for k, xi_k in enumerate(samples):
        if xi_k.size == 0:
            continue
        ax.hist(
            xi_k,
            bins=40,
            density=True,
            histtype="step",
            color=cmap(k / max(K - 1, 1)),
            alpha=0.7,
            lw=0.9,
        )
        ax.axvline(centers[k], color=cmap(k / max(K - 1, 1)), lw=0.4, alpha=0.5)
    ax.set_xlabel(r"$\xi$ / Å")
    ax.set_ylabel("density")
    ax.set_title("umbrella window histograms (post-discard)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "window_histograms.png", dpi=140)
    plt.close(fig)

    # overlap among biased states only
    if overlap.ndim == 2 and overlap.shape[0] >= K:
        Ob = overlap[:K, :K]
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(
            Ob,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=max(0.2, float(np.nanmax(Ob))),
        )
        ax.set_xlabel("window j")
        ax.set_ylabel("window i")
        ax.set_title("MBAR overlap matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / "overlap_matrix.png", dpi=140)
        plt.close(fig)

    # samples per window
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(np.arange(K), N_k[:K], color="C0", width=0.85)
    ax.set_xlabel("window index")
    ax.set_ylabel("N samples (subsampled)")
    ax.set_title("MBAR sample counts")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "N_k.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="US result directory (contains trajectory_cv.npy)",
    )
    p.add_argument(
        "--result-dirs",
        type=str,
        default=None,
        help="Comma-separated US result directories to merge before MBAR",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir (default: <result-dir>/mbar)",
    )
    p.add_argument("--discard-ps", type=float, default=20.0, help="drop frames with t < this")
    p.add_argument("--n-bins", type=int, default=80)
    p.add_argument("--xi-min", type=float, default=None)
    p.add_argument("--xi-max", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-subsample", action="store_true", help="keep all frames (correlated)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dir_list: list[Path] = []
    if args.result_dirs:
        dir_list.extend(Path(x.strip()) for x in args.result_dirs.split(",") if x.strip())
    if args.result_dir is not None:
        dir_list.append(Path(args.result_dir))
    if not dir_list:
        raise SystemExit("need --result-dir or --result-dirs")
    for d in dir_list:
        if not (d / "trajectory_cv.npy").is_file():
            raise FileNotFoundError(f"missing trajectory_cv.npy in {d}")

    result = dir_list[0]
    out = Path(args.out_dir) if args.out_dir else result / "mbar"
    out.mkdir(parents=True, exist_ok=True)

    info = _parse_run_info(result / "run_info.txt")
    T = float(args.temperature if args.temperature is not None else info.get("temperature_K", 300.0))
    beta = 1.0 / (KB_EV_PER_K * T)
    kT_kcal = KB_EV_PER_K * T * EV_TO_KCAL

    samples: list[np.ndarray] = []
    centers_parts: list[np.ndarray] = []
    kappa_parts: list[np.ndarray] = []
    times_use = None
    subsample = not args.no_subsample
    n_frames_keep = 0
    n_frames_tot = 0
    for d in dir_list:
        cv = np.load(d / "trajectory_cv.npy")
        times = np.load(d / "trajectory_cv_times_ps.npy")
        centers_d = np.load(d / "umbrella_centers.npy")
        kappa_d = np.load(d / "umbrella_kappa.npy")
        if cv.ndim == 3:
            xi_traj = cv[..., 0]
        elif cv.ndim == 2:
            xi_traj = cv
        else:
            raise ValueError(f"unexpected trajectory_cv shape {cv.shape} in {d}")
        n_frames, n_win_d = xi_traj.shape
        if int(centers_d.reshape(-1).shape[0]) != n_win_d:
            raise ValueError(f"centers / trajectory window mismatch in {d}")
        mask = times >= float(args.discard_ps)
        if not np.any(mask):
            raise RuntimeError(f"no frames with t >= discard_ps={args.discard_ps} in {d}")
        xi_traj = xi_traj[mask]
        times_use = times[mask]
        n_frames_keep += int(xi_traj.shape[0])
        n_frames_tot += int(n_frames)
        for k in range(n_win_d):
            samples.append(_subsample_window(xi_traj[:, k], subsample=subsample))
        centers_parts.append(np.asarray(centers_d, dtype=np.float64).reshape(-1))
        kap = np.asarray(kappa_d, dtype=np.float64).reshape(-1)
        if kap.size == 1:
            kap = np.full(n_win_d, float(kap[0]))
        kappa_parts.append(kap)
        print(
            f"[mbar] loaded {d.name}  windows={n_win_d}  "
            f"frames_kept={xi_traj.shape[0]}/{n_frames}",
            flush=True,
        )

    centers = np.concatenate(centers_parts, axis=0)
    kappa = np.concatenate(kappa_parts, axis=0)
    order = np.argsort(centers)
    centers = centers[order]
    kappa = kappa[order]
    samples = [samples[i] for i in order]
    n_win = int(centers.size)

    xi_min = float(args.xi_min if args.xi_min is not None else float(centers[0]))
    xi_max = float(args.xi_max if args.xi_max is not None else float(centers[-1]))
    # slight pad so edge bins get counts
    pad = 0.5 * (xi_max - xi_min) / max(args.n_bins, 1)
    xi_lo, xi_hi = xi_min - pad, xi_max + pad

    u_kn, N_k, xi_n = build_u_kn(samples, centers, kappa, beta)
    print(
        f"[mbar] T={T:g} K  windows={n_win}  frames_kept={n_frames_keep}/{n_frames_tot}  "
        f"discard<{args.discard_ps:g} ps  N_tot={int(N_k[:-1].sum())}  "
        f"subsample={subsample}",
        flush=True,
    )
    print(f"[mbar] ξ grid [{xi_lo:.3f}, {xi_hi:.3f}]  n_bins={args.n_bins}", flush=True)

    res = mbar_pmf_1d(
        u_kn,
        N_k,
        xi_n,
        centers=centers,
        kappa=kappa,
        beta=beta,
        n_bins=int(args.n_bins),
        xi_min=xi_lo,
        xi_max=xi_hi,
        kT_kcal=kT_kcal,
        verbose=bool(args.verbose),
    )

    np.savez(
        out / "fes_mbar.npz",
        xi=res["xi"],
        bin_edges=res["bin_edges"],
        F_kcal=res["F_kcal"],
        dF_kcal=res["dF_kcal"],
        p=res["p"],
        dp=res["dp"],
        counts=res["counts"],
        Delta_f_kT=res["Delta_f_kT"],
        dDelta_f_kT=res["dDelta_f_kT"],
        overlap=res["overlap"],
        N_k=res["N_k"],
        f_k_kT=res["f_k_kT"],
        F_windows_kcal=res["F_windows_kcal"],
        centers=np.asarray(centers, dtype=np.float64),
        kappa=np.asarray(kappa, dtype=np.float64),
        temperature_K=T,
        discard_ps=float(args.discard_ps),
        times_ps_used=times_use.astype(np.float64),
    )
    # also a plain text table
    with open(out / "fes_mbar.dat", "w", encoding="utf-8") as fh:
        fh.write("# xi/A    F/kcal/mol    dF/kcal/mol\n")
        for x, f, df in zip(res["xi"], res["F_kcal"], res["dF_kcal"]):
            fh.write(f"{x:12.6f}  {f:12.6f}  {df:12.6f}\n")

    meta = {
        "result_dir": str(result),
        "result_dirs": [str(d) for d in dir_list],
        "temperature_K": T,
        "discard_ps": float(args.discard_ps),
        "n_bins": int(args.n_bins),
        "xi_min": xi_lo,
        "xi_max": xi_hi,
        "n_windows": n_win,
        "N_tot": int(N_k[:-1].sum()),
        "N_k": N_k.astype(int).tolist(),
        "subsample": subsample,
        "kT_kcal": kT_kcal,
        "F_min_xi": float(res["xi"][np.nanargmin(res["F_kcal"])]),
        "F_span_kcal": float(np.nanmax(res["F_kcal"]) - np.nanmin(res["F_kcal"])),
    }
    (out / "mbar_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    plot_results(
        out,
        xi=res["xi"],
        F=res["F_kcal"],
        dF=res["dF_kcal"],
        samples=samples,
        centers=centers,
        overlap=res["overlap"],
        N_k=res["N_k"],
        F_windows=res["F_windows_kcal"],
    )
    print(
        f"[mbar] F span={meta['F_span_kcal']:.2f} kcal/mol  "
        f"min near ξ={meta['F_min_xi']:.3f} Å",
        flush=True,
    )
    print(f"[mbar] wrote {out}/fes_mbar.png  fes_mbar.npz  fes_mbar.dat", flush=True)


if __name__ == "__main__":
    main()
