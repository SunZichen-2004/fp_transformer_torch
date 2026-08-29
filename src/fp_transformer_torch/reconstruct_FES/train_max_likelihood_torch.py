#!/usr/bin/env python3
"""
Maximum-likelihood training of an unbiased free-energy model F_θ(s) from
multi-λ CV samples (PyTorch).

Matches ``train_max_likelihood_jax.py`` for dihedral (torus) data, and also
supports arbitrary CVs (e.g. distances) via an identity embedding.

Likelihood (per sample from dataset k)::

    α_k = λ_k / (1 + λ_k)
    −log p_k(s) = α_k F_θ(s) + log Z_k
    Z_k = ∫_Ω exp(−α_k F_θ(s')) ds'

For low-D identity CVs (e.g. 2D distances), Z is estimated by a dense
Cartesian grid (midpoint quadrature). Monte Carlo remains available for
high-D / torus embeddings. F_θ is learned in reduced (k_B T) units.
"""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LAMBDA_PATTERN = re.compile(r"lambda(\d+p\d+)", re.IGNORECASE)
STAGE_INDEX_PATTERN = re.compile(r"stage(\d+)", re.IGNORECASE)
EV_PER_KT_AT_T = 8.617330337217213e-5  # eV/K  (k_B)


def parse_lambda_from_filename(file_path: str | Path) -> float:
    name = Path(file_path).name
    match = LAMBDA_PATTERN.search(name)
    if match is None:
        raise ValueError(
            f"Could not parse lambda from '{name}'. "
            "Expected substring like 'lambda0p050000'."
        )
    return float(match.group(1).replace("p", ".", 1))


def parse_stage_index_from_filename(file_path: str | Path) -> int | None:
    match = STAGE_INDEX_PATTERN.search(Path(file_path).name)
    if match is None:
        return None
    return int(match.group(1))


def load_cv_datasets(
    data_directory: Path,
    *,
    file_glob: str = "*_cv.npy",
    expected_cv_dim: int | None = None,
    embedding: str = "identity",
    max_stages: int | None = None,
) -> tuple[list[np.ndarray], list[float], list[int], int]:
    """Load multi-λ CV arrays; skip files without a parseable λ.

    If ``max_stages`` is set, keep only the first ``max_stages`` files ordered by
    ``stage####`` index (e.g. 16 → stage0000 … stage0015).
    """
    pattern = str(data_directory / file_glob)
    file_paths = sorted(glob(pattern))
    if not file_paths:
        raise FileNotFoundError(
            f"No files matching '{file_glob}' under {data_directory.resolve()}"
        )

    # Prefer numeric stage order; fall back to path name.
    def _sort_key(path: str) -> tuple[int, str]:
        idx = parse_stage_index_from_filename(path)
        return (idx if idx is not None else 10**9, Path(path).name)

    file_paths = sorted(file_paths, key=_sort_key)
    if max_stages is not None:
        n = int(max_stages)
        if n < 1:
            raise ValueError(f"--max-stages must be >= 1, got {n}")
        staged = [p for p in file_paths if parse_stage_index_from_filename(p) is not None]
        others = [p for p in file_paths if parse_stage_index_from_filename(p) is None]
        if staged:
            file_paths = staged[:n]
            print(
                f"[select] first {len(file_paths)} stage file(s) "
                f"(max_stages={n}); skipped later stages / non-stage files={len(staged)-len(file_paths)+len(others)}",
                flush=True,
            )
        else:
            file_paths = file_paths[:n]
            print(f"[select] first {len(file_paths)} file(s) by name (max_stages={n})", flush=True)

    samples_per_dataset: list[np.ndarray] = []
    lambda_per_dataset: list[float] = []
    sample_counts: list[int] = []
    cv_dim: int | None = None

    for file_path in file_paths:
        try:
            lambda_value = parse_lambda_from_filename(file_path)
        except ValueError:
            print(f"[skip] no lambda in name: {Path(file_path).name}", flush=True)
            continue

        array = np.asarray(np.load(file_path, mmap_mode="r"), dtype=np.float64)
        if array.size == 0:
            raise ValueError(f"File '{file_path}' is empty.")
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array.reshape(array.shape[0], array.shape[1])
        if array.ndim != 2:
            array = array.reshape(-1, array.shape[-1])

        d = int(array.shape[-1])
        if d < 1:
            raise ValueError(f"File '{file_path}' has invalid shape {array.shape}.")
        if cv_dim is None:
            cv_dim = d
            if expected_cv_dim is not None and cv_dim != expected_cv_dim:
                raise ValueError(
                    f"File '{file_path}' has D={cv_dim}, "
                    f"but --cv-dim {expected_cv_dim} was requested."
                )
        elif d != cv_dim:
            raise ValueError(
                f"File '{file_path}' has last dim {d}, expected {cv_dim}."
            )

        samples = array.reshape(-1, cv_dim)
        if embedding == "torus":
            samples = (samples + np.pi) % (2.0 * np.pi) - np.pi

        stage_idx = parse_stage_index_from_filename(file_path)
        tag = f"stage{stage_idx:04d}" if stage_idx is not None else Path(file_path).name
        print(
            f"[load] {tag}  lambda={lambda_value:g}  N={samples.shape[0]}  "
            f"file={Path(file_path).name}",
            flush=True,
        )
        samples_per_dataset.append(samples.astype(np.float64))
        lambda_per_dataset.append(lambda_value)
        sample_counts.append(int(samples.shape[0]))

    if cv_dim is None or not samples_per_dataset:
        raise FileNotFoundError(
            f"No usable λ-tagged files matching '{file_glob}' under {data_directory}"
        )
    return samples_per_dataset, lambda_per_dataset, sample_counts, cv_dim


def embed_features(cv: torch.Tensor, embedding: str) -> torch.Tensor:
    """Map CV [..., D] → features [..., F]."""
    if embedding == "torus":
        sine = torch.sin(cv)
        cosine = torch.cos(cv)
        return torch.stack([sine, cosine], dim=-1).reshape(*cv.shape[:-1], -1)
    if embedding == "identity":
        return cv
    raise ValueError(f"Unknown embedding '{embedding}' (use torus|identity).")


def feature_dim(cv_dim: int, embedding: str) -> int:
    if embedding == "torus":
        return 2 * cv_dim
    if embedding == "identity":
        return cv_dim
    raise ValueError(f"Unknown embedding '{embedding}'")


class FreeEnergyMLP(nn.Module):
    """Scalar free energy F_θ(features) in reduced (kT) units."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        if not hidden_sizes or any(w < 1 for w in hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive ints, got {hidden_sizes}")
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.SiLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).reshape(-1)


def log_hypercube_volume(bounds_lo: torch.Tensor, bounds_hi: torch.Tensor) -> torch.Tensor:
    widths = bounds_hi - bounds_lo
    if torch.any(widths <= 0):
        raise ValueError(f"invalid CV bounds: lo={bounds_lo}, hi={bounds_hi}")
    return torch.sum(torch.log(widths))


def sample_reference_points(
    n: int,
    cv_dim: int,
    *,
    embedding: str,
    bounds_lo: torch.Tensor,
    bounds_hi: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if embedding == "torus":
        # Uniform on [-π, π)^D
        u = torch.rand((n, cv_dim), generator=generator, dtype=bounds_lo.dtype, device=bounds_lo.device)
        return u * (2.0 * torch.pi) - torch.pi
    u = torch.rand((n, cv_dim), generator=generator, dtype=bounds_lo.dtype, device=bounds_lo.device)
    return bounds_lo + u * (bounds_hi - bounds_lo)


def build_reference_grid(
    grid_n: int,
    cv_dim: int,
    *,
    embedding: str,
    bounds_lo: torch.Tensor,
    bounds_hi: torch.Tensor,
) -> torch.Tensor:
    """Dense midpoint Cartesian grid for quadrature of Z (shape ``[n^D, D]``)."""
    n = int(grid_n)
    if n < 2:
        raise ValueError(f"--grid-n must be >= 2, got {n}")
    if cv_dim > 3 and n ** cv_dim > 2_000_000:
        raise ValueError(
            f"grid too large: grid_n={n}, D={cv_dim} → {n ** cv_dim} points; "
            "use --z-method mc or reduce --grid-n"
        )

    axes: list[torch.Tensor] = []
    for d in range(cv_dim):
        lo = bounds_lo[d]
        hi = bounds_hi[d]
        if embedding == "torus":
            # cell centers on [-π, π)
            dx = (2.0 * torch.pi) / float(n)
            axis = -torch.pi + (torch.arange(n, device=lo.device, dtype=lo.dtype) + 0.5) * dx
        else:
            width = hi - lo
            dx = width / float(n)
            axis = lo + (torch.arange(n, device=lo.device, dtype=lo.dtype) + 0.5) * dx
        axes.append(axis)

    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    return grid


def negative_log_likelihood(
    model: FreeEnergyMLP,
    sample_cv: torch.Tensor,
    dataset_index: torch.Tensor,
    alpha_per_dataset: torch.Tensor,
    reference_cv: torch.Tensor,
    *,
    embedding: str,
    log_volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_feat = embed_features(reference_cv, embedding)
    f_ref = model(ref_feat)
    # log Z_k = log V + logsumexp(-α_k F) - log M
    log_m = torch.log(
        torch.tensor(float(reference_cv.shape[0]), dtype=f_ref.dtype, device=f_ref.device)
    )
    # [K, M]
    scaled = -alpha_per_dataset[:, None] * f_ref[None, :]
    log_z = log_volume + torch.logsumexp(scaled, dim=-1) - log_m

    data_feat = embed_features(sample_cv, embedding)
    f_data = model(data_feat)
    alpha = alpha_per_dataset[dataset_index]
    log_z_s = log_z[dataset_index]
    per = alpha * f_data + log_z_s
    loss = per.mean()
    return loss, (alpha * f_data).mean(), log_z_s.mean(), f_ref


def grid_smoothness_penalty(
    f_flat: torch.Tensor,
    grid_n: int,
    *,
    grad_weight: float = 1.0,
    lap_weight: float = 1.0,
) -> torch.Tensor:
    """Finite-difference smoothness on a 2D Cartesian F grid (kT² units).

    ``f_flat`` must match ``build_reference_grid`` with ``indexing='ij'``:
    reshape to ``[n, n]`` with axes (cv0, cv1).
    """
    n = int(grid_n)
    if f_flat.numel() != n * n:
        raise ValueError(
            f"smoothness expects n*n={n*n} values, got {int(f_flat.numel())}"
        )
    f = f_flat.reshape(n, n)
    # first differences along each axis
    gx = f[1:, :] - f[:-1, :]
    gy = f[:, 1:] - f[:, :-1]
    # discrete Laplacian-like second differences
    lxx = f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]
    lyy = f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]
    pen = float(grad_weight) * (gx.pow(2).mean() + gy.pow(2).mean())
    pen = pen + float(lap_weight) * (lxx.pow(2).mean() + lyy.pow(2).mean())
    return pen


def infer_bounds(
    samples_per_dataset: list[np.ndarray],
    *,
    pad: float,
    cli_bounds: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cli_bounds:
        vals = [float(x) for x in cli_bounds.replace(";", ",").split(",") if x.strip()]
        if len(vals) % 2 != 0 or len(vals) < 2:
            raise ValueError(
                "--bounds must be lo1,hi1,lo2,hi2,... got "
                f"{cli_bounds!r}"
            )
        arr = np.asarray(vals, dtype=np.float64).reshape(-1, 2)
        return arr[:, 0].copy(), arr[:, 1].copy()

    stacked = np.concatenate(samples_per_dataset, axis=0)
    lo = stacked.min(axis=0) - float(pad)
    hi = stacked.max(axis=0) + float(pad)
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument(
        "--file-glob",
        type=str,
        default="*_cv.npy",
        help="Glob under data_dir (default: *_cv.npy). Use '*_dihedral.npy' for dihedrals.",
    )
    p.add_argument(
        "--max-stages",
        type=int,
        default=None,
        help="Keep only the first N stage#### files (by stage index). Example: 16 → stage0000–0015.",
    )
    p.add_argument(
        "--embedding",
        type=str,
        default="identity",
        choices=("identity", "torus"),
        help="identity: raw CV (distances, …); torus: sin/cos for angles in radians.",
    )
    p.add_argument("--cv-dim", type=int, default=None)
    p.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="Identity integration domain: lo1,hi1,lo2,hi2,... (default: data min/max + pad).",
    )
    p.add_argument(
        "--bound-pad",
        type=float,
        default=0.2,
        help="Padding added to auto bounds (Å or CV units).",
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--z-method",
        type=str,
        default="grid",
        choices=("grid", "mc"),
        help="Partition-function estimator: dense grid (default) or Monte Carlo.",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=160,
        help="Grid points per CV dimension for --z-method grid (default: 160).",
    )
    p.add_argument(
        "--mc_samples",
        type=int,
        default=8192,
        help="Reference samples per step when --z-method mc.",
    )
    p.add_argument(
        "--smooth-weight",
        type=float,
        default=0.0,
        help="Weight of 2D grid smoothness penalty added to NLL (0 disables).",
    )
    p.add_argument(
        "--smooth-grad-weight",
        type=float,
        default=1.0,
        help="Relative weight of first-difference term inside smoothness penalty.",
    )
    p.add_argument(
        "--smooth-lap-weight",
        type=float,
        default=5.0,
        help="Relative weight of second-difference (Laplacian) term.",
    )
    p.add_argument(
        "--smooth-grid-n",
        type=int,
        default=None,
        help="Grid size for smoothness (default: same as --grid-n when z-method=grid).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", type=str, default="adamw", choices=("adamw", "adam"))
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--ckpt_name", type=str, default="fes_mle.pt")
    p.add_argument("--loss_history_name", type=str, default="loss_history.npy")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temperature", type=float, default=300.0, help="T for unit conversion metadata.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    data_directory = Path(args.data_dir)
    save_directory = Path(args.save_dir)
    save_directory.mkdir(parents=True, exist_ok=True)

    samples_per_dataset, lambda_per_dataset, sample_counts, cv_dim = load_cv_datasets(
        data_directory,
        file_glob=args.file_glob,
        expected_cv_dim=args.cv_dim,
        embedding=args.embedding,
        max_stages=args.max_stages,
    )
    lo_np, hi_np = infer_bounds(
        samples_per_dataset, pad=float(args.bound_pad), cli_bounds=args.bounds
    )
    if lo_np.shape[0] != cv_dim:
        raise ValueError(f"bounds dim {lo_np.shape[0]} != cv_dim {cv_dim}")

    if args.embedding == "torus":
        log_vol = torch.tensor(
            float(cv_dim) * float(np.log(2.0 * np.pi)), dtype=torch.float32, device=device
        )
        bounds_lo = torch.full((cv_dim,), -float(np.pi), dtype=torch.float32, device=device)
        bounds_hi = torch.full((cv_dim,), float(np.pi), dtype=torch.float32, device=device)
    else:
        bounds_lo = torch.as_tensor(lo_np, dtype=torch.float32, device=device)
        bounds_hi = torch.as_tensor(hi_np, dtype=torch.float32, device=device)
        log_vol = log_hypercube_volume(bounds_lo, bounds_hi)

    alpha_list = [lam / (1.0 + lam) for lam in lambda_per_dataset]
    alpha = torch.as_tensor(alpha_list, dtype=torch.float32, device=device)

    cv_blocks = []
    idx_blocks = []
    for k, samples in enumerate(samples_per_dataset):
        cv_blocks.append(torch.as_tensor(samples, dtype=torch.float32, device=device))
        idx_blocks.append(
            torch.full((samples.shape[0],), k, dtype=torch.long, device=device)
        )
    sample_cv = torch.cat(cv_blocks, dim=0)
    dataset_index = torch.cat(idx_blocks, dim=0)

    print(
        f"Loaded {len(samples_per_dataset)} dataset(s), {int(sample_cv.shape[0])} samples, "
        f"D={cv_dim}, embedding={args.embedding}, device={device}",
        flush=True,
    )
    print(f"Z domain lo={lo_np} hi={hi_np}  logV={float(log_vol):.6f}", flush=True)
    for idx, (lam, nk) in enumerate(zip(lambda_per_dataset, sample_counts, strict=True)):
        print(
            f"  dataset {idx}: lambda={lam:g}, alpha={alpha_list[idx]:.6f}, N={nk}",
            flush=True,
        )

    z_method = str(args.z_method)
    ref_grid: torch.Tensor | None = None
    if z_method == "grid":
        ref_grid = build_reference_grid(
            int(args.grid_n),
            cv_dim,
            embedding=args.embedding,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
        )
        print(
            f"Z estimator: grid  n_per_dim={int(args.grid_n)}  "
            f"n_points={int(ref_grid.shape[0])}",
            flush=True,
        )
    else:
        print(f"Z estimator: mc  n_samples={int(args.mc_samples)} / step", flush=True)

    in_dim = feature_dim(cv_dim, args.embedding)
    hidden = tuple(int(w) for w in args.hidden_sizes)
    model = FreeEnergyMLP(in_dim, hidden).to(device)
    print(f"MLP input_dim={in_dim} hidden={hidden}", flush=True)

    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
        print(f"AdamW lr={args.lr:g} wd={args.weight_decay:g}", flush=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        print(f"Adam lr={args.lr:g}", flush=True)

    smooth_weight = float(args.smooth_weight)
    smooth_grid: torch.Tensor | None = None
    smooth_n = 0
    if smooth_weight > 0.0:
        if cv_dim != 2:
            raise ValueError("--smooth-weight > 0 currently requires 2D CV")
        smooth_n = int(args.smooth_grid_n) if args.smooth_grid_n is not None else (
            int(args.grid_n) if z_method == "grid" else 128
        )
        if z_method == "grid" and smooth_n == int(args.grid_n):
            smooth_grid = None  # reuse Z reference grid / f_ref
        else:
            smooth_grid = build_reference_grid(
                smooth_n,
                cv_dim,
                embedding=args.embedding,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
            )
        print(
            f"Smoothness: weight={smooth_weight:g}  grad={args.smooth_grad_weight:g}  "
            f"lap={args.smooth_lap_weight:g}  grid_n={smooth_n}",
            flush=True,
        )

    loss_hist: list[float] = []
    data_hist: list[float] = []
    logz_hist: list[float] = []
    smooth_hist: list[float] = []

    model.train()
    for step in range(int(args.steps)):
        if z_method == "grid":
            assert ref_grid is not None
            ref = ref_grid
        else:
            ref = sample_reference_points(
                int(args.mc_samples),
                cv_dim,
                embedding=args.embedding,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
                generator=generator,
            )
        opt.zero_grad(set_to_none=True)
        nll, mean_af, mean_lz, f_ref = negative_log_likelihood(
            model,
            sample_cv,
            dataset_index,
            alpha,
            ref,
            embedding=args.embedding,
            log_volume=log_vol,
        )
        smooth_pen = torch.zeros((), device=device, dtype=nll.dtype)
        if smooth_weight > 0.0:
            if smooth_grid is None:
                f_smooth = f_ref
                n_s = int(args.grid_n)
            else:
                f_smooth = model(embed_features(smooth_grid, args.embedding))
                n_s = smooth_n
            smooth_pen = grid_smoothness_penalty(
                f_smooth,
                n_s,
                grad_weight=float(args.smooth_grad_weight),
                lap_weight=float(args.smooth_lap_weight),
            )
        loss = nll + smooth_weight * smooth_pen
        loss.backward()
        opt.step()

        loss_hist.append(float(loss.detach()))
        data_hist.append(float(mean_af.detach()))
        logz_hist.append(float(mean_lz.detach()))
        smooth_hist.append(float(smooth_pen.detach()))
        if step % int(args.log_every) == 0 or step == int(args.steps) - 1:
            print(
                f"step {step:5d} | loss {loss_hist[-1]:.6f} | "
                f"nll {float(nll.detach()):.6f} | "
                f"data_term {data_hist[-1]:.6f} | mean_logZ {logz_hist[-1]:.6f} | "
                f"smooth {smooth_hist[-1]:.6f}",
                flush=True,
            )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "hidden_sizes": hidden,
        "cv_dim": int(cv_dim),
        "input_feature_dimension": int(in_dim),
        "embedding": str(args.embedding),
        "bounds_lo": lo_np.astype(np.float64),
        "bounds_hi": hi_np.astype(np.float64),
        "log_volume": float(log_vol.detach().cpu()),
        "lambda_per_dataset": list(map(float, lambda_per_dataset)),
        "alpha_per_dataset": list(map(float, alpha_list)),
        "seed": int(args.seed),
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay) if args.optimizer == "adamw" else 0.0,
        "steps": int(args.steps),
        "z_method": z_method,
        "grid_n": int(args.grid_n) if z_method == "grid" else None,
        "mc_samples": int(args.mc_samples) if z_method == "mc" else None,
        "smooth_weight": float(args.smooth_weight),
        "smooth_grad_weight": float(args.smooth_grad_weight),
        "smooth_lap_weight": float(args.smooth_lap_weight),
        "smooth_grid_n": int(smooth_n) if smooth_weight > 0.0 else None,
        "temperature": float(args.temperature),
        "energy_unit": "kT",
        "file_glob": str(args.file_glob),
        "max_stages": int(args.max_stages) if args.max_stages is not None else None,
        "data_dir": str(data_directory.resolve()),
    }
    ckpt_path = save_directory / args.ckpt_name
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path.resolve()}", flush=True)

    hist = np.stack(
        [
            np.asarray(loss_hist),
            np.asarray(data_hist),
            np.asarray(logz_hist),
            np.asarray(smooth_hist),
        ],
        axis=0,
    )
    hist_path = save_directory / args.loss_history_name
    np.save(hist_path, hist)
    print(
        f"Saved loss history (rows: total_loss, data_term, mean_logZ, smooth) "
        f"to {hist_path.resolve()}",
        flush=True,
    )


def load_free_energy_model(
    ckpt_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FreeEnergyMLP, dict]:
    """Load trained MLP + metadata for evaluation / plotting."""
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FreeEnergyMLP(
        int(ckpt["input_feature_dimension"]),
        tuple(int(w) for w in ckpt["hidden_sizes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_free_energy(
    model: FreeEnergyMLP,
    ckpt: dict,
    cv: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Evaluate reduced free energy F/kT on CV points shaped [N, D] or grid."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    embedding = str(ckpt["embedding"])
    x = torch.as_tensor(np.asarray(cv, dtype=np.float32), device=device)
    flat = x.reshape(-1, int(ckpt["cv_dim"]))
    feat = embed_features(flat, embedding)
    f = model(feat).reshape(x.shape[:-1])
    return f.detach().cpu().numpy()


if __name__ == "__main__":
    main()
