#!/usr/bin/env python3
"""Dump dialanine GAFF + BAOAB-FP (50 steps) reference arrays using JAX fp-transformer.

Run with the JAX uv venv (NOT conda MACE), e.g.::

  cd /share/home/sunzc/work/Fokker_Plank/fp_0429_final/fp-transformer
  export JAX_ENABLE_X64=1
  uv run python /path/to/dump_jax_dialanine_parity.py

Writes under ``fp_transformer_torch/experiments/dialanine/parity_ref/``:
  coords0.npy, momenta0.npy, ou_noise.npy, bandwidth0.npy,
  coordinates.npy, momenta.npy, energies.npy, meta.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force CPU before any JAX import (login nodes may lack system CUDA libs).
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

JAX_ROOT = Path("/share/home/sunzc/work/Fokker_Plank/fp_0429_final/fp-transformer")
TORCH_ROOT = Path("/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch")
DEFAULT_SPONGE = TORCH_ROOT / "experiments" / "dialanine" / "dialanine"
DEFAULT_INIT = TORCH_ROOT / "experiments" / "dialanine" / "init_coords_2048.npy"
DEFAULT_CV = TORCH_ROOT / "experiments" / "dialanine" / "backbone_cv.txt"
DEFAULT_OUT = TORCH_ROOT / "experiments" / "dialanine" / "parity_ref"

# OpenMM-matched constants (same as sample_xponge / water_pure_jaxmd_langevin)
FEMTOSECOND_TO_AKMA_TIME = 0.0204548282809
KB_KCAL_PER_MOL_K = 1.987204259e-3


def timestep_picosecond_to_akma(dt_ps: float) -> float:
    return float(dt_ps) * 1000.0 * FEMTOSECOND_TO_AKMA_TIME


def friction_per_picosecond_to_akma(gamma_ps: float, dt_ps: float, dt_akma: float) -> float:
    return float(gamma_ps) * float(dt_ps) / float(dt_akma)


def read_cv_indices_1based(path: Path) -> np.ndarray:
    """Match ``sample_xponge.read_cv_indices`` (file is 1-based)."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        chunk = []
        for part in parts:
            try:
                chunk.append(int(part) - 1)
            except ValueError:
                continue
            if len(chunk) == 4:
                rows.append(chunk)
                chunk = []
    if not rows:
        raise ValueError(f"no torsion indices in {path}")
    return np.asarray(rows, dtype=np.int32)


def load_masses(prefix: Path, n_atoms: int) -> np.ndarray:
    mass = np.loadtxt(str(prefix) + "_mass.txt", dtype=np.float64).reshape(-1)
    if mass.size == n_atoms + 1 and int(mass[0]) == n_atoms:
        mass = mass[1:]
    return mass[:n_atoms].astype(np.float64)


def init_bandwidth(cv: np.ndarray) -> float:
    """Median torus pairwise distance (sample_xponge.init_bandwidth)."""
    # Import after JAX path is ready
    import jax.numpy as jnp
    from src.fp_transformer.utils import torus_pairwise_distance

    dist2 = torus_pairwise_distance(jnp.asarray(cv, dtype=jnp.float64))
    tri = np.asarray(dist2)
    iu = np.triu_indices(tri.shape[0], k=1)
    vals = tri[iu]
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size == 0:
        return 0.5
    return float(np.median(pos))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=Path, default=DEFAULT_SPONGE)
    p.add_argument("--init-npy", type=Path, default=DEFAULT_INIT)
    p.add_argument("--cv-file", type=Path, default=DEFAULT_CV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baoab-dt", type=float, default=1e-3)
    p.add_argument("--baoab-langevin-gamma", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--rew-lambda-init", type=float, default=0.05)
    p.add_argument("--cv-grad-clip-norm", type=float, default=None)
    p.add_argument("--bw-update-pace", type=int, default=20)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=1.0)
    p.add_argument("--no-update-bandwidth", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    sys.path.insert(0, str(JAX_ROOT))
    # Reuse torch-repo helper that talks to JAX GAFF / BAOAB core
    sys.path.insert(0, str(TORCH_ROOT / "tests"))

    from jax_dialanine_parity import jax_baoab_fp_trajectory, jax_energy_and_force, load_jax_gaff_energy

    sponge = Path(args.sponge_prefix)
    coords_all = np.load(args.init_npy).astype(np.float64)
    if coords_all.ndim != 3 or coords_all.shape[-1] != 3:
        raise ValueError(f"bad init npy shape {coords_all.shape}")
    batch = min(int(args.replicas), int(coords_all.shape[0]))
    coords0 = coords_all[:batch].copy()
    cv_idx = read_cv_indices_1based(Path(args.cv_file))
    masses = load_masses(sponge, coords0.shape[1])

    kB = KB_KCAL_PER_MOL_K
    beta0 = 1.0 / (kB * float(args.temperature))
    dt_akma = timestep_picosecond_to_akma(args.baoab_dt)
    gamma_akma = friction_per_picosecond_to_akma(
        args.baoab_langevin_gamma, args.baoab_dt, dt_akma
    )
    lam = float(args.rew_lambda_init)

    rng = np.random.RandomState(int(args.seed))
    scale_p = np.sqrt(masses.reshape(1, -1, 1) / beta0)
    momenta0 = rng.randn(*coords0.shape).astype(np.float64) * scale_p
    ou_noise = rng.randn(int(args.steps), *coords0.shape).astype(np.float64)

    # Bandwidth from initial CV (JAX Torsion)
    from src.fp_transformer.cv.torsion import Torsion

    torsion = Torsion(cv_idx)
    import jax

    cv0 = np.asarray(jax.device_get(torsion(coords0)), dtype=np.float64)
    bw0 = init_bandwidth(cv0)

    gaff = load_jax_gaff_energy(str(sponge))
    e0, f0 = jax_energy_and_force(gaff, coords0)

    out = jax_baoab_fp_trajectory(
        sponge_prefix=str(sponge),
        cv_indices=cv_idx,
        coords0=coords0,
        momenta0=momenta0,
        masses=masses,
        ou_noise=ou_noise,
        bandwidth0=bw0,
        dt_akma=dt_akma,
        gamma_akma=gamma_akma,
        beta0=beta0,
        lambda_current=lam,
        kB=kB,
        num_steps=int(args.steps),
        cv_grad_clip_norm=args.cv_grad_clip_norm,
        update_bandwidth=not bool(args.no_update_bandwidth),
        update_pace=int(args.bw_update_pace),
        rescale_per_replica=True,
        bw_min=float(args.rescale_bw_min),
        bw_max=float(args.rescale_bw_max),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coords0.npy", coords0)
    np.save(out_dir / "momenta0.npy", momenta0)
    np.save(out_dir / "ou_noise.npy", ou_noise)
    np.save(out_dir / "bandwidth0.npy", np.asarray([bw0], dtype=np.float64))
    np.save(out_dir / "cv_indices_0based.npy", cv_idx.astype(np.int64))
    np.save(out_dir / "masses.npy", masses)
    np.save(out_dir / "energy0.npy", e0)
    np.save(out_dir / "force0.npy", f0)
    np.save(out_dir / "coordinates.npy", out["coordinates"])
    np.save(out_dir / "momenta.npy", out["momenta"])
    np.save(out_dir / "energies.npy", out["energies"])
    np.save(out_dir / "bandwidths.npy", out["bandwidths"])
    np.savez(
        out_dir / "meta.npz",
        steps=int(args.steps),
        batch=batch,
        seed=int(args.seed),
        baoab_dt_ps=float(args.baoab_dt),
        baoab_gamma_ps=float(args.baoab_langevin_gamma),
        dt_akma=dt_akma,
        gamma_akma=gamma_akma,
        beta0=beta0,
        temperature=float(args.temperature),
        lambda_init=lam,
        kB=kB,
        kernel_grad_beta=float(out["kernel_grad_beta"]),
        bandwidth0=bw0,
        update_pace=int(out["update_pace"]),
        update_bandwidth=not bool(args.no_update_bandwidth),
        rescale_per_replica=True,
        bw_min=float(args.rescale_bw_min),
        bw_max=float(args.rescale_bw_max),
        sponge_prefix=str(sponge),
        init_npy=str(args.init_npy),
        cv_file=str(args.cv_file),
    )
    bw_arr = out["bandwidths"]
    print(
        f"[jax dump] wrote {out_dir}  B={batch} steps={args.steps} "
        f"bw0={bw0:.6g}  update_pace={out['update_pace']}  "
        f"bw_final[min/med/max]="
        f"{bw_arr[-1].min():.4g}/{np.median(bw_arr[-1]):.4g}/{bw_arr[-1].max():.4g}  "
        f"E0_mean={float(e0.mean()):.6g}  "
        f"q_final_finite={np.isfinite(out['coordinates'][-1]).all()}"
    )


if __name__ == "__main__":
    main()
