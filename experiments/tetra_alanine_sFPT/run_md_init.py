#!/usr/bin/env python3
"""Unbiased BAOAB Langevin MD for tetra-alanine → init ensemble.

Starts from ``tetra_alanine.pdb`` (same atom order as SPONGE/GAFF), runs
normal MD (no FP kernel), then evenly subsamples frames (after discard)
into ``init_coords_{n_init}.npy`` for sFPT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from io_utils import (  # noqa: E402
    load_pdb_coords,
    load_torsion_indices,
    plot_phi_psi,
)
from fp_transformer_torch import (  # noqa: E402
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaffPotential,
    GaussianKernel,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance  # noqa: E402
from fp_transformer_torch.units import (  # noqa: E402
    KB_KCAL_PER_MOL_K,
    friction_per_picosecond_to_akma,
    timestep_picosecond_to_akma,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, default=EXPERIMENT_DIR / "tetra_alanine.pdb")
    p.add_argument(
        "--sponge-prefix", type=Path, default=EXPERIMENT_DIR / "tetra_alanine"
    )
    p.add_argument("--cv-file", type=Path, default=EXPERIMENT_DIR / "backbone_phi.txt")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--dump-every", type=int, default=2)
    p.add_argument("--discard-steps", type=int, default=2000, help="Equilibration discard")
    p.add_argument("--n-init", type=int, default=4092)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "md_init",
    )
    args = p.parse_args()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xyz = load_pdb_coords(Path(args.pdb))
    pot = GaffPotential(str(args.sponge_prefix))
    if xyz.shape[0] != pot.num_atoms:
        raise ValueError(f"pdb n_atoms={xyz.shape[0]} != GAFF {pot.num_atoms}")

    coordinate = torch.as_tensor(xyz[None, ...], dtype=dtype, device=device)
    energy_system = EnergySystem(pot, coordinate)

    tors = load_torsion_indices(Path(args.cv_file))
    tors = tors[tors.max(axis=1) < pot.num_atoms]
    if tors.shape[0] not in (3, 6):
        raise ValueError(f"expected 3 φ or 6 φψ torsions, got {tors.shape[0]}")
    collective_variable = Torsion(tors)
    kernel = GaussianKernel(
        1,
        bandwidth=0.5,
        distance_function=torus_pairwise_distance,
        rescale_per_replica=False,
        device=device,
        dtype=dtype,
    )

    mass_path = Path(str(args.sponge_prefix) + "_mass.txt")
    mass_vals = np.loadtxt(mass_path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == pot.num_atoms and mass_vals.size == pot.num_atoms + 1:
        mass_vals = mass_vals[1:]
    mass = torch.as_tensor(mass_vals[: pot.num_atoms], dtype=dtype, device=device).view(
        1, -1, 1
    )

    dt_akma = timestep_picosecond_to_akma(float(args.timestep))
    gamma_akma = friction_per_picosecond_to_akma(
        float(args.langevin_friction), float(args.timestep), dt_akma
    )
    beta0 = 1.0 / (KB_KCAL_PER_MOL_K * float(args.temperature))

    integrator = BaoabFokkerPlanckIntegrator(
        energy_system,
        collective_variable,
        kernel,
        timestep=dt_akma,
        langevin_friction=gamma_akma,
        beta0=beta0,
        lambda_init=1.0,
        lambda_max=1.0,
        ess_threshold=0.9,
        min_lambda_step=0.01,
        max_lambda_step=0.1,
        num_lambda_trials=5,
        mass=mass,
        boltzmann_constant=KB_KCAL_PER_MOL_K,
        update_pace=10**9,
        use_kernel_gradient=False,
        collective_variable_grad_clip=None,
        device=device,
        dtype=dtype,
    )
    integrator.kernel_grad_beta = 1e12

    torch.manual_seed(int(args.seed))
    state = integrator.init_state(seed=int(args.seed))
    n_steps = int(args.steps)
    dump_every = max(1, int(args.dump_every))
    discard = max(0, int(args.discard_steps))

    frames: list[np.ndarray] = []
    dihedrals: list[np.ndarray] = []
    print(
        f"[md] pdb={args.pdb}  n_atoms={pot.num_atoms}  steps={n_steps}  "
        f"dump_every={dump_every}  discard={discard}  device={device}",
        flush=True,
    )
    for step in range(1, n_steps + 1):
        state = integrator.step(state, step_index=step, update_bandwidth=False)
        if step > discard and (step % dump_every == 0 or step == n_steps):
            frames.append(state.coordinate.detach().cpu().numpy()[0].copy())
            cv = collective_variable(state.coordinate).detach().cpu().numpy().reshape(-1)
            dihedrals.append(cv.copy())
        if step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps:
            t = integrator._last_step_metrics.get("kinetic_temperature_mean", float("nan"))
            print(
                f"[md] step={step}/{n_steps}  T_kin={t:.1f} K  n_frames={len(frames)}",
                flush=True,
            )

    traj = np.stack(frames, axis=0)
    dih = np.stack(dihedrals, axis=0)
    traj_path = out / f"md_traj_s{args.seed}_{n_steps}.npy"
    dih_path = out / f"md_dihedral_s{args.seed}_{n_steps}.npy"
    np.save(traj_path, traj.astype(np.float64))
    np.save(dih_path, dih.astype(np.float64))
    print(f"[md] wrote {traj_path}  shape={traj.shape}", flush=True)

    n_init = int(args.n_init)
    if traj.shape[0] < n_init:
        raise RuntimeError(
            f"only {traj.shape[0]} frames after discard; need n_init={n_init}. "
            f"Increase --steps or lower --dump-every/--discard-steps."
        )
    idx = np.linspace(0, traj.shape[0] - 1, n_init, dtype=np.int64)
    init = traj[idx]
    init_path = EXPERIMENT_DIR / f"init_coords_{n_init}.npy"
    np.save(init_path, init.astype(np.float64))
    np.save(out / f"init_coords_{n_init}.npy", init.astype(np.float64))
    np.save(out / f"init_dihedral_{n_init}.npy", dih[idx].astype(np.float64))
    print(
        f"[md] wrote init ensemble {init_path}  shape={init.shape}  idx[0..3]={idx[:4]}",
        flush=True,
    )

    plot_phi_psi(
        dih[idx],
        out / f"init_phi_psi_{n_init}.png",
        title=f"MD init 3φ  N={n_init}",
    )
    print("[md] done", flush=True)


if __name__ == "__main__":
    main()
