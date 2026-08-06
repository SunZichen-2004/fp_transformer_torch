#!/usr/bin/env python3
"""Compare torch dialanine BAOAB-FP trajectory to JAX dump in parity_ref/.

Focus: bandwidth update (``update_pace`` + per-replica rescale) vs JAX FPT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fp_transformer_torch import (  # noqa: E402
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaffPotential,
    GaussianKernel,
    PerReplicaRescaleConfig,
    SamplerState,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance  # noqa: E402

ATOL_ENERGY = 1e-6
ATOL_FORCE = 5e-6
ATOL_TRAJ = 5e-6
# Bandwidth updates amplify tiny CV drift; 1e-7 is still ≪ physical σ scale.
ATOL_BW = 1e-7


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ref-dir",
        type=Path,
        default=ROOT / "experiments" / "dialanine" / "parity_ref",
    )
    p.add_argument("--atol-traj", type=float, default=ATOL_TRAJ)
    p.add_argument("--atol-bw", type=float, default=ATOL_BW)
    args = p.parse_args()
    ref = Path(args.ref_dir)
    required = [
        "coords0.npy",
        "momenta0.npy",
        "ou_noise.npy",
        "bandwidth0.npy",
        "cv_indices_0based.npy",
        "masses.npy",
        "coordinates.npy",
        "momenta.npy",
        "energies.npy",
        "bandwidths.npy",
        "meta.npz",
        "energy0.npy",
        "force0.npy",
    ]
    missing = [n for n in required if not (ref / n).is_file()]
    if missing:
        raise FileNotFoundError(f"missing JAX dump files in {ref}: {missing}")

    meta = np.load(ref / "meta.npz", allow_pickle=True)
    coords0 = np.load(ref / "coords0.npy")
    momenta0 = np.load(ref / "momenta0.npy")
    ou_noise = np.load(ref / "ou_noise.npy")
    bw0 = float(np.load(ref / "bandwidth0.npy").reshape(-1)[0])
    cv_idx = np.load(ref / "cv_indices_0based.npy")
    masses = np.load(ref / "masses.npy")
    e_jax0 = np.load(ref / "energy0.npy")
    f_jax0 = np.load(ref / "force0.npy")
    q_jax = np.load(ref / "coordinates.npy")
    p_jax = np.load(ref / "momenta.npy")
    e_jax = np.load(ref / "energies.npy")
    bw_jax = np.load(ref / "bandwidths.npy")

    sponge = Path(str(meta["sponge_prefix"]))
    dt = float(meta["dt_akma"])
    gamma = float(meta["gamma_akma"])
    beta0 = float(meta["beta0"])
    kB = float(meta["kB"])
    lam = float(meta["lambda_init"])
    num_steps = int(meta["steps"])
    batch = int(coords0.shape[0])
    update_pace = int(meta["update_pace"]) if "update_pace" in meta.files else 20
    update_bw = (
        bool(meta["update_bandwidth"]) if "update_bandwidth" in meta.files else True
    )
    bw_min = float(meta["bw_min"]) if "bw_min" in meta.files else 1e-3
    bw_max = float(meta["bw_max"]) if "bw_max" in meta.files else 1.0

    pot = GaffPotential(str(sponge))
    x0 = torch.as_tensor(coords0, dtype=torch.float64)
    e_torch0 = pot.energy(x0).detach().cpu().numpy()
    f_torch0 = pot.force(x0).detach().cpu().numpy()
    np.testing.assert_allclose(e_torch0, e_jax0, rtol=0.0, atol=ATOL_ENERGY)
    np.testing.assert_allclose(f_torch0, f_jax0, rtol=0.0, atol=ATOL_FORCE)
    print(f"[torch] energy/force vs JAX dump OK  (atol_E={ATOL_ENERGY}, atol_F={ATOL_FORCE})")

    mass_t = torch.as_tensor(masses, dtype=torch.float64).view(1, -1, 1)
    cv = Torsion(cv_idx)
    rescale_cfg = PerReplicaRescaleConfig(
        enabled=True,
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=0.5,
        adaptive_band_low=0.8,
        adaptive_band_high=1.2,
        bandwidth_min=bw_min,
        bandwidth_max=bw_max,
        log_step_size=0.05,
        log_error_clip=0.5,
        global_pull=0.02,
        # Match JAX dump: unnormalized K, raw row_sum, clip [2,4]
        use_normalized_kernel=False,
        normalize_row_sum_by_replicas=False,
        fixed_row_sum_min=2.0,
        fixed_row_sum_max=4.0,
    )
    kernel = GaussianKernel(
        batch,
        bandwidth=bw0,
        distance_function=torus_pairwise_distance,
        rescale_per_replica=True,
        rescale_per_replica_config=rescale_cfg,
        dtype=torch.float64,
    )
    integ = BaoabFokkerPlanckIntegrator(
        EnergySystem(pot, x0.clone()),
        cv,
        kernel,
        timestep=dt,
        langevin_friction=gamma,
        beta0=beta0,
        lambda_init=lam,
        lambda_max=lam,
        ess_threshold=0.97,
        min_lambda_step=0.001,
        max_lambda_step=0.02,
        num_lambda_trials=2,
        mass=mass_t,
        boltzmann_constant=kB,
        update_pace=update_pace,
        use_kernel_gradient=True,
        collective_variable_grad_clip=None,
        dtype=torch.float64,
    )
    integ.sync_lambda_to_kernel_grad_beta()
    state = SamplerState(
        coordinate=x0.clone(),
        momentum=torch.as_tensor(momenta0, dtype=torch.float64),
        generator=torch.Generator().manual_seed(0),
    )

    print(
        f"[torch] update_pace={update_pace}  update_bandwidth={update_bw}  "
        f"bw_clip=[{bw_min:g},{bw_max:g}]"
    )

    torch_q = [coords0.copy()]
    torch_p = [momenta0.copy()]
    torch_e = [e_torch0.reshape(-1)]
    torch_bw = [kernel.bandwidth.detach().cpu().numpy().reshape(-1).copy()]
    for step in range(num_steps):
        noise = torch.as_tensor(ou_noise[step], dtype=torch.float64)
        # None → use integrator update_pace gate (same as JAX FPT).
        state = integ.step(
            state,
            step_index=step,
            ou_noise=noise,
            update_bandwidth=None if update_bw else False,
        )
        torch_q.append(state.coordinate.detach().cpu().numpy())
        torch_p.append(state.momentum.detach().cpu().numpy())
        torch_e.append(
            pot.energy(state.coordinate).detach().cpu().numpy().reshape(-1)
        )
        torch_bw.append(kernel.bandwidth.detach().cpu().numpy().reshape(-1).copy())

    torch_q = np.stack(torch_q, axis=0)
    torch_p = np.stack(torch_p, axis=0)
    torch_e = np.stack(torch_e, axis=0)
    torch_bw = np.stack(torch_bw, axis=0)

    np.testing.assert_allclose(
        torch_bw, bw_jax, rtol=0.0, atol=args.atol_bw, err_msg="bandwidth traj"
    )
    np.testing.assert_allclose(
        torch_e, e_jax, rtol=0.0, atol=5e-6, err_msg="energy traj"
    )
    np.testing.assert_allclose(
        torch_q, q_jax, rtol=0.0, atol=args.atol_traj, err_msg="coordinate traj"
    )
    np.testing.assert_allclose(
        torch_p, p_jax, rtol=0.0, atol=args.atol_traj, err_msg="momentum traj"
    )
    dq = float(np.max(np.abs(torch_q - q_jax)))
    dp = float(np.max(np.abs(torch_p - p_jax)))
    de = float(np.max(np.abs(torch_e - e_jax)))
    db = float(np.max(np.abs(torch_bw - bw_jax)))
    print(
        f"[parity] OK  max|Δq|={dq:.3e}  max|Δp|={dp:.3e}  max|ΔE|={de:.3e}  "
        f"max|Δbw|={db:.3e}  (steps={num_steps}, B={batch}, pace={update_pace})"
    )
    # Summarize bandwidth evolution at update steps
    for s in range(0, num_steps + 1, update_pace):
        if s >= torch_bw.shape[0]:
            break
        b = torch_bw[s]
        print(
            f"  step={s:4d}  bw[min/med/max]="
            f"{b.min():.6g}/{np.median(b):.6g}/{b.max():.6g}"
        )


if __name__ == "__main__":
    main()
