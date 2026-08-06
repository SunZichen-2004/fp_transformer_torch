"""JAX dialanine GAFF + BAOAB-FP helpers for torch↔JAX parity smoke tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

JAX_ROOT = Path("/share/home/sunzc/work/Fokker_Plank/fp_0429_final/fp-transformer")


def _ensure_jax_path() -> None:
    root = str(JAX_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def jax_available() -> bool:
    if not (JAX_ROOT / "src" / "fp_transformer").is_dir():
        return False
    try:
        _ensure_jax_path()
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        import jax  # noqa: F401

        return True
    except Exception:
        return False


def load_jax_gaff_energy(sponge_prefix: str):
    _ensure_jax_path()
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    from src.fp_transformer.potential.GAFF import GAFF
    from src.fp_transformer.potential.gaff_utils import set_gaff_max_values

    # Tight pad: dialanine has 22 atoms; keep defaults but energy uses real n_atoms.
    set_gaff_max_values()  # defaults
    return GAFF(sponge_prefix=sponge_prefix)


def jax_energy_and_force(gaff, coords_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(E[B,1], F[B,N,3])`` with ``F = -∇U`` (kcal/mol/Å)."""
    _ensure_jax_path()
    import jax
    import jax.numpy as jnp

    coords = jnp.asarray(coords_np, dtype=jnp.float64)

    def energy_sum(x):
        return jnp.sum(gaff(x, beta=1.0))

    e = np.asarray(jax.device_get(gaff(coords, beta=1.0)), dtype=np.float64)
    g = jax.grad(energy_sum)(coords)
    force = -np.asarray(jax.device_get(g), dtype=np.float64)
    return e.reshape(-1, 1), force


def jax_baoab_fp_trajectory(
    *,
    sponge_prefix: str,
    cv_indices: np.ndarray,
    coords0: np.ndarray,
    momenta0: np.ndarray,
    masses: np.ndarray,
    ou_noise: np.ndarray,
    bandwidth0: float | np.ndarray,
    dt_akma: float,
    gamma_akma: float,
    beta0: float,
    lambda_current: float,
    kB: float,
    num_steps: int,
    cv_grad_clip_norm: float | None = None,
    update_bandwidth: bool = True,
    update_pace: int = 20,
    rescale_per_replica: bool = True,
    bw_min: float = 1e-3,
    bw_max: float = 1.0,
    adaptive_ref_quantile: float = 0.5,
    adaptive_band_low: float = 0.8,
    adaptive_band_high: float = 1.2,
    log_step_size: float = 0.05,
    log_error_clip: float = 0.5,
    global_pull: float = 0.02,
) -> dict[str, Any]:
    """Run ``num_steps`` BAOAB-FP steps with *injected* OU noise (JAX GAFF + torus kernel).

    Mirrors ``baoab_fp_sampler._step_core_`` + ``G = ∇U + pb/(λ β0)``.
    Bandwidth refresh uses the same ``update_pace`` gate as JAX FPT
    (``step % update_pace == 0`` → ``Gaussian.update`` / rescale).
    """
    _ensure_jax_path()
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax
    import jax.numpy as jnp
    from jax import grad

    from src.fp_transformer.cv.torsion import Torsion
    from src.fp_transformer.kernel.gaussian import Gaussian, PerReplicaRescaleConfig
    from src.fp_transformer.potential.GAFF import GAFF
    from src.fp_transformer.utils import torus_pairwise_distance

    gaff = GAFF(sponge_prefix=sponge_prefix)
    n_atoms = int(gaff.n_atoms)
    batch = int(coords0.shape[0])
    assert coords0.shape == (batch, n_atoms, 3)
    assert momenta0.shape == coords0.shape
    assert ou_noise.shape == (num_steps, batch, n_atoms, 3)

    torsion = Torsion(np.asarray(cv_indices, dtype=np.int32))
    bw = np.asarray(bandwidth0, dtype=np.float64).reshape(-1)
    if bw.size == 1:
        bw = np.full((batch,), float(bw[0]), dtype=np.float64)
    cfg = PerReplicaRescaleConfig(
        enabled=bool(rescale_per_replica),
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=float(adaptive_ref_quantile),
        adaptive_band_low=float(adaptive_band_low),
        adaptive_band_high=float(adaptive_band_high),
        bw_min=float(bw_min),
        bw_max=float(bw_max),
        log_step_size=float(log_step_size),
        log_error_clip=float(log_error_clip),
        global_pull=float(global_pull),
    )
    kernel = Gaussian(
        batch,
        update_bw_method="rescale",
        bandwidth=jnp.asarray(bw),
        distance_func=torus_pairwise_distance,
        rff=False,
        rescale_per_replica=bool(rescale_per_replica),
        rescale_per_replica_config=cfg,
    )

    mass = jnp.asarray(masses, dtype=jnp.float64).reshape(1, n_atoms, 1)
    inverse_mass = 1.0 / mass
    kgb = float(lambda_current) * float(beta0)
    kB_T = 1.0 / float(beta0)
    exp_m = float(np.exp(-float(gamma_akma) * float(dt_akma)))
    ou_var = kB_T * (1.0 - float(np.exp(-2.0 * float(gamma_akma) * float(dt_akma))))
    scale = jnp.sqrt(jnp.maximum(ou_var * mass, 0.0))
    clip = cv_grad_clip_norm
    pace = max(1, int(update_pace))

    def potential_energy(x):
        return gaff(x, beta=1.0)[:, 0]

    def system_grad(x):
        # ∇U per replica via grad of sum
        return grad(lambda z: jnp.sum(potential_energy(z)))(x)

    def clip_cv_grad(g):
        if clip is None or clip <= 0:
            return g
        b, n, _d = g.shape
        norm = jnp.linalg.norm(g) / (b * n)
        scale_c = jnp.minimum(1.0, clip / (norm + 1e-12))
        return g * scale_c

    def fp_grad(x, bw_vec):
        base = system_grad(x)
        cv = torsion(x)

        def log_density_sum(s):
            return jnp.sum(kernel.kernel_jit(s, bw_vec))

        kg = grad(log_density_sum)(cv)
        kg = clip_cv_grad(kg)

        def cv_dot(z):
            return jnp.sum(torsion(z) * kg)

        pb = grad(cv_dot)(x)
        return base + pb / kgb

    q = jnp.asarray(coords0, dtype=jnp.float64)
    p = jnp.asarray(momenta0, dtype=jnp.float64)
    bw_vec = jnp.asarray(bw, dtype=jnp.float64)
    traj_q = [np.asarray(jax.device_get(q))]
    traj_p = [np.asarray(jax.device_get(p))]
    traj_e = [np.asarray(jax.device_get(potential_energy(q))).reshape(-1)]
    traj_bw = [np.asarray(jax.device_get(bw_vec)).reshape(-1)]

    dt = float(dt_akma)
    for step in range(int(num_steps)):
        if update_bandwidth and (step % pace == 0):
            cv = torsion(q)
            kernel.update(cv)
            bw_vec = jnp.asarray(kernel.bandwidth, dtype=jnp.float64)
        g0 = fp_grad(q, bw_vec)
        f0 = -g0
        p_q = p + 0.5 * dt * f0
        q_h = q + 0.5 * dt * (p_q * inverse_mass)
        xi = jnp.asarray(ou_noise[step], dtype=jnp.float64)
        p_tq = exp_m * p_q + scale * xi
        q_n = q_h + 0.5 * dt * (p_tq * inverse_mass)
        g1 = fp_grad(q_n, bw_vec)
        f1 = -g1
        p = p_tq + 0.5 * dt * f1
        q = q_n
        traj_q.append(np.asarray(jax.device_get(q)))
        traj_p.append(np.asarray(jax.device_get(p)))
        traj_e.append(np.asarray(jax.device_get(potential_energy(q))).reshape(-1))
        traj_bw.append(np.asarray(jax.device_get(bw_vec)).reshape(-1))

    return {
        "coordinates": np.stack(traj_q, axis=0),
        "momenta": np.stack(traj_p, axis=0),
        "energies": np.stack(traj_e, axis=0),
        "bandwidths": np.stack(traj_bw, axis=0),
        "kernel_grad_beta": kgb,
        "update_pace": pace,
    }
