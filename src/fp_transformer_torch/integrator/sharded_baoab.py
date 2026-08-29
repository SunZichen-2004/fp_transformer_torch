"""BAOAB sFPT with replica-resident multi-GPU MM.

Each GPU keeps its replica coordinates / momenta / MM forces. Only CV atoms
are copied to the primary device for the Fokker–Planck kernel; bias forces
are scattered back onto those atoms. Matches torch-sim NVT Langevin: one
force evaluation per step (cached F for the first B kick).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from ..collective_variable.torsion import Torsion
from ..kernel.gaussian import GaussianKernel
from ..math.pullback import pullback_collective_variable_gradient
from .torch_sim import _clip_collective_variable_gradient

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def cv_atom_indices(collective_variable: Any) -> torch.Tensor:
    """Atom indices the CV actually reads (full-system numbering)."""
    if isinstance(collective_variable, Torsion):
        idx = collective_variable.torsion_indices.reshape(-1)
        return torch.unique(idx.to(dtype=torch.long), sorted=True)
    for attr in ("atom_indices", "indices", "particle_indices"):
        raw = getattr(collective_variable, attr, None)
        if raw is not None:
            idx = torch.as_tensor(raw, dtype=torch.long).reshape(-1)
            return torch.unique(idx, sorted=True)
    raise TypeError(
        "cannot infer CV atoms from "
        f"{type(collective_variable).__name__}; pass cv_atom_idx="
    )


def compact_torsion(cv: Torsion, cv_idx: torch.Tensor) -> Torsion:
    """Rewrite torsion indices into the compact CV-atom layout."""
    inv = torch.empty(int(cv_idx.max().item()) + 1, dtype=torch.long)
    inv[cv_idx] = torch.arange(int(cv_idx.numel()), dtype=torch.long)
    local = inv[cv.torsion_indices]
    return Torsion(local)


def wrap_to_box(pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Map each Cartesian coordinate into ``[0, L)``.

    ``round``-wrap to ``[-L/2, L/2]`` puts the cut through the box centre,
    which splits a solute sitting near ``L/2`` and wrecks dihedrals that
    have no minimum-image convention.
    """
    lengths = box.clamp_min(1e-12)
    return pos - lengths * torch.floor(pos / lengths)


def unwrap_to_ref(
    pos: torch.Tensor,
    box: torch.Tensor,
    ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reassemble atoms into one image via MIC relative to ``ref``.

    ``pos`` is ``(..., n_atoms, 3)``. Default ``ref`` is the first atom.
    """
    lengths = box.clamp_min(1e-12)
    if ref is None:
        ref = pos[..., :1, :]
    delta = pos - ref
    return ref + (delta - lengths * torch.round(delta / lengths))


@dataclass
class _GpuShard:
    worker: Any
    device: torch.device
    stream: torch.cuda.Stream | None
    event: torch.cuda.Event | None
    state: Any
    momenta: torch.Tensor
    forces: torch.Tensor
    inv_mass: torch.Tensor
    masses: torch.Tensor
    lo: int
    hi: int
    n_local: int
    cv_idx: torch.Tensor
    box: torch.Tensor | None


class ShardedBaoabFokkerPlanckIntegrator:
    """Replica-sharded BAOAB + FP kernel on the primary GPU.

    ``run(sim_state, ...)`` has the same heal / anneal API as
    ``TorchSimFokkerPlanckIntegrator``.
    """

    def __init__(
        self,
        mm_model,
        collective_variable: Callable[[torch.Tensor], torch.Tensor],
        gaussian_kernel: GaussianKernel,
        *,
        timestep: float,
        langevin_friction: float,
        temperature: float,
        beta0: float,
        lambda_init: float,
        lambda_max: float,
        ess_threshold: float,
        min_lambda_step: float,
        max_lambda_step: float,
        num_lambda_trials: int,
        update_pace: int = 20,
        log_density_epsilon: float = 1e-12,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        cv_atom_idx: torch.Tensor | None = None,
        use_threads: bool = True,
    ):
        workers = list(getattr(mm_model, "_mm_workers", []) or [])
        if not workers:
            workers = [mm_model]
        self.mm_model = mm_model
        self.workers = workers
        self.collective_variable = collective_variable
        self.gaussian_kernel = gaussian_kernel
        self.timestep = float(timestep)
        self.langevin_friction = float(langevin_friction)
        try:
            from torch_sim.units import MetalUnits

            # Same conversion ts.integrate applies (user ps / ps^-1 → metal internals).
            _time = float(MetalUnits.time)
        except Exception:
            _time = 1.0
        self._dt = self.timestep * _time
        self._gamma = self.langevin_friction / _time
        self.temperature = float(temperature)
        self.beta0 = float(beta0)
        self.kB = 1.0 / (self.beta0 * self.temperature)
        self.lambda_init = float(lambda_init)
        self.lambda_max = float(lambda_max)
        self.ess_threshold = float(ess_threshold)
        self.min_lambda_step = float(min_lambda_step)
        self.max_lambda_step = float(max_lambda_step)
        self.num_lambda_trials = max(2, int(num_lambda_trials))
        self.update_pace = max(1, int(update_pace))
        self.log_density_epsilon = float(log_density_epsilon)
        self.device = torch.device(device or mm_model.device)
        self.dtype = dtype
        self.use_threads = bool(use_threads) and len(workers) > 1

        self.lambda_current = self.lambda_init
        self.kernel_grad_beta = self.lambda_current * self.beta0
        self._density_cache: dict[str, Any] = {}
        self._last_step_metrics: dict[str, float] = {}
        self._shards: list[_GpuShard] = []
        self._n_systems = 0
        self._n_per = int(mm_model.topology.n_atoms)
        self._template_state = None
        self._pool: ThreadPoolExecutor | None = None

        if cv_atom_idx is not None:
            self.cv_idx = torch.as_tensor(cv_atom_idx, dtype=torch.long)
        else:
            self.cv_idx = cv_atom_indices(collective_variable)
        self.cv_idx = self.cv_idx.to(dtype=torch.long)
        if isinstance(collective_variable, Torsion):
            self._cv_compact = compact_torsion(collective_variable, self.cv_idx)
        else:
            self._cv_compact = collective_variable
        self.n_cv = int(self.cv_idx.numel())
        self._cv_pos = torch.empty(0, self.n_cv, 3, device=self.device, dtype=self.dtype)
        self._cv_force = torch.empty_like(self._cv_pos)
        self._cv_idx_primary = self.cv_idx.to(device=self.device)
        self._last_bias_rms = 0.0

        _dyn = getattr(torch, "_dynamo", None)
        if _dyn is not None:
            _dyn.config.disable = True

    def sync_lambda_to_kernel_grad_beta(self) -> None:
        self.kernel_grad_beta = float(self.lambda_current) * float(self.beta0)

    def _ensure_pool(self) -> ThreadPoolExecutor | None:
        if not self.use_threads:
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=len(self.workers))
        return self._pool

    def _bind_state(self, state) -> None:
        import torch_sim as ts
        from ..potential.sharded_mace import shard_system_ranges

        n_per = self._n_per
        n_atoms = int(state.positions.shape[0])
        n_systems = n_atoms // n_per
        if n_atoms != n_systems * n_per:
            raise ValueError("unequal atoms per system")
        self._n_systems = n_systems
        self._template_state = state
        ranges = shard_system_ranges(n_systems, len(self.workers))
        workers = self.workers[: len(ranges)]
        pos_view = state.positions.view(n_systems, n_per, 3)
        mass_view = state.masses.view(n_systems, n_per)
        z_view = state.atomic_numbers.view(n_systems, n_per)
        cell_src = state.row_vector_cell
        mom_src = getattr(state, "momenta", None)
        kT = 1.0 / self.beta0
        shards: list[_GpuShard] = []
        for (lo, hi), worker in zip(ranges, workers):
            dev = worker.device
            b = hi - lo
            pos = pos_view[lo:hi].reshape(-1, 3).to(
                device=dev, dtype=self.dtype, non_blocking=True
            )
            mass = mass_view[lo:hi].reshape(-1).to(
                device=dev, dtype=self.dtype, non_blocking=True
            )
            z = z_view[lo:hi].reshape(-1).to(device=dev, dtype=torch.long, non_blocking=True)
            if cell_src.shape[0] == n_systems:
                cell = cell_src[lo:hi]
            else:
                cell = cell_src[0:1].expand(b, 3, 3)
            cell = cell.to(device=dev, dtype=self.dtype, non_blocking=True).contiguous()
            system_idx = torch.arange(b, device=dev, dtype=torch.long).repeat_interleave(
                n_per
            )
            pbc = state.pbc
            if torch.is_tensor(pbc):
                pbc = pbc.to(device=dev, non_blocking=True)
            shard_state = ts.SimState(
                positions=pos,
                masses=mass,
                cell=cell,
                pbc=pbc,
                atomic_numbers=z,
                system_idx=system_idx,
            )
            inv_mass = (1.0 / mass.clamp_min(1e-12)).unsqueeze(-1)
            if mom_src is not None:
                momenta = mom_src.view(n_systems, n_per, 3)[lo:hi].reshape(-1, 3).to(
                    device=dev, dtype=self.dtype, non_blocking=True
                )
            else:
                scale = torch.sqrt(mass.unsqueeze(-1) * kT)
                momenta = scale * torch.randn(
                    pos.shape, device=dev, dtype=self.dtype
                )
            box = None
            if worker._ortho_box is not None:
                box = worker._ortho_box.to(device=dev, dtype=self.dtype)
            # Default stream only. Side streams raced with GPU0 bias and
            # dropped the FP force (90523: bandwidth frozen, CV stuck in β).
            shards.append(
                _GpuShard(
                    worker=worker,
                    device=dev,
                    stream=None,
                    event=None,
                    state=shard_state,
                    momenta=momenta,
                    forces=torch.zeros_like(pos),
                    inv_mass=inv_mass,
                    masses=mass,
                    lo=lo,
                    hi=hi,
                    n_local=b,
                    cv_idx=self.cv_idx.to(device=dev),
                    box=box,
                )
            )
        self._shards = shards
        self._cv_pos = torch.empty(
            n_systems, self.n_cv, 3, device=self.device, dtype=self.dtype
        )
        self._cv_force = torch.empty_like(self._cv_pos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _sync_devices(self) -> None:
        if not torch.cuda.is_available():
            return
        seen: set[int] = set()
        for shard in self._shards:
            if shard.device.type != "cuda":
                continue
            idx = shard.device.index
            if idx is None:
                idx = torch.cuda.current_device()
            if idx in seen:
                continue
            seen.add(idx)
            torch.cuda.synchronize(shard.device)

    def _kick(self, shard: _GpuShard, dt_half: float) -> None:
        shard.momenta.add_(shard.forces, alpha=dt_half)

    def _drift(self, shard: _GpuShard, dt_half: float) -> None:
        # Match torch-sim NVT Langevin: do not wrap. MM / PME already use MIC.
        # Per-atom wrap splits a solute that sits across a box face and wrecks
        # dihedrals that have no minimum-image convention.
        shard.state.positions.add_(shard.momenta * shard.inv_mass, alpha=dt_half)

    def _ou(self, shard: _GpuShard, dt: float) -> None:
        c1 = float(np.exp(-self._gamma * dt))
        kT = 1.0 / self.beta0
        c2 = float(np.sqrt(max(kT * (1.0 - c1 * c1), 0.0)))
        noise = torch.randn(
            shard.momenta.shape, device=shard.device, dtype=shard.momenta.dtype
        )
        shard.momenta.mul_(c1).add_(noise * torch.sqrt(shard.masses).unsqueeze(-1), alpha=c2)

    def _mm_one(self, shard: _GpuShard) -> None:
        out = shard.worker(shard.state)
        shard.forces.copy_(out["forces"])
        pos = shard.state.positions.view(shard.n_local, self._n_per, 3)
        cv = pos[:, shard.cv_idx, :]
        if shard.box is not None:
            cv = unwrap_to_ref(cv, shard.box)
        self._cv_pos[shard.lo : shard.hi].copy_(cv)

    def _bias_and_scatter(self) -> None:
        pos_cv = self._cv_pos.detach().requires_grad_(True)
        kernel_grad = self.gaussian_kernel.kernel_gradient(
            self._cv_compact(pos_cv),
            self.gaussian_kernel.bandwidth,
        )
        kernel_grad = _clip_collective_variable_gradient(kernel_grad, None)
        g_bias = pullback_collective_variable_gradient(
            pos_cv, self._cv_compact, kernel_grad
        )
        self._cv_force.copy_((-g_bias / float(self.kernel_grad_beta)).detach())
        n_per = self._n_per
        for shard in self._shards:
            extra = self._cv_force[shard.lo : shard.hi].to(device=shard.device)
            # Advanced indexing `view[:, idx, :]` returns a copy; add_ would
            # be discarded. index_add_ writes back into shard.forces.
            view = shard.forces.view(shard.n_local, n_per, 3)
            view.index_add_(1, shard.cv_idx, extra)
        self._last_bias_rms = float(self._cv_force.square().mean().sqrt().item())

    def _evaluate_force(self) -> None:
        pool = self._ensure_pool()
        if pool is not None:
            futs = [pool.submit(self._mm_one, s) for s in self._shards]
            for fut in futs:
                fut.result()
        else:
            for shard in self._shards:
                self._mm_one(shard)
        self._sync_devices()
        self._bias_and_scatter()
        self._sync_devices()

    def _kinetic_stats(self) -> dict[str, float]:
        temps: list[torch.Tensor] = []
        ndof = float(self._n_per * 3)
        kB = self.kB
        for shard in self._shards:
            ke = 0.5 * (shard.momenta.square() * shard.inv_mass).sum(dim=-1)
            ke = ke.view(shard.n_local, self._n_per).sum(dim=-1)
            temps.append((2.0 * ke) / (ndof * kB))
        t = torch.cat([x.to(device=self.device) for x in temps]).detach()
        return {
            "kinetic_temperature_mean": float(t.mean().item()),
            "kinetic_temperature_std": float(t.std(unbiased=False).item())
            if t.numel() > 1
            else 0.0,
            "kinetic_temperature_min": float(t.min().item()),
            "kinetic_temperature_max": float(t.max().item()),
        }

    def gather_sim_state(self):
        """Full coordinates / momenta back on the primary device."""
        import torch_sim as ts

        n_per = self._n_per
        n_sys = self._n_systems
        pos = torch.empty(n_sys, n_per, 3, device=self.device, dtype=self.dtype)
        mom = torch.empty_like(pos)
        mass = torch.empty(n_sys, n_per, device=self.device, dtype=self.dtype)
        z = torch.empty(n_sys, n_per, device=self.device, dtype=torch.long)
        for shard in self._shards:
            if shard.stream is not None:
                shard.stream.synchronize()
            pos[shard.lo : shard.hi] = shard.state.positions.view(
                shard.n_local, n_per, 3
            ).to(self.device)
            mom[shard.lo : shard.hi] = shard.momenta.view(shard.n_local, n_per, 3).to(
                self.device
            )
            mass[shard.lo : shard.hi] = shard.masses.view(shard.n_local, n_per).to(
                self.device
            )
            z[shard.lo : shard.hi] = shard.state.atomic_numbers.view(
                shard.n_local, n_per
            ).to(self.device)
        box = self._shards[0].box
        if box is not None:
            pos.copy_(unwrap_to_ref(pos, box.to(device=self.device, dtype=self.dtype)))
        cell = self._template_state.row_vector_cell
        if cell.shape[0] == 1 and n_sys > 1:
            cell = cell.expand(n_sys, 3, 3).contiguous()
        cell = cell.to(device=self.device, dtype=self.dtype)
        out = ts.SimState(
            positions=pos.reshape(-1, 3),
            masses=mass.reshape(-1),
            cell=cell,
            pbc=self._template_state.pbc,
            atomic_numbers=z.reshape(-1),
            system_idx=torch.arange(n_sys, device=self.device).repeat_interleave(n_per),
        )
        return out

    def _cv_all(self) -> torch.Tensor:
        return self._cv_compact(self._cv_pos)

    def step(self, *, step_index: int = 0) -> None:
        dt = self._dt
        do_bw = step_index % self.update_pace == 0
        if do_bw:
            collective = self._cv_all().detach()
            self.gaussian_kernel.update_bandwidth(collective)

        dt_half = 0.5 * dt
        for shard in self._shards:
            self._kick(shard, dt_half)
            self._drift(shard, dt_half)
            self._ou(shard, dt)
            self._drift(shard, dt_half)

        self._evaluate_force()

        for shard in self._shards:
            self._kick(shard, dt_half)

        if step_index % self.update_pace == 0:
            stats = self._kinetic_stats()
            bw = self.gaussian_kernel.bandwidth.detach()
            stats.update(
                {
                    "bandwidth_min": float(bw.min().item()),
                    "bandwidth_max": float(bw.max().item()),
                    "bandwidth_median": float(bw.median().item()),
                    "kernel_grad_beta": float(self.kernel_grad_beta),
                    "force_bias_rms": float(self._last_bias_rms),
                }
            )
            rs = getattr(self.gaussian_kernel, "_last_row_sum_stats", None) or {}
            for key in (
                "row_sum_min",
                "row_sum_max",
                "row_sum_median",
                "target_low",
                "target_high",
            ):
                if key in rs:
                    stats[key] = float(rs[key])
            self._last_step_metrics = stats

    def run_stage(
        self,
        system,
        num_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
    ):
        self.sync_lambda_to_kernel_grad_beta()
        if not self._shards:
            self._bind_state(system)
            self._evaluate_force()
        num_steps = int(num_steps)
        iterator = range(num_steps)
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                iterator,
                total=num_steps,
                desc="sharded BAOAB-FP",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            iterator = progress
        t0 = time.perf_counter()
        for step_index in iterator:
            self.step(step_index=step_index)
            done = step_index + 1
            if verbose and done % verbose_pace == 0:
                elapsed = max(time.perf_counter() - t0, 1e-12)
                speed = done / elapsed
                metrics = self._last_step_metrics
                msg = (
                    f"torch_sim Step {done}, "
                    f"lambda={self.lambda_current:.6g}, "
                    f"kernel_grad_beta={self.kernel_grad_beta:.6g}, "
                    f"T_kin mean/std/min/max="
                    f"{metrics.get('kinetic_temperature_mean', float('nan')):.2f}/"
                    f"{metrics.get('kinetic_temperature_std', float('nan')):.2f}/"
                    f"{metrics.get('kinetic_temperature_min', float('nan')):.2f}/"
                    f"{metrics.get('kinetic_temperature_max', float('nan')):.2f} K "
                    f"(target={self.temperature:.2f} K), "
                    f"Kernel bw max/min/median: "
                    f"{metrics.get('bandwidth_max', float('nan')):.4e} / "
                    f"{metrics.get('bandwidth_min', float('nan')):.4e} / "
                    f"{metrics.get('bandwidth_median', float('nan')):.4e}"
                    + (
                        f", F_bias_rms={metrics.get('force_bias_rms', float('nan')):.3e}"
                        if "force_bias_rms" in metrics
                        else ""
                    )
                    + (
                        f", mean_row[min/med/max]="
                        f"{metrics.get('row_sum_min', float('nan')):.4g}/"
                        f"{metrics.get('row_sum_median', float('nan')):.4g}/"
                        f"{metrics.get('row_sum_max', float('nan')):.4g}"
                        f" target=[{metrics.get('target_low', float('nan')):.4g},"
                        f"{metrics.get('target_high', float('nan')):.4g}]"
                        if "row_sum_min" in metrics
                        else ""
                    )
                    + f", {speed:.2f} it/s"
                )
                if progress is not None:
                    progress.write(msg)
                else:
                    print(msg, flush=True)
        elapsed = max(time.perf_counter() - t0, 1e-12)
        speed = num_steps / elapsed
        if progress is not None:
            progress.close()
        print(
            f"[ShardedBaoabFokkerPlanck] stage done: {num_steps} steps in {elapsed:.2f}s "
            f"→ {speed:.2f} it/s",
            flush=True,
        )
        return self.gather_sim_state()

    def estimate_collective_variable_density(self, coordinate: torch.Tensor) -> dict[str, Any]:
        collective = self.collective_variable(coordinate).detach()
        bandwidth = self.gaussian_kernel.bandwidth.detach()
        kernel_matrix = self.gaussian_kernel.kernel_matrix(collective, bandwidth)
        row_sum = torch.sum(kernel_matrix, dim=-1)
        batch = int(collective.shape[0])
        log_density = torch.log(row_sum / float(batch) + self.log_density_epsilon)
        log_density_np = log_density.detach().cpu().numpy().astype(np.float64)

        def log_density_fn(points: np.ndarray) -> np.ndarray:
            del points
            return log_density_np

        info = {
            "collective_variable_reference": collective.detach().cpu().numpy(),
            "bandwidth": bandwidth.detach().cpu().numpy(),
            "log_density_fn": log_density_fn,
            "row_sum": row_sum.detach().cpu().numpy(),
            "num_replicas": batch,
            "log_density": log_density_np,
        }
        self._density_cache = info
        return info

    @staticmethod
    def compute_ess(weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        sum_squares = float(np.sum(weights * weights))
        if sum_squares <= 0.0:
            return 0.0
        return (total * total) / sum_squares

    @staticmethod
    def compute_reweight_weights(
        collective_variable_values: np.ndarray,
        density_info: dict[str, Any],
        lambda_current: float,
        lambda_next: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_density = density_info["log_density_fn"](
            np.asarray(collective_variable_values, dtype=np.float64)
        )
        eta = (lambda_next - lambda_current) / (lambda_current * (1.0 + lambda_next))
        log_weights = eta * log_density
        log_weights = log_weights - np.max(log_weights)
        unnormalized = np.exp(log_weights)
        normalized = unnormalized / np.sum(unnormalized)
        return unnormalized, normalized

    def choose_next_lambda(
        self,
        collective_variable_values: np.ndarray,
        density_info: dict[str, Any],
        lambda_current: float,
    ) -> float:
        low = lambda_current + self.min_lambda_step
        high = min(self.lambda_max, lambda_current + self.max_lambda_step)
        if low >= high - 1e-15:
            return min(lambda_current + self.min_lambda_step, self.lambda_max)
        trials = np.linspace(low, high, self.num_lambda_trials)
        batch = int(np.asarray(collective_variable_values).shape[0])
        chosen = lambda_current
        for candidate in trials:
            unnormalized, _ = self.compute_reweight_weights(
                collective_variable_values,
                density_info,
                lambda_current,
                float(candidate),
            )
            ess_ratio = self.compute_ess(unnormalized) / float(batch)
            if ess_ratio >= self.ess_threshold:
                chosen = float(candidate)
        if chosen <= lambda_current + 1e-15:
            chosen = float(trials[0])
        return min(chosen, self.lambda_max)

    def advance_lambda(self, state, collective_variable_values, density_info, lambda_current, lambda_next):
        unnormalized, normalized = self.compute_reweight_weights(
            collective_variable_values, density_info, lambda_current, lambda_next
        )
        self.lambda_current = float(lambda_next)
        self.sync_lambda_to_kernel_grad_beta()
        return state, unnormalized, normalized

    def run(
        self,
        system,
        stage_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
        on_lambda_change: Optional[Callable[..., Any]] = None,
        on_stage_end: Optional[Callable[..., Any]] = None,
        heal_lambdas: Optional[Sequence[float]] = None,
        heal_steps: int | None = None,
        max_stages: int | None = None,
    ):
        from ..potential.mace import sim_state_to_batched_coordinates

        def _call_stage_end(stage_i: int, lam: float, st, *, phase: str) -> None:
            if on_stage_end is None:
                return
            try:
                on_stage_end(stage_i, float(lam), st, phase=phase)
            except TypeError:
                on_stage_end(stage_i, float(lam), st)

        self._bind_state(system)
        self._evaluate_force()
        state = system
        tag = "ShardedBaoabFokkerPlanck"
        if verbose:
            fac = self._dt / max(self.timestep, 1e-30)
            print(
                f"[{tag}] metal units: dt={self._dt:.6g} "
                f"(user {self.timestep:g} ps × {fac:.6g}), "
                f"gamma={self._gamma:.6g} "
                f"(user {self.langevin_friction:g} /ps); "
                f"gamma*dt={self._gamma * self._dt:.6g}",
                flush=True,
            )

        if heal_lambdas:
            heal_list = [float(x) for x in heal_lambdas]
            n_heal = int(heal_steps) if heal_steps is not None else int(stage_steps)
            if verbose:
                print(
                    f"[{tag}] heal: {len(heal_list)} λ values {heal_list} × {n_heal} steps each",
                    flush=True,
                )
            for heal_i, lam in enumerate(heal_list):
                self.lambda_current = float(lam)
                self.sync_lambda_to_kernel_grad_beta()
                if verbose:
                    print(
                        f"[{tag}] heal {heal_i}: {n_heal} steps at lambda={self.lambda_current:.8g}",
                        flush=True,
                    )
                state = self.run_stage(
                    state, n_heal, verbose=verbose, verbose_pace=verbose_pace,
                    show_progress=show_progress,
                )
                _call_stage_end(heal_i, float(self.lambda_current), state, phase="heal")

        self.lambda_current = float(self.lambda_init)
        self.sync_lambda_to_kernel_grad_beta()
        stage_index = 0
        if verbose:
            print(
                f"[{tag}] anneal: start λ={self.lambda_current:.8g} "
                f"→ max={self.lambda_max:.8g}, {stage_steps} steps/stage",
                flush=True,
            )
        while self.lambda_current < self.lambda_max - 1e-12:
            if max_stages is not None and stage_index >= int(max_stages):
                if verbose:
                    print(f"[{tag}] max_stages={max_stages} reached", flush=True)
                break
            if verbose:
                print(
                    f"[{tag}] anneal stage {stage_index}: "
                    f"{stage_steps} steps at lambda={self.lambda_current:.8g}",
                    flush=True,
                )
            state = self.run_stage(
                state,
                int(stage_steps),
                verbose=verbose,
                verbose_pace=verbose_pace,
                show_progress=show_progress,
            )
            _call_stage_end(stage_index, float(self.lambda_current), state, phase="anneal")
            coords = sim_state_to_batched_coordinates(state)
            density_info = self.estimate_collective_variable_density(coords)
            collective_np = self.collective_variable(coords).detach().cpu().numpy()
            lambda_k = float(self.lambda_current)
            lambda_next = float(self.choose_next_lambda(collective_np, density_info, lambda_k))
            if lambda_next <= lambda_k + 1e-15:
                if verbose:
                    print(f"[{tag}] cannot increase lambda beyond {lambda_k}", flush=True)
                break
            state_before = state
            state, unnormalized, normalized = self.advance_lambda(
                state, collective_np, density_info, lambda_k, lambda_next
            )
            if verbose:
                print(
                    f"[{tag}] lambda {lambda_k:.8g} → {lambda_next:.8g} "
                    f"(ESS/N={self.compute_ess(unnormalized) / collective_np.shape[0]:.4g})",
                    flush=True,
                )
            if on_lambda_change is not None:
                on_lambda_change(
                    lambda_k, lambda_next, state, density_info, stage_index,
                    state_before_advance=state_before,
                    unnormalized_weights=unnormalized,
                    normalized_weights=normalized,
                )
            stage_index += 1
        return state
