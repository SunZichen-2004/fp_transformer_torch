"""Fokker–Planck bias as a torch-sim ``ModelInterface`` (+ annealing driver).

``SumModel(mace_model, FPTBias(...))`` adds the FP pullback force on top of the
base ML potential. Sign convention matches
``BaoabFokkerPlanckIntegrator.fokker_planck_gradient``::

    G = ∇U + pullback(∇_s log-kernel) / kernel_grad_beta
    F = -G

so the bias force contributed by ``FPTBias`` is::

    F_bias = -pullback / kernel_grad_beta

Base ``∇U`` / ``F = -∇U`` comes from MACE (or any other child model).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from ..kernel.gaussian import GaussianKernel
from ..math.pullback import pullback_collective_variable_gradient
from ..potential.mace import (
    sim_state_to_batched_coordinates,
)
from .sampler_state import SamplerState

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _clip_collective_variable_gradient(
    gradient: torch.Tensor,
    max_norm: float | None,
) -> torch.Tensor:
    if max_norm is None:
        return gradient
    batch, atoms, dim = gradient.shape
    del dim
    norm = torch.linalg.norm(gradient) / float(batch * atoms)
    scale = torch.clamp(
        torch.tensor(max_norm, device=gradient.device, dtype=gradient.dtype)
        / (norm + 1e-12),
        max=1.0,
    )
    return gradient * scale


try:
    from torch_sim.models.interface import ModelInterface as _ModelInterface
except ImportError:  # pragma: no cover
    _ModelInterface = torch.nn.Module  # type: ignore[misc, assignment]


class FPTBias(_ModelInterface):
    """torch-sim ``ModelInterface`` returning only the FP pullback force.

    Energy is returned as zeros (unused by Langevin force evaluation).
    Force sign matches BAOAB ``F = -G`` with
    ``G_bias = pullback / kernel_grad_beta``.
    """

    def __init__(
        self,
        collective_variable: Callable[[torch.Tensor], torch.Tensor],
        gaussian_kernel: GaussianKernel,
        *,
        kernel_grad_beta: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
        use_kernel_gradient: bool = True,
        collective_variable_grad_clip: float | None = None,
        update_pace: int = 20,
        n_atoms_per_system: int | None = None,
    ):
        if _ModelInterface is torch.nn.Module:
            raise ImportError(
                "torch_sim is required for FPTBias. Activate conda env MACE_torchsim."
            )
        super().__init__()
        self.collective_variable = collective_variable
        self.gaussian_kernel = gaussian_kernel
        self.kernel_grad_beta = float(kernel_grad_beta)
        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"
        self.use_kernel_gradient = bool(use_kernel_gradient)
        self.collective_variable_grad_clip = collective_variable_grad_clip
        self.update_pace = max(1, int(update_pace))
        self.n_atoms_per_system = n_atoms_per_system
        self._forward_count = 0
        self._last_metrics: dict[str, float] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, value: bool) -> None:  # noqa: FBT001
        if value:
            raise NotImplementedError("FPTBias does not compute stress")
        self._compute_stress = False

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @compute_forces.setter
    def compute_forces(self, value: bool) -> None:  # noqa: FBT001
        self._compute_forces = bool(value)

    @property
    def memory_scales_with(self):
        return self._memory_scales_with

    def set_kernel_grad_beta(self, value: float) -> None:
        self.kernel_grad_beta = float(value)

    def bias_gradient(self, coordinate: torch.Tensor) -> torch.Tensor:
        """``pullback / kernel_grad_beta`` shaped ``(B, N, 3)`` (same as G_bias)."""
        if not self.use_kernel_gradient:
            return torch.zeros_like(coordinate)
        collective = self.collective_variable(coordinate)
        kernel_grad = self.gaussian_kernel.kernel_gradient(
            collective, self.gaussian_kernel.bandwidth
        )
        kernel_grad = _clip_collective_variable_gradient(
            kernel_grad, self.collective_variable_grad_clip
        )
        pullback = pullback_collective_variable_gradient(
            coordinate, self.collective_variable, kernel_grad
        )
        return pullback / float(self.kernel_grad_beta)

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_atoms = int(state.n_atoms)
        if self.n_atoms_per_system is not None:
            n_per = int(self.n_atoms_per_system)
            if n_per * n_systems != n_atoms:
                raise ValueError(
                    f"n_atoms_per_system={n_per} incompatible with "
                    f"n_atoms={n_atoms}, n_systems={n_systems}"
                )
        else:
            if n_atoms % n_systems != 0:
                raise ValueError("unequal atoms per system in SimState")
            n_per = n_atoms // n_systems

        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        g_bias = self.bias_gradient(coordinate)
        # F = -G  ⇒  F_bias = -pullback / kernel_grad_beta
        forces = (-g_bias).reshape(n_atoms, 3).detach()
        energy = torch.zeros(n_systems, device=self._device, dtype=self._dtype)
        self._forward_count += 1

        bw = self.gaussian_kernel.bandwidth.detach()
        self._last_metrics = {
            "bandwidth_min": float(bw.min().item()),
            "bandwidth_max": float(bw.max().item()),
            "bandwidth_median": float(bw.median().item()),
            "kernel_grad_beta": float(self.kernel_grad_beta),
            "force_bias_rms": float(torch.sqrt(torch.mean(forces * forces)).item()),
        }
        rs = getattr(self.gaussian_kernel, "_last_row_sum_stats", None) or {}
        for key in (
            "row_sum_min",
            "row_sum_max",
            "row_sum_median",
            "target_low",
            "target_high",
        ):
            if key in rs:
                self._last_metrics[key] = float(rs[key])

        return {"energy": energy, "forces": forces}


class TorchSimFokkerPlanckIntegrator:
    """λ-annealing driver around ``torch_sim.integrate`` + ``SumModel``.

    Thermal bath / BAOAB is handled by ``ts.Integrator.nvt_langevin`` (metal units:
    timestep in ps, friction in 1/ps, temperature in K). FP bias strength is
    ``kernel_grad_beta = λ · β0`` with ``β0 = 1/(k_B T)`` in eV⁻¹.
    """

    def __init__(
        self,
        model,
        fpt_bias: FPTBias,
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
        dtype: torch.dtype = torch.float64,
    ):
        self.model = model
        self.fpt_bias = fpt_bias
        self.collective_variable = collective_variable
        self.gaussian_kernel = gaussian_kernel
        self.timestep = float(timestep)
        self.langevin_friction = float(langevin_friction)
        self.temperature = float(temperature)
        self.beta0 = float(beta0)
        self.lambda_init = float(lambda_init)
        self.lambda_max = float(lambda_max)
        self.ess_threshold = float(ess_threshold)
        self.min_lambda_step = float(min_lambda_step)
        self.max_lambda_step = float(max_lambda_step)
        self.num_lambda_trials = max(2, int(num_lambda_trials))
        self.update_pace = max(1, int(update_pace))
        self.log_density_epsilon = float(log_density_epsilon)
        self.device = torch.device(device or fpt_bias.device)
        self.dtype = dtype

        self.lambda_current = self.lambda_init
        self.kernel_grad_beta = self.lambda_current * self.beta0
        self.fpt_bias.set_kernel_grad_beta(self.kernel_grad_beta)
        self.fpt_bias.update_pace = self.update_pace
        self._density_cache: dict[str, Any] = {}
        self._last_step_metrics: dict[str, float] = {}

    def sync_lambda_to_kernel_grad_beta(self) -> None:
        self.kernel_grad_beta = float(self.lambda_current) * float(self.beta0)
        self.fpt_bias.set_kernel_grad_beta(self.kernel_grad_beta)

    @staticmethod
    def kinetic_temperature_stats(state) -> dict[str, float]:
        """Per-replica kinetic temperature (K) mean / std / min / max from MDState."""
        if not hasattr(state, "momenta") or state.momenta is None:
            return {}
        if hasattr(state, "calc_temperature"):
            t_per = state.calc_temperature()
        else:
            import torch_sim as ts

            t_per = ts.calc_temperature(
                masses=state.masses,
                momenta=state.momenta,
                system_idx=state.system_idx,
            )
        t = t_per.detach().reshape(-1).to(dtype=torch.float64)
        if t.numel() == 0:
            return {}
        return {
            "kinetic_temperature_mean": float(t.mean().item()),
            "kinetic_temperature_std": float(t.std(unbiased=False).item())
            if t.numel() > 1
            else 0.0,
            "kinetic_temperature_min": float(t.min().item()),
            "kinetic_temperature_max": float(t.max().item()),
        }

    def estimate_collective_variable_density(self, coordinate: torch.Tensor) -> dict[str, Any]:
        collective = self.collective_variable(coordinate).detach()
        bandwidth = self.gaussian_kernel.bandwidth.detach()
        kernel_matrix = self.gaussian_kernel.kernel_matrix(collective, bandwidth)
        row_sum = torch.sum(kernel_matrix, dim=-1)
        batch = int(collective.shape[0])
        log_density = torch.log(
            row_sum / float(batch) + self.log_density_epsilon
        )
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
        eta = (lambda_next - lambda_current) / (
            lambda_current * (1.0 + lambda_next)
        )
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

    def advance_lambda(
        self,
        state,
        collective_variable_values: np.ndarray,
        density_info: dict[str, Any],
        lambda_current: float,
        lambda_next: float,
    ):
        unnormalized, normalized = self.compute_reweight_weights(
            collective_variable_values,
            density_info,
            lambda_current,
            lambda_next,
        )
        self.lambda_current = float(lambda_next)
        self.sync_lambda_to_kernel_grad_beta()
        return state, unnormalized, normalized

    def run_stage(
        self,
        system,
        num_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
    ):
        """Run ``num_steps`` of torch-sim NVT Langevin with current λ."""
        import torch_sim as ts

        self.sync_lambda_to_kernel_grad_beta()
        num_steps = int(num_steps)
        # Chunk by update_pace so bandwidth can refresh between integrate calls
        # while preserving momenta (MDState passed back into integrate).
        remaining = num_steps
        state = system
        done = 0
        t0 = time.perf_counter()
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                total=num_steps,
                desc="torch_sim BAOAB-FP",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
            )

        while remaining > 0:
            chunk = min(self.update_pace, remaining)
            # Refresh bandwidth from current positions before the chunk.
            try:
                coords = sim_state_to_batched_coordinates(state)
                collective = self.collective_variable(coords)
                self.gaussian_kernel.update_bandwidth(collective.detach())
            except Exception:
                # First call may be a list[Atoms]; bandwidth updates inside FPTBias.
                pass

            state = ts.integrate(
                system=state,
                model=self.model,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=chunk,
                temperature=self.temperature,
                timestep=self.timestep,
                gamma=self.langevin_friction,
                pbar=False,
            )
            remaining -= chunk
            done += chunk
            self._last_step_metrics = dict(self.fpt_bias._last_metrics)
            self._last_step_metrics.update(self.kinetic_temperature_stats(state))
            self._last_step_metrics["target_temperature"] = float(self.temperature)
            if progress is not None:
                progress.update(chunk)
            if verbose and (done % verbose_pace == 0 or remaining == 0):
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

        if progress is not None:
            progress.close()
        elapsed = max(time.perf_counter() - t0, 1e-12)
        print(
            f"[TorchSimFokkerPlanck] stage done: {num_steps} steps in {elapsed:.2f}s "
            f"→ {num_steps / elapsed:.2f} it/s",
            flush=True,
        )
        return state

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
        max_runtime_sec: float | None = None,
    ):
        """Run optional λ-heal schedule, then ESS-based λ annealing.

        Heal phase (optional)
        ---------------------
        For each ``λ`` in ``heal_lambdas``, hard-set the FP strength and run
        ``heal_steps`` (default: ``stage_steps``) of dynamics. No ESS reweighting
        between heal λ values — this is intentional for a non-monotonic warmup
        schedule such as ``0.2, 0.1, 0.5, 0.02, 0.01``.

        Anneal phase
        ------------
        Starts at ``lambda_init`` and increases toward ``lambda_max`` using
        ESS-constrained reweighting (original behaviour).

        ``max_stages`` / ``max_runtime_sec`` stop the anneal loop early (heal
        time is included in the runtime budget).
        """
        state = system
        t_run0 = time.perf_counter()

        def _runtime_exceeded() -> bool:
            if max_runtime_sec is None:
                return False
            return (time.perf_counter() - t_run0) >= float(max_runtime_sec)

        def _call_stage_end(stage_i: int, lam: float, st, *, phase: str) -> None:
            if on_stage_end is None:
                return
            try:
                on_stage_end(stage_i, float(lam), st, phase=phase)
            except TypeError:
                on_stage_end(stage_i, float(lam), st)

        # --- Heal / warmup: prescribed λ sequence ---------------------------------
        if heal_lambdas:
            heal_list = [float(x) for x in heal_lambdas]
            n_heal = int(heal_steps) if heal_steps is not None else int(stage_steps)
            if n_heal <= 0:
                raise ValueError(f"heal_steps must be > 0, got {n_heal}")
            if verbose:
                print(
                    f"[TorchSimFokkerPlanck] heal: {len(heal_list)} λ values "
                    f"{heal_list} × {n_heal} steps each",
                    flush=True,
                )
            for heal_i, lam in enumerate(heal_list):
                if _runtime_exceeded():
                    if verbose:
                        print(
                            "[TorchSimFokkerPlanck] max_runtime_sec reached during heal",
                            flush=True,
                        )
                    return state
                self.lambda_current = float(lam)
                self.sync_lambda_to_kernel_grad_beta()
                if verbose:
                    print(
                        f"[TorchSimFokkerPlanck] heal {heal_i}: "
                        f"{n_heal} steps at lambda={self.lambda_current:.8g}",
                        flush=True,
                    )
                state = self.run_stage(
                    state,
                    n_heal,
                    verbose=verbose,
                    verbose_pace=verbose_pace,
                    show_progress=show_progress,
                )
                _call_stage_end(
                    heal_i, float(self.lambda_current), state, phase="heal"
                )

        # --- Anneal: ESS-driven λ increase from lambda_init ----------------------
        self.lambda_current = float(self.lambda_init)
        self.sync_lambda_to_kernel_grad_beta()
        stage_index = 0
        if verbose:
            print(
                f"[TorchSimFokkerPlanck] anneal: start λ={self.lambda_current:.8g} "
                f"→ max={self.lambda_max:.8g}, {stage_steps} steps/stage",
                flush=True,
            )
        while self.lambda_current < self.lambda_max - 1e-12:
            if max_stages is not None and stage_index >= int(max_stages):
                if verbose:
                    print(
                        f"[TorchSimFokkerPlanck] max_stages={max_stages} reached",
                        flush=True,
                    )
                break
            if _runtime_exceeded():
                if verbose:
                    print(
                        "[TorchSimFokkerPlanck] max_runtime_sec reached during anneal",
                        flush=True,
                    )
                break
            if verbose:
                print(
                    f"[TorchSimFokkerPlanck] anneal stage {stage_index}: "
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
            _call_stage_end(
                stage_index, float(self.lambda_current), state, phase="anneal"
            )
            coords = sim_state_to_batched_coordinates(state)
            density_info = self.estimate_collective_variable_density(coords)
            collective_np = (
                self.collective_variable(coords).detach().cpu().numpy()
            )
            lambda_k = float(self.lambda_current)
            lambda_next = float(
                self.choose_next_lambda(collective_np, density_info, lambda_k)
            )
            if lambda_next <= lambda_k + 1e-15:
                if verbose:
                    print(
                        f"[TorchSimFokkerPlanck] cannot increase lambda beyond {lambda_k}",
                        flush=True,
                    )
                break
            state_before = state
            state, unnormalized, normalized = self.advance_lambda(
                state, collective_np, density_info, lambda_k, lambda_next
            )
            if verbose:
                print(
                    f"[TorchSimFokkerPlanck] lambda {lambda_k:.8g} → {lambda_next:.8g} "
                    f"(ESS/N={self.compute_ess(unnormalized) / collective_np.shape[0]:.4g})",
                    flush=True,
                )
            if on_lambda_change is not None:
                on_lambda_change(
                    lambda_k,
                    lambda_next,
                    state,
                    density_info,
                    stage_index,
                    state_before_advance=state_before,
                    unnormalized_weights=unnormalized,
                    normalized_weights=normalized,
                )
            stage_index += 1
        return state

    def to_sampler_state(self, state) -> SamplerState:
        """Convert a torch-sim MD/SimState to ``SamplerState`` (coords only)."""
        coords = sim_state_to_batched_coordinates(state).to(
            device=self.device, dtype=self.dtype
        )
        momentum = getattr(state, "momenta", None)
        if momentum is not None:
            n_sys = int(state.n_systems)
            n_per = int(state.n_atoms) // n_sys
            momentum = momentum.view(n_sys, n_per, 3).to(
                device=self.device, dtype=self.dtype
            )
        else:
            momentum = torch.zeros_like(coords)
        return SamplerState(coordinate=coords, momentum=momentum)


# Re-export umbrella / well-tempered MetaD APIs (implemented in enhanced_sampling.py).
from .enhanced_sampling import (  # noqa: E402
    TorchSimUmbrellaSampling,
    TorchSimWelltemperMetadynamic,
    UmbrellaBias,
    WellTemperedMetadynamicsBias,
    make_umbrella_centers_1d,
    make_umbrella_centers_polyline,
)

__all__ = [
    "FPTBias",
    "TorchSimFokkerPlanckIntegrator",
    "TorchSimUmbrellaSampling",
    "TorchSimWelltemperMetadynamic",
    "UmbrellaBias",
    "WellTemperedMetadynamicsBias",
    "make_umbrella_centers_1d",
    "make_umbrella_centers_polyline",
]
