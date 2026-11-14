"""BAOAB + Fokker–Planck pullback with prescribed healing and fixed production λ."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional

from .protocol import validate_lambda, run_fixed_protocol

import numpy as np
import torch

from ..kernel.gaussian import GaussianKernel
from ..math.pullback import pullback_collective_variable_gradient
from ..system.energy_system import EnergySystem
from .sampler_state import SamplerState

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


class BaoabFokkerPlanckIntegrator:
    """BAOAB Langevin with FP Gaussian-kernel force and fixed production λ.

    Thermal bath uses fixed ``beta0``. FP pullback is scaled by
    ``kernel_grad_beta``:

    - ``"lambda"`` (default): ``λ · β0``  →  force ``F / λ``
    - ``"lambda_over_one_plus_lambda"``: ``λ/(1+λ) · β0``  →  force ``F(1+λ)/λ``

    """

    def __init__(
        self,
        energy_system: EnergySystem,
        collective_variable: Callable[[torch.Tensor], torch.Tensor],
        gaussian_kernel: GaussianKernel,
        *,
        timestep: float,
        langevin_friction: float,
        beta0: float,
        production_lambda: float,
        mass: float | torch.Tensor = 1.0,
        boltzmann_constant: float = 1.0,
        update_pace: int = 20,
        use_kernel_gradient: bool = True,
        collective_variable_grad_clip: float | None = None,
        log_density_epsilon: float = 1e-12,
        kernel_grad_beta_mode: str = "lambda",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.energy_system = energy_system
        self.collective_variable = collective_variable
        self.gaussian_kernel = gaussian_kernel
        self.timestep = float(timestep)
        self.langevin_friction = float(langevin_friction)
        self.beta0 = float(beta0)
        self.boltzmann_constant = float(boltzmann_constant)
        self.production_lambda = validate_lambda(production_lambda)
        self.update_pace = max(1, int(update_pace))
        self.use_kernel_gradient = bool(use_kernel_gradient)
        self.collective_variable_grad_clip = collective_variable_grad_clip
        self.log_density_epsilon = float(log_density_epsilon)
        self.device = torch.device(device or energy_system.coordinate.device)
        self.dtype = dtype

        mass_tensor = torch.as_tensor(mass, dtype=dtype, device=self.device)
        self.mass = mass_tensor
        self.inverse_mass = 1.0 / mass_tensor

        mode = str(kernel_grad_beta_mode).strip().lower()
        aliases = {
            "lambda": "lambda",
            "lam": "lambda",
            "f_over_lambda": "lambda",
            "lambda_over_one_plus_lambda": "lambda_over_one_plus_lambda",
            "lam_over_1p_lam": "lambda_over_one_plus_lambda",
            "f_one_plus_lambda_over_lambda": "lambda_over_one_plus_lambda",
        }
        if mode not in aliases:
            raise ValueError(
                "kernel_grad_beta_mode must be 'lambda' or "
                f"'lambda_over_one_plus_lambda', got {kernel_grad_beta_mode!r}"
            )
        self.kernel_grad_beta_mode = aliases[mode]

        self.lambda_current = self.production_lambda
        self.kernel_grad_beta = 0.0
        self.sync_lambda_to_kernel_grad_beta()
        self._density_cache: dict[str, Any] = {}
        self._last_step_metrics: dict[str, float] = {}
        self._last_cv_grad_stats: dict[str, float] = {}

    def sync_lambda_to_kernel_grad_beta(self) -> None:
        lam = float(self.lambda_current)
        beta0 = float(self.beta0)
        if self.kernel_grad_beta_mode == "lambda_over_one_plus_lambda":
            self.kernel_grad_beta = (lam / (1.0 + lam)) * beta0
        else:
            self.kernel_grad_beta = lam * beta0

    def sample_momentum_maxwell_boltzmann(
        self, coordinate: torch.Tensor, *, seed: int | None = None
    ) -> torch.Tensor:
        """``p ~ N(0, m / β0)`` at the thermal bath temperature."""
        kB_T = 1.0 / self.beta0
        mass_local = self.mass
        while mass_local.ndim < coordinate.ndim:
            mass_local = mass_local.unsqueeze(0)
        mass_local = mass_local.to(device=coordinate.device, dtype=coordinate.dtype)
        scale = torch.sqrt(torch.clamp(kB_T * mass_local, min=0.0))
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            noise = torch.randn(
                coordinate.shape,
                dtype=coordinate.dtype,
                generator=generator,
            ).to(device=coordinate.device)
        else:
            noise = torch.randn(
                coordinate.shape,
                dtype=coordinate.dtype,
                device=coordinate.device,
            )
        return scale * noise

    def init_state(
        self,
        coordinate: torch.Tensor | None = None,
        *,
        seed: int = 0,
        momentum: torch.Tensor | None = None,
    ) -> SamplerState:
        coordinate = (
            self.energy_system.coordinate if coordinate is None else coordinate
        ).to(device=self.device, dtype=self.dtype)
        if momentum is None:
            momentum = self.sample_momentum_maxwell_boltzmann(
                coordinate, seed=int(seed)
            )
        else:
            momentum = momentum.to(device=self.device, dtype=self.dtype)
        try:
            generator = torch.Generator(device=str(self.device))
        except TypeError:
            generator = torch.Generator()
        generator.manual_seed(int(seed) + 1)
        self.energy_system.coordinate = coordinate
        return SamplerState(
            coordinate=coordinate, momentum=momentum, generator=generator
        )

    def _apply_inverse_mass(self, momentum: torch.Tensor) -> torch.Tensor:
        inverse = self.inverse_mass
        while inverse.ndim < momentum.ndim:
            inverse = inverse.unsqueeze(0)
        return momentum * inverse

    def _clip_collective_variable_gradient(
        self, gradient: torch.Tensor
    ) -> torch.Tensor:
        batch, atoms, _dim = gradient.shape
        denom = float(batch * atoms)
        norm = torch.linalg.norm(gradient) / denom
        max_norm = self.collective_variable_grad_clip
        stats: dict[str, float] = {
            "cv_grad_norm": float(norm.item()),
            "cv_grad_clip": (float(max_norm) if max_norm is not None else float("nan")),
        }
        if max_norm is None:
            stats["cv_grad_norm_post"] = stats["cv_grad_norm"]
            stats["cv_grad_scale"] = 1.0
            self._last_cv_grad_stats = stats
            return gradient
        scale = torch.clamp(
            torch.tensor(max_norm, device=gradient.device, dtype=gradient.dtype)
            / (norm + 1e-12),
            max=1.0,
        )
        out = gradient * scale
        stats["cv_grad_scale"] = float(scale.item())
        stats["cv_grad_norm_post"] = float((torch.linalg.norm(out) / denom).item())
        self._last_cv_grad_stats = stats
        return out

    def fokker_planck_gradient(
        self,
        coordinate: torch.Tensor,
        *,
        beta: float,
        kernel_grad_beta: float,
    ) -> torch.Tensor:
        """``G = ∇U/β + pullback(∇_s log-kernel) / kernel_grad_beta``."""
        system_grad_times_beta = self.energy_system.potential_gradient(coordinate, beta)
        base = system_grad_times_beta / beta
        if not self.use_kernel_gradient:
            return base
        collective = self.collective_variable(coordinate)
        kernel_grad = self.gaussian_kernel.kernel_gradient(
            collective, self.gaussian_kernel.bandwidth
        )
        kernel_grad = self._clip_collective_variable_gradient(kernel_grad)
        pullback = pullback_collective_variable_gradient(
            coordinate, self.collective_variable, kernel_grad
        )
        return base + pullback / float(kernel_grad_beta)

    def kinetic_temperature(self, momentum: torch.Tensor) -> torch.Tensor:
        """Per-replica kinetic temperature after OU: ``T = 2 K / (f k_B)``."""
        inverse = self.inverse_mass
        while inverse.ndim < momentum.ndim:
            inverse = inverse.unsqueeze(0)
        kinetic_energy = 0.5 * torch.sum((momentum**2) * inverse, dim=(1, 2))
        spatial_dim = float(momentum.shape[-1])
        num_atoms = float(momentum.shape[-2])
        degrees_of_freedom = spatial_dim * num_atoms
        return (2.0 * kinetic_energy) / (degrees_of_freedom * self.boltzmann_constant)

    def step(
        self,
        state: SamplerState,
        *,
        step_index: int = 0,
        ou_noise: torch.Tensor | None = None,
        update_bandwidth: bool | None = None,
    ) -> SamplerState:
        """One BAOAB step: B(dt/2) → A(dt/2) → O(dt) → A(dt/2) → B(dt/2).

        ``ou_noise`` (optional) injects a fixed ``N(0,1)`` tensor for the O-step,
        used by JAX↔torch parity smoke tests. When omitted, fresh noise is drawn.

        ``update_bandwidth`` overrides the usual ``step_index % update_pace`` gate
        when not ``None`` (parity tests freeze bandwidth with ``False``).
        """
        do_bw = (
            bool(update_bandwidth)
            if update_bandwidth is not None
            else (step_index % self.update_pace == 0)
        )
        if do_bw:
            collective = self.collective_variable(state.coordinate)
            self.gaussian_kernel.update_bandwidth(collective)

        self._last_cv_grad_stats = {}

        dt = self.timestep
        gamma = self.langevin_friction
        beta = self.beta0
        kB_T = 1.0 / beta
        kgb = self.kernel_grad_beta

        g0 = self.fokker_planck_gradient(
            state.coordinate, beta=beta, kernel_grad_beta=kgb
        )
        force0 = -g0
        momentum_quarter = state.momentum + 0.5 * dt * force0
        coordinate_half = state.coordinate + 0.5 * dt * self._apply_inverse_mass(
            momentum_quarter
        )

        exp_decay = float(np.exp(-gamma * dt))
        ou_variance = kB_T * (1.0 - float(np.exp(-2.0 * gamma * dt)))
        mass_local = 1.0 / self.inverse_mass
        while mass_local.ndim < state.momentum.ndim:
            mass_local = mass_local.unsqueeze(0)
        scale = torch.sqrt(torch.clamp(ou_variance * mass_local, min=0.0))
        if ou_noise is None:
            noise = torch.randn(
                state.momentum.shape,
                dtype=state.momentum.dtype,
                device=state.momentum.device,
            )
        else:
            noise = ou_noise.to(
                device=state.momentum.device, dtype=state.momentum.dtype
            )
            if noise.shape != state.momentum.shape:
                raise ValueError(
                    f"ou_noise shape {tuple(noise.shape)} != "
                    f"momentum {tuple(state.momentum.shape)}"
                )
        momentum_three_quarter = exp_decay * momentum_quarter + scale * noise

        coordinate_next = coordinate_half + 0.5 * dt * self._apply_inverse_mass(
            momentum_three_quarter
        )
        g1 = self.fokker_planck_gradient(
            coordinate_next, beta=beta, kernel_grad_beta=kgb
        )
        force1 = -g1
        momentum_next = momentum_three_quarter + 0.5 * dt * force1

        temperature = self.kinetic_temperature(momentum_three_quarter)
        bandwidth = self.gaussian_kernel.bandwidth.detach()
        self._last_step_metrics = {
            "kinetic_temperature_mean": float(temperature.mean().item()),
            "kinetic_temperature_std": float(temperature.std(unbiased=False).item()),
            "bandwidth_min": float(bandwidth.min().item()),
            "bandwidth_max": float(bandwidth.max().item()),
            "bandwidth_median": float(bandwidth.median().item()),
            "target_temperature": float(1.0 / (self.beta0 * self.boltzmann_constant)),
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
                self._last_step_metrics[key] = float(rs[key])
        if "target_mode" in rs:
            self._last_step_metrics["target_mode"] = rs["target_mode"]
        for key, val in self._last_cv_grad_stats.items():
            self._last_step_metrics[key] = val

        self.energy_system.coordinate = coordinate_next
        return SamplerState(
            coordinate=coordinate_next,
            momentum=momentum_next,
            generator=state.generator,
        )

    def run_stage(
        self,
        num_steps: int,
        *,
        initial_state: SamplerState | None = None,
        seed: int = 0,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
    ) -> SamplerState:
        self.sync_lambda_to_kernel_grad_beta()
        state = initial_state or self.init_state(seed=seed)
        num_steps = int(num_steps)
        iterator = range(num_steps)
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                iterator,
                total=num_steps,
                desc="BAOAB",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            iterator = progress
        elif show_progress:
            print(
                "[BaoabFokkerPlanck] tqdm not installed; "
                "speed will be reported every verbose_pace steps",
                flush=True,
            )

        t0 = time.perf_counter()
        for step_index in iterator:
            state = self.step(state, step_index=step_index)
            done = step_index + 1
            if verbose and done % verbose_pace == 0:
                elapsed = max(time.perf_counter() - t0, 1e-12)
                speed = done / elapsed
                metrics = self._last_step_metrics
                msg = (
                    f"BAOAB Step {done}, "
                    f"lambda={self.lambda_current:.6g}, "
                    f"kernel_grad_beta={self.kernel_grad_beta:.6g}, "
                    f"T_kin (from KE): "
                    f"{metrics.get('kinetic_temperature_mean', float('nan')):.4f} K, "
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
                    + (
                        (
                            f", cv_grad={metrics['cv_grad_norm']:.4g}"
                            + (
                                f"→{metrics['cv_grad_norm_post']:.4g}"
                                if metrics.get("cv_grad_scale", 1.0) < 1.0 - 1e-12
                                else ""
                            )
                            + (
                                f" clip={metrics['cv_grad_clip']:.4g}"
                                if np.isfinite(
                                    metrics.get("cv_grad_clip", float("nan"))
                                )
                                else " clip=off"
                            )
                            + f" scale={metrics.get('cv_grad_scale', 1.0):.3g}"
                        )
                        if "cv_grad_norm" in metrics
                        else ""
                    )
                    + f", {speed:.2f} it/s"
                )
                if progress is not None:
                    progress.write(msg)
                    progress.set_postfix_str(f"{speed:.2f} it/s")
                else:
                    print(msg, flush=True)

        elapsed = max(time.perf_counter() - t0, 1e-12)
        speed = num_steps / elapsed
        if progress is not None:
            progress.close()
        print(
            f"[BaoabFokkerPlanck] stage done: {num_steps} steps in {elapsed:.2f}s "
            f"→ {speed:.2f} it/s",
            flush=True,
        )
        return state

    def estimate_collective_variable_density(
        self, coordinate: torch.Tensor
    ) -> dict[str, Any]:
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

    def run(
        self,
        production_steps: int,
        *,
        initial_state=None,
        seed=0,
        dump_every=100,
        heal_lambdas=None,
        heal_steps=None,
        on_frame=None,
        verbose=False,
        verbose_pace=100,
        show_progress=True,
    ):
        """Prescribed healing followed by fixed-lambda production; no reweighting."""
        state = (
            initial_state if initial_state is not None else self.init_state(seed=seed)
        )

        def advance(st, steps):
            return self.run_stage(
                steps,
                initial_state=st,
                verbose=verbose,
                verbose_pace=verbose_pace,
                show_progress=show_progress,
            )

        return run_fixed_protocol(
            self,
            state,
            advance,
            production_steps,
            dump_every,
            heal_lambdas,
            heal_steps,
            on_frame,
        )
