"""Umbrella sampling and well-tempered metadynamics (torch-sim ModelInterface).

``SumModel(mace, bias[, wall])`` with NVT Langevin. Metadynamics supports
multi-walker replicas that **share** one bias potential (hills deposited from
all walkers into the same ``V(s)``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from ..math.bspline import periodic_bspline_eval, periodic_cubic_bspline_coeffs
from ..potential.mace import sim_state_to_batched_coordinates

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:
    from torch_sim.models.interface import ModelInterface as _ModelInterface
except ImportError:  # pragma: no cover
    _ModelInterface = torch.nn.Module  # type: ignore[misc, assignment]

KB_EV_PER_K = 8.617330337217213e-5


def _cv_matrix(collective: torch.Tensor) -> torch.Tensor:
    """Normalize CV tensor to ``(B, n_cv)``."""
    if collective.ndim == 3:
        return collective.squeeze(-1)
    if collective.ndim == 2:
        return collective
    raise ValueError(f"unexpected CV shape {tuple(collective.shape)}")


def _n_atoms_per_system(state, n_atoms_per_system: int | None) -> int:
    n_systems = int(state.n_systems)
    n_atoms = int(state.n_atoms)
    if n_atoms_per_system is not None:
        n_per = int(n_atoms_per_system)
        if n_per * n_systems != n_atoms:
            raise ValueError(
                f"n_atoms_per_system={n_per} incompatible with "
                f"n_atoms={n_atoms}, n_systems={n_systems}"
            )
        return n_per
    if n_atoms % n_systems != 0:
        raise ValueError("unequal atoms per system in SimState")
    return n_atoms // n_systems


class _BiasModelBase(_ModelInterface):
    """Shared ModelInterface plumbing for CV biases."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        n_atoms_per_system: int | None = None,
    ):
        if _ModelInterface is torch.nn.Module:
            raise ImportError(
                "torch_sim is required. Activate conda env MACE_torchsim."
            )
        super().__init__()
        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"
        self.n_atoms_per_system = n_atoms_per_system
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
            raise NotImplementedError(f"{type(self).__name__} does not compute stress")
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


class UmbrellaBias(_BiasModelBase):
    """Harmonic umbrella on a CV: ``U = ½ Σ_α κ_α (s_α − s0_α)²``.

    Parallel multi-window US: each replica ``i`` has its own center ``s0[i]``
    (shape ``(B, n_cv)``). All windows advance in one batched NVT step.

    For periodic CVs (φ/ψ), pass ``period`` (scalar or per-CV); deviations use
    minimum-image wrapping on the torus.
    """

    def __init__(
        self,
        collective_variable: Callable[[torch.Tensor], torch.Tensor],
        centers: torch.Tensor | np.ndarray,
        *,
        kappa: float | Sequence[float] | torch.Tensor = 50.0,
        period: float | Sequence[float] | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        n_atoms_per_system: int | None = None,
    ):
        super().__init__(
            device=device, dtype=dtype, n_atoms_per_system=n_atoms_per_system
        )
        self.collective_variable = collective_variable
        centers_t = torch.as_tensor(centers, device=self._device, dtype=self._dtype)
        if centers_t.ndim == 1:
            centers_t = centers_t.unsqueeze(0)
        if centers_t.ndim != 2:
            raise ValueError(f"centers must be (B,n_cv) or (n_cv,), got {centers_t.shape}")
        self.centers = centers_t  # (B, n_cv)
        self.n_windows = int(centers_t.shape[0])
        self.n_cv = int(centers_t.shape[1])
        kap = torch.as_tensor(kappa, device=self._device, dtype=self._dtype)
        # kappa: scalar | (n_cv,) | (B, n_cv)
        if kap.ndim == 0:
            kap = kap.expand(self.n_cv).clone()
        if kap.ndim == 1:
            if int(kap.numel()) == 1:
                kap = kap.reshape(()).expand(self.n_cv).clone()
            if int(kap.numel()) != self.n_cv:
                raise ValueError(
                    f"kappa length {kap.numel()} != n_cv {self.n_cv}"
                )
            kap = kap.unsqueeze(0).expand(self.n_windows, -1).clone()
        elif kap.ndim == 2:
            if kap.shape != (self.n_windows, self.n_cv):
                raise ValueError(
                    f"kappa shape {tuple(kap.shape)} != "
                    f"({self.n_windows}, {self.n_cv})"
                )
        else:
            raise ValueError(f"kappa must be scalar/(n_cv,)/(B,n_cv), got {kap.shape}")
        self.kappa = kap  # (B, n_cv)
        if period is None:
            self.period = None
        else:
            per = torch.as_tensor(period, device=self._device, dtype=self._dtype)
            if per.ndim == 0:
                per = per.expand(self.n_cv).clone()
            elif int(per.numel()) == 1:
                per = per.reshape(()).expand(self.n_cv).clone()
            if int(per.numel()) != self.n_cv:
                raise ValueError(f"period length {per.numel()} != n_cv {self.n_cv}")
            if bool((per <= 0).any().item()):
                raise ValueError("period components must be > 0")
            self.period = per
        self._last_cv: torch.Tensor | None = None

    def set_centers(self, centers: torch.Tensor | np.ndarray) -> None:
        c = torch.as_tensor(centers, device=self._device, dtype=self._dtype)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        if c.shape != self.centers.shape:
            raise ValueError(f"centers shape {c.shape} != {self.centers.shape}")
        self.centers = c

    def _wrap_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Minimum-image ``s − s0`` when ``period`` is set."""
        if self.period is None:
            return delta
        p = self.period.view(*([1] * (delta.ndim - 1)), -1)
        return delta - p * torch.round(delta / p)

    def cv_energy(self, s: torch.Tensor) -> torch.Tensor:
        """``s`` shaped ``(B, n_cv)`` → energy ``(B,)``."""
        B = s.shape[0]
        centers = self.centers
        kappa = self.kappa
        if centers.shape[0] == 1 and B > 1:
            centers = centers.expand(B, -1)
            kappa = kappa.expand(B, -1)
        elif centers.shape[0] != B:
            raise ValueError(
                f"centers B={centers.shape[0]} incompatible with CV B={B}"
            )
        delta = self._wrap_delta(s - centers)
        return 0.5 * torch.sum(kappa * delta * delta, dim=-1)

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_per = _n_atoms_per_system(state, self.n_atoms_per_system)
        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        coordinate = coordinate.detach().requires_grad_(True)
        s = _cv_matrix(self.collective_variable(coordinate))
        energy = self.cv_energy(s)
        (grad_x,) = torch.autograd.grad(
            energy.sum(), coordinate, create_graph=False, retain_graph=False
        )
        forces = (-grad_x).reshape(n_systems * n_per, 3).detach()
        energy_out = energy.detach()
        s_det = s.detach()
        self._last_cv = s_det
        centers = self.centers
        if centers.shape[0] == 1 and n_systems > 1:
            centers = centers.expand(n_systems, -1)
        delta = self._wrap_delta(s_det - centers)
        rms_dev = torch.sqrt(torch.mean(delta * delta, dim=-1))
        self._last_metrics = {
            "umbrella_energy_mean": float(energy_out.mean().item()),
            "umbrella_energy_max": float(energy_out.max().item()),
            "umbrella_rms_dev_mean": float(rms_dev.mean().item()),
            "umbrella_rms_dev_max": float(rms_dev.max().item()),
            "force_bias_rms": float(torch.sqrt(torch.mean(forces * forces)).item()),
            "n_windows": float(n_systems),
        }
        return {"energy": energy_out, "forces": forces}


def make_umbrella_centers_polyline(
    waypoints: Sequence[Sequence[float]],
    n_windows: int,
) -> np.ndarray:
    """Evenly space ``n_windows`` centers along a polyline in CV space.

    ``waypoints``: list of CV points, e.g. L-path
    ``[(1.5, 5.0), (1.5, 1.5), (4.0, 1.5)]``.
    Returns ``(n_windows, n_cv)``.
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("need >=2 waypoints shaped (N, n_cv)")
    n_windows = int(n_windows)
    if n_windows < 2:
        raise ValueError("n_windows must be >= 2")
    seg = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    total = float(np.sum(lengths))
    if total <= 0:
        raise ValueError("polyline has zero length")
    targets = np.linspace(0.0, total, n_windows)
    out = np.zeros((n_windows, pts.shape[1]), dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    for i, t in enumerate(targets):
        if t >= total:
            out[i] = pts[-1]
            continue
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(lengths) - 1)
        frac = (t - cum[j]) / lengths[j] if lengths[j] > 0 else 0.0
        out[i] = pts[j] + frac * seg[j]
    return out


def make_umbrella_centers_1d(
    lo: float,
    hi: float,
    n_windows: int,
    *,
    fixed: dict[int, float] | None = None,
    n_cv: int = 2,
    axis: int = 0,
) -> np.ndarray:
    """Centers along one CV axis; other axes taken from ``fixed`` (default 0)."""
    n_windows = int(n_windows)
    centers = np.zeros((n_windows, int(n_cv)), dtype=np.float64)
    if fixed:
        for ax, val in fixed.items():
            centers[:, int(ax)] = float(val)
    centers[:, int(axis)] = np.linspace(float(lo), float(hi), n_windows)
    return centers



class WellTemperedMetadynamicsBias(_BiasModelBase):
    """Well-tempered MetaD with **shared** hills across walkers.

    Bias in CV space::

        V(s) = Σ_h w_h exp( −½ Σ_α ((s_α − c_{h,α}) / σ_α)² )

    Deposit (every ``deposit`` call, typically once per ``deposit_pace`` MD steps)
    one Gaussian per walker into the same hill list::

        w = w0 · exp( −V(s) / (k_B ΔT) ),  ΔT = T (γ − 1)

    where ``γ`` is ``bias_factor``. Metal units: energy in eV, CV dimensionless.

    For periodic CVs (e.g. backbone φ/ψ), pass ``period`` (scalar or per-CV);
    Gaussian distances then use the minimum-image convention on the torus.

    If ``grid_n`` is set (int or per-CV), hills are splatted onto a periodic
    grid and ``V(s)`` / forces are cubic B-spline interpolants of that grid.
    Evaluation is then **independent of the number of Gaussians** (needed for
    3D MetaD, where ``n_hills`` otherwise grows as walkers × steps / pace).
    """

    def __init__(
        self,
        collective_variable: Callable[[torch.Tensor], torch.Tensor],
        *,
        n_cv: int = 2,
        sigma: float | Sequence[float] = 0.1,
        height: float = 0.02,
        bias_factor: float = 10.0,
        temperature: float = 300.0,
        kB: float = KB_EV_PER_K,
        max_hills: int = 200_000,
        period: float | Sequence[float] | None = None,
        grid_n: int | Sequence[int] | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        n_atoms_per_system: int | None = None,
    ):
        super().__init__(
            device=device, dtype=dtype, n_atoms_per_system=n_atoms_per_system
        )
        self.collective_variable = collective_variable
        self.n_cv = int(n_cv)
        sig = torch.as_tensor(sigma, device=self._device, dtype=self._dtype).reshape(-1)
        if sig.numel() == 1:
            sig = sig.expand(self.n_cv).clone()
        if int(sig.numel()) != self.n_cv:
            raise ValueError(f"sigma length {sig.numel()} != n_cv {self.n_cv}")
        self.sigma = sig
        if period is None:
            self.period = None
        else:
            per = torch.as_tensor(
                period, device=self._device, dtype=self._dtype
            ).reshape(-1)
            if per.numel() == 1:
                per = per.expand(self.n_cv).clone()
            if int(per.numel()) != self.n_cv:
                raise ValueError(f"period length {per.numel()} != n_cv {self.n_cv}")
            if bool((per <= 0).any()):
                raise ValueError("period components must be > 0")
            self.period = per
        self.height0 = float(height)
        self.bias_factor = float(bias_factor)
        if self.bias_factor <= 1.0:
            raise ValueError("bias_factor (γ) must be > 1 for well-tempered MetaD")
        self.temperature = float(temperature)
        self.kB = float(kB)
        # max_hills <= 0 → unlimited (buffers grow on demand)
        self.max_hills = int(max_hills)
        init_cap = 65_536 if self.max_hills <= 0 else self.max_hills
        self.register_buffer(
            "_centers",
            torch.zeros(init_cap, self.n_cv, device=self._device, dtype=self._dtype),
        )
        self.register_buffer(
            "_heights",
            torch.zeros(init_cap, device=self._device, dtype=self._dtype),
        )
        self.n_hills = 0
        self._deposit_count = 0
        self._init_bias_grid(grid_n)

    def _init_bias_grid(self, grid_n: int | Sequence[int] | None) -> None:
        """Optional periodic cubic B-spline grid for O(1) ``V(s)`` evaluation."""
        if grid_n is None:
            self._use_grid = False
            return
        if isinstance(grid_n, (int, float)):
            if int(grid_n) <= 0:
                self._use_grid = False
                return
            gn = [int(grid_n)] * self.n_cv
        else:
            gn = [int(x) for x in grid_n]
            if len(gn) == 1:
                gn = gn * self.n_cv
        if len(gn) != self.n_cv:
            raise ValueError(f"grid_n length {len(gn)} != n_cv {self.n_cv}")
        if any(n < 8 for n in gn):
            raise ValueError("grid_n must be >= 8 per CV for cubic B-spline")
        if self.period is None:
            raise ValueError("B-spline grid bias requires periodic CVs (period=...)")
        if self.n_cv > 3:
            raise ValueError("B-spline grid bias supports n_cv <= 3")
        self._use_grid = True
        self._grid_shape = tuple(gn)
        origin = -0.5 * self.period
        dx = self.period / torch.as_tensor(
            gn, device=self._device, dtype=self._dtype
        )
        axes = [
            origin[d] + torch.arange(gn[d], device=self._device, dtype=self._dtype) * dx[d]
            for d in range(self.n_cv)
        ]
        self.register_buffer("_grid_origin", origin)
        self.register_buffer("_grid_dx", dx)
        for d, ax in enumerate(axes):
            self.register_buffer(f"_grid_ax{d}", ax)
        zeros = torch.zeros(self._grid_shape, device=self._device, dtype=self._dtype)
        self.register_buffer("_grid", zeros)
        self.register_buffer("_coeffs", zeros.clone())

    @property
    def use_grid(self) -> bool:
        return bool(getattr(self, "_use_grid", False))

    def grid_axes(self) -> list[torch.Tensor]:
        if not self.use_grid:
            raise RuntimeError("bias grid is not enabled")
        return [getattr(self, f"_grid_ax{d}") for d in range(self.n_cv)]

    def _wrap_cv(self, diff: torch.Tensor, dim: int) -> torch.Tensor:
        p = self.period[dim]
        return diff - p * torch.round(diff / p)

    def _add_gaussians_to_grid(
        self, centers: torch.Tensor, heights: torch.Tensor
    ) -> None:
        """Splat Gaussians onto nodal samples (min-image, no truncation)."""
        if not self.use_grid:
            return
        B = int(centers.shape[0])
        if B == 0:
            return
        sig = self.sigma
        axes = self.grid_axes()
        n_elem = int(self._grid.numel())
        max_bytes = 64 * 1024 * 1024
        chunk = max(1, min(B, max_bytes // max(n_elem * int(self._grid.element_size()), 1)))
        nd = self.n_cv
        if nd == 1:
            a0 = axes[0]
            n0 = int(a0.numel())
            for i0 in range(0, B, chunk):
                c = centers[i0 : i0 + chunk]
                h = heights[i0 : i0 + chunk]
                b = int(c.shape[0])
                d0 = self._wrap_cv(a0.view(1, n0) - c[:, 0].view(b, 1), 0)
                g = h.view(b, 1) * torch.exp(-0.5 * (d0 / sig[0]) ** 2)
                self._grid.add_(g.sum(dim=0))
        elif nd == 2:
            a0, a1 = axes
            n0, n1 = int(a0.numel()), int(a1.numel())
            for i0 in range(0, B, chunk):
                c = centers[i0 : i0 + chunk]
                h = heights[i0 : i0 + chunk]
                b = int(c.shape[0])
                d0 = self._wrap_cv(a0.view(1, n0, 1) - c[:, 0].view(b, 1, 1), 0)
                d1 = self._wrap_cv(a1.view(1, 1, n1) - c[:, 1].view(b, 1, 1), 1)
                g = h.view(b, 1, 1) * torch.exp(
                    -0.5 * ((d0 / sig[0]) ** 2 + (d1 / sig[1]) ** 2)
                )
                self._grid.add_(g.sum(dim=0))
        elif nd == 3:
            a0, a1, a2 = axes
            n0, n1, n2 = int(a0.numel()), int(a1.numel()), int(a2.numel())
            for i0 in range(0, B, chunk):
                c = centers[i0 : i0 + chunk]
                h = heights[i0 : i0 + chunk]
                b = int(c.shape[0])
                d0 = self._wrap_cv(a0.view(1, n0, 1, 1) - c[:, 0].view(b, 1, 1, 1), 0)
                d1 = self._wrap_cv(a1.view(1, 1, n1, 1) - c[:, 1].view(b, 1, 1, 1), 1)
                d2 = self._wrap_cv(a2.view(1, 1, 1, n2) - c[:, 2].view(b, 1, 1, 1), 2)
                g = h.view(b, 1, 1, 1) * torch.exp(
                    -0.5
                    * (
                        (d0 / sig[0]) ** 2
                        + (d1 / sig[1]) ** 2
                        + (d2 / sig[2]) ** 2
                    )
                )
                self._grid.add_(g.sum(dim=0))
        else:
            raise ValueError(f"unsupported n_cv={nd} for grid splat")

    def _sync_bspline_coeffs(self) -> None:
        if not self.use_grid:
            return
        self._coeffs.copy_(periodic_cubic_bspline_coeffs(self._grid))

    def _rebuild_grid_from_hills(self, *, chunk: int = 64) -> None:
        if not self.use_grid:
            return
        self._grid.zero_()
        if self.n_hills == 0:
            self._coeffs.zero_()
            return
        for i0 in range(0, self.n_hills, int(chunk)):
            sl = slice(i0, min(self.n_hills, i0 + int(chunk)))
            self._add_gaussians_to_grid(self._centers[sl], self._heights[sl])
        self._sync_bspline_coeffs()

    def _cv_diff(self, s: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """``(B, n_cv)`` vs ``(H, n_cv)`` → ``(B, H, n_cv)`` (optionally wrapped)."""
        diff = s.unsqueeze(1) - centers.unsqueeze(0)
        if self.period is not None:
            p = self.period.view(1, 1, -1)
            diff = diff - p * torch.round(diff / p)
        return diff

    @property
    def delta_T(self) -> float:
        return self.temperature * (self.bias_factor - 1.0)

    def _ensure_capacity(self, need: int) -> None:
        """Grow hill buffers if needed. ``max_hills <= 0`` means unlimited."""
        need = int(need)
        cap = int(self._centers.shape[0])
        if need <= cap:
            return
        if self.max_hills > 0 and need > self.max_hills:
            raise RuntimeError(
                f"max_hills={self.max_hills} exceeded (need={need}, n_hills={self.n_hills})"
            )
        new_cap = max(need, max(cap * 2, 65_536))
        if self.max_hills > 0:
            new_cap = min(new_cap, self.max_hills)
        new_c = torch.zeros(
            new_cap, self.n_cv, device=self._device, dtype=self._dtype
        )
        new_h = torch.zeros(new_cap, device=self._device, dtype=self._dtype)
        if self.n_hills > 0:
            new_c[: self.n_hills] = self._centers[: self.n_hills]
            new_h[: self.n_hills] = self._heights[: self.n_hills]
        self.register_buffer("_centers", new_c)
        self.register_buffer("_heights", new_h)

    def load_hills(self, path: str | Path) -> int:
        """Load centers/heights from a ``hills.npz`` (resume MetaD)."""
        z = np.load(Path(path))
        if "centers" not in z.files or "heights" not in z.files:
            raise ValueError(f"{path} missing centers/heights")
        n = (
            int(z["n_hills"])
            if "n_hills" in z.files
            else int(np.asarray(z["heights"]).reshape(-1).shape[0])
        )
        centers = np.asarray(z["centers"][:n], dtype=np.float64)
        heights = np.asarray(z["heights"][:n], dtype=np.float64).reshape(-1)
        if centers.ndim != 2 or centers.shape[1] != self.n_cv:
            raise ValueError(
                f"hills centers shape {centers.shape} incompatible with n_cv={self.n_cv}"
            )
        if int(heights.shape[0]) != n:
            raise ValueError("centers/heights length mismatch")
        self._ensure_capacity(n)
        self._centers[:n] = torch.as_tensor(
            centers, device=self._device, dtype=self._dtype
        )
        self._heights[:n] = torch.as_tensor(
            heights, device=self._device, dtype=self._dtype
        )
        self.n_hills = n
        self._rebuild_grid_from_hills()
        return n

    def bias_potential_cv(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate shared ``V(s)`` for ``s`` shaped ``(B, n_cv)``."""
        B = s.shape[0]
        if self.n_hills == 0:
            return torch.zeros(B, device=self._device, dtype=self._dtype)
        if self.use_grid:
            return periodic_bspline_eval(
                self._coeffs,
                s,
                origin=self._grid_origin,
                period=self.period,
            )
        centers = self._centers[: self.n_hills]  # (H, n_cv)
        heights = self._heights[: self.n_hills]  # (H,)
        n_h = int(centers.shape[0])
        # Keep (B, Hchunk, n_cv) from exploding on FES dumps (R=256 → 5e5 hills).
        max_elems = 8_000_000
        chunk = max(1, min(n_h, max_elems // max(B * self.n_cv, 1)))
        v = torch.zeros(B, device=self._device, dtype=self._dtype)
        sig = self.sigma.view(1, 1, -1)
        for i0 in range(0, n_h, chunk):
            c = centers[i0 : i0 + chunk]
            h = heights[i0 : i0 + chunk]
            diff = self._cv_diff(s, c) / sig
            gauss = torch.exp(-0.5 * (diff * diff).sum(dim=-1))
            v = v + (h.unsqueeze(0) * gauss).sum(dim=-1)
        return v

    def deposit(self, coordinate: torch.Tensor) -> dict[str, float]:
        """Deposit one WT hill per replica into the shared bias."""
        with torch.no_grad():
            s = _cv_matrix(self.collective_variable(coordinate)).detach()
            if s.shape[-1] != self.n_cv:
                raise ValueError(f"CV dim {s.shape[-1]} != n_cv {self.n_cv}")
            ok = torch.isfinite(s).all(dim=-1)
            if not bool(ok.any()):
                return {
                    "n_hills": float(self.n_hills),
                    "deposit_height_mean": float("nan"),
                    "deposit_height_min": float("nan"),
                    "bias_energy_at_deposit_mean": float("nan"),
                    "deposit_skipped_nan": float(s.shape[0]),
                }
            if not bool(ok.all()):
                s = s[ok]
            V = self.bias_potential_cv(s)
            kb_dT = self.kB * self.delta_T
            w = self.height0 * torch.exp(-V / kb_dT)
            B = int(s.shape[0])
            self._ensure_capacity(self.n_hills + B)
            sl = slice(self.n_hills, self.n_hills + B)
            self._centers[sl] = s.to(device=self._device, dtype=self._dtype)
            self._heights[sl] = w.to(device=self._device, dtype=self._dtype)
            self.n_hills += B
            self._deposit_count += 1
            if self.use_grid:
                self._add_gaussians_to_grid(self._centers[sl], self._heights[sl])
                self._sync_bspline_coeffs()
            return {
                "n_hills": float(self.n_hills),
                "deposit_height_mean": float(w.mean().item()),
                "deposit_height_min": float(w.min().item()),
                "bias_energy_at_deposit_mean": float(V.mean().item()),
            }

    def free_energy_projection(
        self,
        axis_i: int,
        axis_j: int,
        grid_i: np.ndarray,
        grid_j: np.ndarray,
        *,
        fixed: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Estimate ``F ≈ −γ/(γ−1) V`` on a 2D slice of CV space.

        Axes ``axis_i`` / ``axis_j`` vary on ``grid_*``; remaining CVs are held
        at values from ``fixed`` (default 0.0).
        """
        if not (0 <= axis_i < self.n_cv and 0 <= axis_j < self.n_cv):
            raise ValueError(f"axes out of range for n_cv={self.n_cv}")
        if axis_i == axis_j:
            raise ValueError("axis_i and axis_j must differ")
        gi = np.asarray(grid_i, dtype=np.float64)
        gj = np.asarray(grid_j, dtype=np.float64)
        GI, GJ = np.meshgrid(gi, gj, indexing="xy")
        n = GI.size
        pts = np.zeros((n, self.n_cv), dtype=np.float64)
        if fixed:
            for ax, val in fixed.items():
                pts[:, int(ax)] = float(val)
        pts[:, axis_i] = GI.ravel()
        pts[:, axis_j] = GJ.ravel()
        with torch.no_grad():
            s_t = torch.as_tensor(pts, device=self._device, dtype=self._dtype)
            V = self.bias_potential_cv(s_t).detach().cpu().numpy()
        scale = self.bias_factor / (self.bias_factor - 1.0)
        return (-scale * V).reshape(GJ.shape)

    def free_energy_grid(
        self,
        s1_grid: np.ndarray,
        s2_grid: np.ndarray,
        *,
        fixed: dict[int, float] | None = None,
    ) -> np.ndarray:
        """2D FES on CV axes 0 and 1 (other CVs fixed; see ``free_energy_projection``)."""
        return self.free_energy_projection(0, 1, s1_grid, s2_grid, fixed=fixed)

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_per = _n_atoms_per_system(state, self.n_atoms_per_system)
        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        coordinate = coordinate.detach().requires_grad_(True)
        s = _cv_matrix(self.collective_variable(coordinate))
        energy = self.bias_potential_cv(s)
        if self.n_hills == 0:
            forces = torch.zeros(
                n_systems * n_per, 3, device=self._device, dtype=self._dtype
            )
            self._last_metrics = {
                "n_hills": 0.0,
                "metad_energy_mean": 0.0,
                "force_bias_rms": 0.0,
            }
            return {"energy": energy.detach(), "forces": forces}

        (grad_x,) = torch.autograd.grad(
            energy.sum(), coordinate, create_graph=False, retain_graph=False
        )
        forces = (-grad_x).reshape(n_systems * n_per, 3).detach()
        energy_out = energy.detach()
        self._last_metrics = {
            "n_hills": float(self.n_hills),
            "metad_energy_mean": float(energy_out.mean().item()),
            "metad_energy_max": float(energy_out.max().item()),
            "force_bias_rms": float(torch.sqrt(torch.mean(forces * forces)).item()),
        }
        return {"energy": energy_out, "forces": forces}


def _kinetic_temperature_stats(state) -> dict[str, float]:
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


class _TorchSimEnhancedDriver:
    """Common NVT Langevin loop around a SumModel."""

    def __init__(
        self,
        model,
        *,
        timestep: float,
        langevin_friction: float,
        temperature: float,
        update_pace: int = 20,
    ):
        self.model = model
        self.timestep = float(timestep)
        self.langevin_friction = float(langevin_friction)
        self.temperature = float(temperature)
        self.update_pace = max(1, int(update_pace))
        self._last_step_metrics: dict[str, float] = {}

    def run_chunked(
        self,
        system,
        num_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
        on_chunk_end: Optional[Callable[..., Any]] = None,
        label: str = "enhanced",
    ):
        import torch_sim as ts

        num_steps = int(num_steps)
        remaining = num_steps
        state = system
        done = 0
        t0 = time.perf_counter()
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                total=num_steps,
                desc=label,
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        while remaining > 0:
            chunk = min(self.update_pace, remaining)
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
            self._last_step_metrics = _kinetic_temperature_stats(state)
            self._last_step_metrics["target_temperature"] = float(self.temperature)
            if on_chunk_end is not None:
                extra = on_chunk_end(state, done) or {}
                self._last_step_metrics.update(extra)
            if progress is not None:
                progress.update(chunk)
            if verbose and (done % verbose_pace == 0 or remaining == 0):
                elapsed = max(time.perf_counter() - t0, 1e-12)
                m = self._last_step_metrics
                extra_bits = []
                if "n_hills" in m:
                    extra_bits.append(f"n_hills={int(m.get('n_hills', 0))}")
                if "umbrella_rms_dev_mean" in m:
                    extra_bits.append(
                        f"umb_rms={m.get('umbrella_rms_dev_mean', float('nan')):.3f}"
                    )
                msg = (
                    f"{label} Step {done}/{num_steps}, "
                    f"T_kin mean/std/min/max="
                    f"{m.get('kinetic_temperature_mean', float('nan')):.2f}/"
                    f"{m.get('kinetic_temperature_std', float('nan')):.2f}/"
                    f"{m.get('kinetic_temperature_min', float('nan')):.2f}/"
                    f"{m.get('kinetic_temperature_max', float('nan')):.2f} K"
                    + (", " + ", ".join(extra_bits) if extra_bits else "")
                    + f", {done / elapsed:.2f} it/s"
                )
                if progress is not None:
                    progress.write(msg)
                else:
                    print(msg, flush=True)
        if progress is not None:
            progress.close()
        elapsed = max(time.perf_counter() - t0, 1e-12)
        print(
            f"[{label}] done: {num_steps} steps in {elapsed:.2f}s "
            f"→ {num_steps / elapsed:.2f} it/s",
            flush=True,
        )
        return state


class TorchSimUmbrellaSampling(_TorchSimEnhancedDriver):
    """Parallel multi-window umbrella sampling (batched NVT Langevin).

    Each replica is one umbrella window with its own harmonic center in
    ``UmbrellaBias.centers``. Windows share the physical model but are
    otherwise independent — suitable for WHAM/MBAR post-processing via the
    dumped CV trajectory ``(n_frames, B, n_cv)``.
    """

    def __init__(
        self,
        model,
        umbrella_bias: UmbrellaBias,
        *,
        timestep: float,
        langevin_friction: float,
        temperature: float,
        update_pace: int = 20,
    ):
        super().__init__(
            model,
            timestep=timestep,
            langevin_friction=langevin_friction,
            temperature=temperature,
            update_pace=update_pace,
        )
        self.umbrella_bias = umbrella_bias
        self.cv_trajectory: list[np.ndarray] = []
        self.cv_times_ps: list[float] = []

    def clear_cv_trajectory(self) -> None:
        self.cv_trajectory.clear()
        self.cv_times_ps.clear()

    def run(
        self,
        system,
        num_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
        on_dump: Optional[Callable[..., Any]] = None,
        dump_every: int | None = None,
        record_cv: bool = True,
    ):
        dump_every = int(dump_every) if dump_every is not None else max(
            self.update_pace, 1
        )
        num_steps = int(num_steps)

        def _on_chunk(state, done: int):
            metrics = dict(self.umbrella_bias._last_metrics)
            cv = self.umbrella_bias._last_cv
            if cv is not None:
                metrics["cv_mean_0"] = float(cv[:, 0].mean().item())
                metrics["cv_mean_1"] = (
                    float(cv[:, 1].mean().item()) if cv.shape[1] > 1 else float("nan")
                )
            if done % dump_every == 0 or done == num_steps:
                if record_cv and cv is not None:
                    self.cv_trajectory.append(cv.detach().cpu().numpy())
                    self.cv_times_ps.append(float(done) * self.timestep)
                if on_dump is not None:
                    on_dump(done, state, metrics)
            return metrics

        return self.run_chunked(
            system,
            num_steps,
            verbose=verbose,
            verbose_pace=verbose_pace,
            show_progress=show_progress,
            on_chunk_end=_on_chunk,
            label="umbrella",
        )

    def stacked_cv_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times_ps [T], cv [T, B, n_cv])``."""
        if not self.cv_trajectory:
            return (
                np.zeros((0,), dtype=np.float64),
                np.zeros(
                    (0, self.umbrella_bias.n_windows, self.umbrella_bias.n_cv),
                    dtype=np.float64,
                ),
            )
        return (
            np.asarray(self.cv_times_ps, dtype=np.float64),
            np.stack(self.cv_trajectory, axis=0),
        )


class TorchSimWelltemperMetadynamic(_TorchSimEnhancedDriver):
    """Multi-walker well-tempered MetaD with a **shared** bias potential."""

    def __init__(
        self,
        model,
        metad_bias: WellTemperedMetadynamicsBias,
        *,
        timestep: float,
        langevin_friction: float,
        temperature: float,
        deposit_pace: int = 50,
        update_pace: int | None = None,
    ):
        pace = int(update_pace) if update_pace is not None else int(deposit_pace)
        super().__init__(
            model,
            timestep=timestep,
            langevin_friction=langevin_friction,
            temperature=temperature,
            update_pace=pace,
        )
        self.metad_bias = metad_bias
        self.deposit_pace = max(1, int(deposit_pace))
        # Align chunk size with deposit pace so hills update between integrate calls.
        self.update_pace = self.deposit_pace

    def run(
        self,
        system,
        num_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 100,
        show_progress: bool = True,
        on_dump: Optional[Callable[..., Any]] = None,
        dump_every: int | None = None,
    ):
        dump_every = int(dump_every) if dump_every is not None else self.deposit_pace

        def _on_chunk(state, done: int):
            coords = sim_state_to_batched_coordinates(state)
            dep = self.metad_bias.deposit(coords)
            metrics = dict(self.metad_bias._last_metrics)
            metrics.update(dep)
            if on_dump is not None and (
                done % dump_every == 0 or done == int(num_steps)
            ):
                on_dump(done, state, metrics)
            return metrics

        return self.run_chunked(
            system,
            num_steps,
            verbose=verbose,
            verbose_pace=verbose_pace,
            show_progress=show_progress,
            on_chunk_end=_on_chunk,
            label="wt-metad",
        )
