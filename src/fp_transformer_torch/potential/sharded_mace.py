"""Single-process multi-GPU MACE: shard replicas across devices, gather to primary.

FPTBias / Gaussian kernel still require the full replica batch on the primary
device. Only the MACE energy/force evaluation is sharded (per-system independent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .mace import load_mace_torch_sim_model

try:
    from torch_sim.models.interface import ModelInterface as _ModelInterface
except ImportError:  # pragma: no cover
    _ModelInterface = torch.nn.Module  # type: ignore[misc, assignment]


def parse_cuda_devices(
    devices: str | int | Sequence[str | int | torch.device] | None,
    *,
    fallback: torch.device | str | None = None,
) -> list[torch.device]:
    """Parse a device list for sharded MACE.

    Accepts:
      - ``None`` → single device from ``fallback`` (default cuda/cpu)
      - ``int`` → ``cuda:0 .. cuda:n-1`` (or ``[cpu]`` if n<=0 / no CUDA)
      - ``\"cuda:0,cuda:1\"`` / ``\"0,1,2\"`` comma-separated
      - sequence of device specs
    """
    if devices is None:
        if fallback is None:
            fallback = "cuda" if torch.cuda.is_available() else "cpu"
        return [torch.device(fallback)]

    if isinstance(devices, int):
        n = int(devices)
        if n <= 1 or not torch.cuda.is_available():
            if fallback is None:
                fallback = "cuda" if torch.cuda.is_available() else "cpu"
            return [torch.device(fallback)]
        return [torch.device(f"cuda:{i}") for i in range(n)]

    if isinstance(devices, str):
        parts = [p.strip() for p in devices.split(",") if p.strip()]
        if not parts:
            raise ValueError("empty --devices string")
        resolved: list[torch.device] = []
        for part in parts:
            if part.isdigit():
                resolved.append(torch.device(f"cuda:{part}"))
            else:
                resolved.append(torch.device(part))
        return resolved

    out: list[torch.device] = []
    for item in devices:
        if isinstance(item, torch.device):
            out.append(item)
        elif isinstance(item, int):
            out.append(torch.device(f"cuda:{item}"))
        else:
            s = str(item).strip()
            out.append(torch.device(f"cuda:{s}" if s.isdigit() else s))
    if not out:
        raise ValueError("empty devices sequence")
    return out


def shard_system_ranges(n_systems: int, n_shards: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` half-open ranges covering ``0 .. n_systems``.

    Empty trailing shards (when ``n_systems < n_shards``) are omitted.
    """
    if n_systems <= 0:
        raise ValueError(f"n_systems must be positive, got {n_systems}")
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    n_shards = min(n_shards, n_systems)
    base, rem = divmod(n_systems, n_shards)
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


class ShardedMaceModel(_ModelInterface):
    """``ModelInterface`` wrapper that evaluates MACE on multiple GPUs.

    The public ``device`` / ``dtype`` match the **primary** (first) device so
    ``SumModel(ShardedMaceModel(...), FPTBias(...))`` stays valid. Each forward:

    1. Split ``state`` by system index into contiguous shards
    2. Copy each shard to its GPU and run a local ``MaceModel``
    3. Gather energy / forces back to the primary device in system order
    """

    def __init__(
        self,
        models: Sequence[Any],
        *,
        primary_device: torch.device | None = None,
    ):
        super().__init__()
        if not models:
            raise ValueError("ShardedMaceModel requires at least one MaceModel")
        self.models = list(models)
        self.devices = [torch.device(m.device) for m in self.models]
        self._device = (
            torch.device(primary_device)
            if primary_device is not None
            else self.devices[0]
        )
        self._dtype = self.models[0].dtype
        for i, m in enumerate(self.models):
            if m.dtype != self._dtype:
                raise ValueError(
                    f"dtype mismatch: model 0 has {self._dtype}, model {i} has {m.dtype}"
                )
            if not bool(m.compute_forces):
                raise ValueError("all shard MACE models must compute_forces=True")
        self._compute_forces = True
        self._compute_stress = all(bool(m.compute_stress) for m in self.models)
        self._memory_scales_with = getattr(
            self.models[0], "memory_scales_with", "n_atoms_x_density"
        )
        # Keep child modules registered for .parameters() / state_dict if needed.
        for i, m in enumerate(self.models):
            self.add_module(f"mace_{i}", m)

    @classmethod
    def from_checkpoint(
        cls,
        model: str | Path = "off-small",
        *,
        devices: str | int | Sequence[str | int | torch.device] | None = None,
        dtype: torch.dtype = torch.float32,
        compute_forces: bool = True,
        compute_stress: bool = False,
        head: str | None = ...,  # type: ignore[assignment]
        cache_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> "ShardedMaceModel":
        """Load one frozen MACE copy per device."""
        device_list = parse_cuda_devices(devices)
        models = [
            load_mace_torch_sim_model(
                model,
                device=dev,
                dtype=dtype,
                compute_forces=compute_forces,
                compute_stress=compute_stress,
                head=head,
                cache_dir=cache_dir,
                **kwargs,
            )
            for dev in device_list
        ]
        return cls(models, primary_device=device_list[0])

    @property
    def r_max(self):
        return getattr(self.models[0], "r_max", None)

    @property
    def n_gpus(self) -> int:
        return len(self.models)

    def _shard_sim_state(self, state, lo: int, hi: int, device: torch.device):
        """Build a contiguous system shard on ``device`` (equal atoms/system)."""
        import torch_sim as ts

        n_systems = int(state.n_systems)
        n_atoms = int(state.n_atoms)
        if n_atoms % n_systems != 0:
            shard = state[list(range(lo, hi))]
            if shard.device != device or shard.dtype != self._dtype:
                shard = shard.to(device, self._dtype)
            return shard

        n_per = n_atoms // n_systems
        b = hi - lo
        pos = state.positions.view(n_systems, n_per, 3)[lo:hi].reshape(-1, 3)
        z = state.atomic_numbers.view(n_systems, n_per)[lo:hi].reshape(-1)
        mass = state.masses.view(n_systems, n_per)[lo:hi].reshape(-1)
        cell = state.cell[lo:hi]
        pbc = state.pbc
        if torch.is_tensor(pbc):
            pbc = pbc.to(device=device, non_blocking=True)

        pos = pos.to(device=device, dtype=self._dtype, non_blocking=True)
        mass = mass.to(device=device, dtype=self._dtype, non_blocking=True)
        cell = cell.to(device=device, dtype=self._dtype, non_blocking=True)
        z = z.to(device=device, dtype=torch.long, non_blocking=True)
        system_idx = torch.arange(b, device=device, dtype=torch.long).repeat_interleave(
            n_per
        )

        kwargs: dict[str, Any] = {
            "positions": pos,
            "masses": mass,
            "cell": cell,
            "pbc": pbc,
            "atomic_numbers": z,
            "system_idx": system_idx,
        }
        extras = getattr(state, "_system_extras", None) or {}
        for name, val in extras.items():
            if torch.is_tensor(val) and val.shape[0] == n_systems:
                kwargs[name] = val[lo:hi].to(
                    device=device, dtype=self._dtype, non_blocking=True
                )
            elif val is not None:
                kwargs[name] = val
        return ts.SimState(**kwargs)

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        ranges = shard_system_ranges(n_systems, len(self.models))
        active = list(zip(ranges, self.models, self.devices))

        if len(active) == 1:
            (lo, hi), model, _dev = active[0]
            assert lo == 0 and hi == n_systems
            out = model(state if state.device == model.device else state.to(model.device))
            energy = out["energy"].to(device=self._device, dtype=self._dtype)
            forces = out["forces"].to(device=self._device, dtype=self._dtype)
            result = {"energy": energy, "forces": forces}
            if "stress" in out and out["stress"] is not None:
                result["stress"] = out["stress"].to(
                    device=self._device, dtype=self._dtype
                )
            return result

        # Materialize shards on each device, then evaluate concurrently.
        modules: list[Any] = []
        inputs: list[tuple[Any]] = []
        devices: list[torch.device] = []
        for (lo, hi), model, dev in active:
            shard = self._shard_sim_state(state, lo, hi, dev)
            modules.append(model)
            inputs.append((shard,))
            devices.append(dev)

        from torch.nn.parallel.parallel_apply import parallel_apply

        shard_outs = parallel_apply(modules, inputs, devices=devices)

        if torch.cuda.is_available():
            for _dev in {d for d in devices if d.type == "cuda"}:
                torch.cuda.synchronize(_dev)

        energies = [
            o["energy"].to(device=self._device, dtype=self._dtype) for o in shard_outs
        ]
        forces = [
            o["forces"].to(device=self._device, dtype=self._dtype) for o in shard_outs
        ]
        result = {
            "energy": torch.cat(energies, dim=0),
            "forces": torch.cat(forces, dim=0),
        }
        if self._compute_stress and all("stress" in o for o in shard_outs):
            stresses = [
                o["stress"].to(device=self._device, dtype=self._dtype) for o in shard_outs
            ]
            result["stress"] = torch.cat(stresses, dim=0)
        return result


def load_sharded_mace_torch_sim_model(
    model: str | Path = "off-small",
    *,
    devices: str | int | Sequence[str | int | torch.device] | None = None,
    n_gpus: int | None = None,
    dtype: torch.dtype = torch.float32,
    compute_forces: bool = True,
    compute_stress: bool = False,
    head: str | None = ...,  # type: ignore[assignment]
    cache_dir: Path | str | None = None,
    **kwargs: Any,
) -> ShardedMaceModel | Any:
    """Load sharded MACE when ``n_gpus``/``devices`` > 1, else a single ``MaceModel``.

    ``n_gpus`` is a convenience alias for ``devices=n_gpus``.
    """
    if n_gpus is not None and devices is None:
        devices = int(n_gpus)
    device_list = parse_cuda_devices(devices)
    if len(device_list) == 1:
        return load_mace_torch_sim_model(
            model,
            device=device_list[0],
            dtype=dtype,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            head=head,
            cache_dir=cache_dir,
            **kwargs,
        )
    return ShardedMaceModel.from_checkpoint(
        model,
        devices=device_list,
        dtype=dtype,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
        head=head,
        cache_dir=cache_dir,
        **kwargs,
    )
