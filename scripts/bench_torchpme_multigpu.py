#!/usr/bin/env python3
"""Compute-node bench: batched torchpme correctness + R=512 multi-GPU profile.

Compares nvalchemiops PME (1 GPU / N GPU streams) vs torchpme (batched FFT,
1 GPU / N GPU CUDA streams) on the dialanine-water box.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from fp_transformer_torch.potential.mace import batched_coordinates_to_sim_state  # noqa: E402
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402


def _load_xyz(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(coords, dtype=np.float64)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_state(coords: np.ndarray, topo: MLMMTopology, cell: np.ndarray, replicas: int, device, dtype):
    batch = np.repeat(coords[None, ...], replicas, axis=0)
    coordinate = torch.as_tensor(batch, dtype=dtype, device=device)
    return batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=topo.atomic_numbers.to(device),
        masses=topo.masses.to(device=device, dtype=dtype),
        cell=cell,
        pbc=True,
        charge=0.0,
        spin=1.0,
    )


def _make_model(
    topo: MLMMTopology,
    *,
    electrostatics: str,
    n_gpus: int,
    device: torch.device,
    dtype: torch.dtype,
    cutoff: float,
    mesh: float,
    method: str,
) -> MechanicalEmbeddingMLMM:
    devices = parse_cuda_devices(n_gpus)
    return MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        enable_ml_coulomb_correction=False,
        cutoff=cutoff,
        electrostatics=electrostatics,
        pme_accuracy=1e-4,
        pme_mesh_spacing=mesh,
        torchpme_method=method,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        mm_devices=devices if n_gpus > 1 else None,
    )


def _time_forward(model, state, *, warmup: int, repeats: int) -> float:
    for _ in range(max(0, warmup)):
        _ = model(state)
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(state)
    _sync()
    return (time.perf_counter() - t0) * 1e3 / max(repeats, 1)


def run_pytest() -> int:
    print("\n========== pytest torchpme ==========", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests" / "test_mechanical_embedding_mlmm.py"),
        "-k",
        "torchpme",
        "-q",
        "--tb=short",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    print(f"pytest exit={proc.returncode}", flush=True)
    return int(proc.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--box-dir",
        type=Path,
        default=ROOT / "experiments/dialanine_water/mm_box",
    )
    p.add_argument("--replicas", type=int, default=512)
    p.add_argument("--check-replicas", type=int, default=8)
    p.add_argument("--n-gpus", type=int, default=0, help="0 → all visible GPUs")
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument("--torchpme-method", type=str, default="p3m", choices=["p3m", "pme"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--skip-pytest", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    n_vis = int(torch.cuda.device_count())
    n_gpus = n_vis if int(args.n_gpus) <= 0 else min(int(args.n_gpus), n_vis)
    if n_gpus < 1:
        raise SystemExit("need at least 1 GPU")
    device = torch.device("cuda:0")
    print(
        f"host GPUs={n_vis}  using n_gpus={n_gpus}  "
        f"names={[torch.cuda.get_device_name(i) for i in range(n_vis)]}",
        flush=True,
    )

    pytest_rc = 0 if args.skip_pytest else run_pytest()

    box = Path(args.box_dir)
    topo64 = MLMMTopology.load_npz(box / "topology.npz").to(device, torch.float64)
    topo32 = MLMMTopology.load_npz(box / "topology.npz").to(device, torch.float32)
    cell = np.loadtxt(box / "cell.txt", dtype=np.float64).reshape(3, 3)
    coords = _load_xyz(box / "box.xyz")
    print(f"box N={coords.shape[0]}  L={cell[0, 0]:.3f} Å  R_check={args.check_replicas}", flush=True)

    print("\n========== correctness (float64) ==========", flush=True)
    state64 = _make_state(coords, topo64, cell, int(args.check_replicas), device, torch.float64)
    models = {
        "pme_1gpu": _make_model(
            topo64, electrostatics="pme", n_gpus=1, device=device,
            dtype=torch.float64, cutoff=args.pme_cutoff, mesh=args.pme_mesh_spacing,
            method=args.torchpme_method,
        ),
        "torchpme_1gpu": _make_model(
            topo64, electrostatics="torchpme", n_gpus=1, device=device,
            dtype=torch.float64, cutoff=args.pme_cutoff, mesh=args.pme_mesh_spacing,
            method=args.torchpme_method,
        ),
    }
    if n_gpus > 1:
        models["torchpme_ngpu"] = _make_model(
            topo64, electrostatics="torchpme", n_gpus=n_gpus, device=device,
            dtype=torch.float64, cutoff=args.pme_cutoff, mesh=args.pme_mesh_spacing,
            method=args.torchpme_method,
        )
        models["pme_ngpu"] = _make_model(
            topo64, electrostatics="pme", n_gpus=n_gpus, device=device,
            dtype=torch.float64, cutoff=args.pme_cutoff, mesh=args.pme_mesh_spacing,
            method=args.torchpme_method,
        )

    outs = {}
    for name, model in models.items():
        print(f"  forward {name}  workers={len(model._mm_workers)}", flush=True)
        _sync()
        outs[name] = model(state64)
        _sync()
        e = outs[name]["energy"]
        print(
            f"    E_mean={float(e.mean()):.6f}  E_std={float(e.std()):.4g}  "
            f"F_rms={float(torch.sqrt((outs[name]['forces'] ** 2).mean())):.6f}",
            flush=True,
        )

    def _rel(a, b, key: str) -> tuple[float, float]:
        x, y = outs[a][key], outs[b][key]
        denom = y.norm().clamp_min(1e-12) if key == "forces" else y.abs().mean().clamp_min(1e-12)
        if key == "energy":
            return float((x - y).abs().max() / denom), float((x - y).abs().mean() / denom)
        return float((x - y).norm() / denom), float((x - y).abs().mean() / y.abs().mean().clamp_min(1e-12))

    pairs = [("torchpme_1gpu", "pme_1gpu")]
    if "torchpme_ngpu" in outs:
        pairs.append(("torchpme_ngpu", "torchpme_1gpu"))
        pairs.append(("pme_ngpu", "pme_1gpu"))
    print("\nrelative errors (row vs col):", flush=True)
    for a, b in pairs:
        de_max, de_mean = _rel(a, b, "energy")
        df_n, df_m = _rel(a, b, "forces")
        print(
            f"  {a:16s} vs {b:16s}  "
            f"dE_max={de_max:.4g}  dE_mean={de_mean:.4g}  "
            f"dF_rel={df_n:.4g}  dF_mae={df_m:.4g}",
            flush=True,
        )
        if a.startswith("torchpme_ngpu") and b.startswith("torchpme"):
            if de_max > 1e-4 or df_n > 1e-4:
                print("  [WARN] multi-GPU torchpme drifted from 1-GPU torchpme", flush=True)

    timings: list[tuple[str, int, float]] = []
    if not args.skip_profile:
        print(
            f"\n========== profile R={args.replicas} float32 warmup={args.warmup} repeats={args.repeats} ==========",
            flush=True,
        )
        state32 = _make_state(
            coords, topo32, cell, int(args.replicas), device, torch.float32
        )
        jobs = [
            ("pme", 1),
            ("torchpme", 1),
        ]
        if n_gpus > 1:
            jobs.extend([("pme", n_gpus), ("torchpme", n_gpus)])
        for elec, ng in jobs:
            label = f"{elec}_{ng}gpu"
            print(f"\n--- {label} ---", flush=True)
            model = _make_model(
                topo32,
                electrostatics=elec,
                n_gpus=ng,
                device=device,
                dtype=torch.float32,
                cutoff=args.pme_cutoff,
                mesh=args.pme_mesh_spacing,
                method=args.torchpme_method,
            )
            print(f"workers={len(model._mm_workers)}", flush=True)
            ms_fwd = _time_forward(
                model, state32, warmup=int(args.warmup), repeats=int(args.repeats)
            )
            timings.append((label, ng, ms_fwd))
            print(f"forward_mean  {ms_fwd:8.2f} ms   ({1000.0 / ms_fwd:.3f} it/s)", flush=True)
            prof = model.profile_forward(
                state32, warmup=1, repeats=max(2, int(args.repeats) // 2)
            )
            total = prof["forward_total"]
            for k in (
                "mm_sharded",
                "neighbor_list",
                "pme_or_dsf",
                "pme_exclusion",
                "lj",
                "bonded",
                "forward_total",
            ):
                v = prof[k]
                if k == "forward_total":
                    print(f"  {k:18s}  {v:8.2f} ms", flush=True)
                else:
                    pct = 100.0 * v / total if total > 0 else 0.0
                    print(f"  {k:18s}  {v:8.2f} ms   ({pct:5.1f}%)", flush=True)
            del model
            torch.cuda.empty_cache()

        print("\n========== summary (ms / forward, R=512) ==========", flush=True)
        ref = next((t for t in timings if t[0] == "pme_1gpu"), None)
        for label, ng, ms in timings:
            speedup = (ref[2] / ms) if ref and ms > 0 else float("nan")
            print(f"  {label:18s}  {ms:8.2f} ms   vs 1×PME ×{speedup:.2f}", flush=True)

    if pytest_rc != 0:
        raise SystemExit(f"pytest failed with code {pytest_rc}")
    print("\nbench ok", flush=True)


if __name__ == "__main__":
    main()
