#!/usr/bin/env python3
"""Parity + throughput smoke for ShardedMaceModel (submit on a GPU node).

Compares single-GPU MACE forces vs 2-GPU sharded forces on the same dialanine
batch, then optionally times forward passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from fp_transformer_torch.potential.mace import (  # noqa: E402
    atomic_numbers_from_pdb_organic,
    batched_coordinates_to_sim_state,
    load_mace_torch_sim_model,
)
from fp_transformer_torch.potential.sharded_mace import (  # noqa: E402
    load_sharded_mace_torch_sim_model,
    shard_system_ranges,
)
from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", default="off-small")
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--n-gpus", type=int, default=2)
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    n_visible = torch.cuda.device_count()
    if args.n_gpus > n_visible:
        raise SystemExit(f"--n-gpus={args.n_gpus} > visible GPUs {n_visible}")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    pdb = ROOT / "experiments/dialanine/dialanine.pdb"
    sponge = ROOT / "experiments/dialanine/dialanine"
    init_npy = ROOT / "experiments/dialanine_mace/init_coords_1024.npy"

    z = atomic_numbers_from_pdb_organic(pdb)
    n_atoms = int(z.numel())
    if init_npy.is_file():
        coords_np = np.load(init_npy)[: args.replicas]
    else:
        base = load_sponge_coordinates(sponge, num_atoms=n_atoms)
        coords_np = np.repeat(base[None, ...], args.replicas, axis=0)
    coords = torch.as_tensor(coords_np, dtype=dtype, device="cuda:0")
    mass = torch.ones(n_atoms, dtype=dtype, device="cuda:0")
    state = batched_coordinates_to_sim_state(
        coords, atomic_numbers=z.to("cuda:0"), masses=mass, pbc=False
    )

    print(f"replicas={args.replicas}  n_gpus={args.n_gpus}  dtype={args.dtype}")
    print(f"shard ranges: {shard_system_ranges(args.replicas, args.n_gpus)}")

    single = load_mace_torch_sim_model(
        args.mace_model,
        device="cuda:0",
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )
    sharded = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=args.n_gpus,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )

    out1 = single(state)
    outn = sharded(state)
    e_err = (out1["energy"] - outn["energy"]).abs()
    f_err = (out1["forces"] - outn["forces"]).abs()
    print(
        f"energy max|Δ|={float(e_err.max()):.3e}  "
        f"forces max|Δ|={float(f_err.max()):.3e}  "
        f"forces rms={float(f_err.pow(2).mean().sqrt()):.3e}"
    )
    ok = torch.allclose(out1["energy"], outn["energy"], atol=args.atol, rtol=args.rtol)
    ok = ok and torch.allclose(
        out1["forces"], outn["forces"], atol=args.atol, rtol=args.rtol
    )
    if not ok:
        raise SystemExit("PARITY FAIL")
    print("PARITY OK")

    def _bench(model, label: str) -> float:
        for _ in range(args.warmup):
            model(state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.bench_iters):
            model(state)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        it_s = args.bench_iters / dt
        print(f"[bench] {label}: {args.bench_iters} iters in {dt:.3f}s → {it_s:.2f} it/s")
        return it_s

    r1 = _bench(single, "1gpu")
    rn = _bench(sharded, f"{args.n_gpus}gpu")
    print(f"speedup={rn / r1:.2f}x")


if __name__ == "__main__":
    main()
