#!/usr/bin/env python3
"""Solvated Claisen + mechanical embedding ML/MM + FPT.

Same box / topology as ``rearrangement_water_metadynamics`` (MACE solute +
TIP3P/PME water). CVs: d1=d(O7,C8), d2=d(C1,C10). Default ``λ_init=0.1``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import (  # noqa: E402
    atomic_numbers_from_symbols,
    load_xyz,
    save_stage_artifacts,
)
from fp_transformer_torch import (  # noqa: E402
    FPTBias,
    GaussianKernel,
    PerReplicaRescaleConfig,
    QuadraticDistanceWall,
    SoftminDistanceUpperWall,
    MultiParticlePairDistance,
    TorchSimFokkerPlanckIntegrator,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402

DEFAULT_BOX = EXPERIMENT_DIR / "mlmm_box"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
KB_EV_PER_K = 8.617330337217213e-5


def initial_bandwidth_from_collective_variable(cv: torch.Tensor) -> float:
    distance = pairwise_distance(cv)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    return median if np.isfinite(median) and median > 0 else 0.5


def load_pair_indices(path: Path) -> list[tuple[int, int]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ints = []
        for part in line.replace(",", " ").split():
            try:
                ints.append(int(part) - 1)
            except ValueError:
                continue
        if len(ints) >= 2:
            rows.append((ints[0], ints[1]))
    if len(rows) != 2:
        raise ValueError(f"need exactly 2 pairs in {path}, got {len(rows)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--xyz", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--cell-file", type=str, default=None)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=256)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=0.5)
    p.add_argument("--ess-threshold", type=float, default=0.97)
    p.add_argument("--min-lambda-step", type=float, default=0.001)
    p.add_argument("--max-lambda-step", type=float, default=0.002)
    p.add_argument("--num-lambda-trials", type=int, default=10)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--pre-md-steps", type=int, default=100)
    p.add_argument("--pre-md-dump-every", type=int, default=50)
    p.add_argument("--max-xyz-frames", type=int, default=32)
    p.add_argument("--heal-lambdas", type=str, default="0.1,0.05,0.02")
    p.add_argument("--heal-steps", type=int, default=100)
    p.add_argument("--pme-cutoff", type=float, default=9.0)
    p.add_argument(
        "--pme-mesh-spacing",
        type=float,
        default=1.5,
        help="PME reciprocal mesh spacing (Å); larger = faster, less accurate",
    )
    p.add_argument(
        "--electrostatics",
        type=str,
        default="pme",
        choices=["pme", "dsf"],
        help="MM long-range electrostatics: pme (accurate) or dsf (faster)",
    )
    p.add_argument("--dsf-alpha", type=float, default=0.2, help="DSF damping α (1/Å)")
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--softmin-a", type=float, default=10.0)
    p.add_argument("--softmin-d-wall", type=float, default=2.7)
    p.add_argument("--softmin-k", type=float, default=10.0)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=20)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cv-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=4.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=5.0)
    p.add_argument("--rew-lambda-init", type=float, default=None)
    p.add_argument("--rew-lambda-max", type=float, default=None)
    p.add_argument("--rew-ess-threshold", type=float, default=None)
    p.add_argument("--rew-min-lambda-step", type=float, default=None)
    p.add_argument("--rew-max-lambda-step", type=float, default=None)
    p.add_argument("--rew-num-lambda-trials", type=int, default=None)
    p.add_argument("--rew-anneal-ref-temperature", type=float, default=None)
    return p


def _resolve(args):
    if args.steps is not None:
        args.stage_steps = args.steps
    if args.traj_full:
        args.traj_solute_only = False
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    for src, dst in [
        ("rew_lambda_init", "lambda_init"),
        ("rew_lambda_max", "lambda_max"),
        ("rew_ess_threshold", "ess_threshold"),
        ("rew_min_lambda_step", "min_lambda_step"),
        ("rew_max_lambda_step", "max_lambda_step"),
        ("rew_num_lambda_trials", "num_lambda_trials"),
    ]:
        v = getattr(args, src)
        if v is not None:
            setattr(args, dst, v)
    temp = args.temperature
    if args.rew_anneal_ref_temperature is not None:
        temp = args.rew_anneal_ref_temperature
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    heal = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    args.heal_lambda_list = heal
    box = Path(args.box_dir)
    args.xyz = Path(args.xyz or (box / "box.xyz"))
    args.topology = Path(args.topology or (box / "topology.npz"))
    args.cell_file = Path(args.cell_file or (box / "cell.txt"))
    return args


def main() -> None:
    args = _resolve(build_parser().parse_args())
    primary = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if args.devices:
        mace_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mace_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mace_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    import torch_sim as ts
    from torch_sim.models.interface import SumModel

    if not Path(args.topology).is_file():
        raise FileNotFoundError(
            f"missing topology {args.topology}; run prepare_mlmm_box.py first"
        )

    symbols, xyz_coords = load_xyz(Path(args.xyz))
    topo = MLMMTopology.load_npz(args.topology).to(device, dtype)
    cell_np = np.loadtxt(args.cell_file, dtype=np.float64).reshape(3, 3)
    cell = torch.as_tensor(cell_np, dtype=dtype, device=device)
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(
            f"xyz atoms {xyz_coords.shape[0]} != topology N={num_atoms}"
        )

    pairs = load_pair_indices(Path(args.cv_file))
    n_solute = int(topo.ml_mask.sum().item())
    solute_symbols = symbols[:n_solute]

    def _art_coords_symbols(coords: np.ndarray):
        if args.traj_solute_only:
            return coords[..., :n_solute, :], solute_symbols
        return coords, symbols

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    mlmm = MechanicalEmbeddingMLMM(
        topo,
        mace_model=mace_model,
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        dsf_alpha=float(args.dsf_alpha),
        pme_accuracy=1e-4,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        mm_devices=None,  # PME not thread-safe; multi-GPU MM shard is slower than 1×GPU
    )
    print(
        f"Claisen-water FPT  device={device}  mace={args.mace_model}  "
        f"N={num_atoms}  ML={n_solute}  R={args.replicas}  λ_init={args.lambda_init}  "
        f"elec={args.electrostatics}  mace_gpus={len(mace_devices)}  "
        f"mm_gpus={len(mlmm._mm_workers) or 1}",
        flush=True,
    )

    cv_fn = MultiParticlePairDistance(pairs)
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    softmin_wall = SoftminDistanceUpperWall(
        pairs[0],
        pairs[1],
        a=float(args.softmin_a),
        d_wall=float(args.softmin_d_wall),
        k=float(args.softmin_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )

    out = Path(
        args.output
        or (
            EXPERIMENT_DIR
            / "results"
            / f"claisen_water_fpt_{args.mace_model}_{args.replicas}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)

    if args.init_npy:
        loaded = np.load(Path(args.init_npy))[: args.replicas]
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[pre-md] steps={args.pre_md_steps}", flush=True)
        md_model = SumModel(mlmm, wall, softmin_wall)
        state_md = batched_coordinates_to_sim_state(
            coordinate,
            atomic_numbers=z_t,
            masses=mass_t,
            cell=cell,
            pbc=True,
            charge=float(args.charge),
            spin=float(args.spin),
        )
        # attach MM charges for any consumer that reads them
        extras = dict(state_md.system_extras)
        atom_extras = dict(state_md.atom_extras)
        atom_extras["partial_charges"] = topo.charges.repeat(args.replicas)
        state_md = ts.SimState(
            positions=state_md.positions,
            masses=state_md.masses,
            cell=state_md.row_vector_cell,
            pbc=True,
            atomic_numbers=state_md.atomic_numbers,
            system_idx=state_md.system_idx,
            _system_extras=extras,
            _atom_extras=atom_extras,
        )
        done = 0
        dump_every = max(1, int(args.pre_md_dump_every))
        md_steps = max(0, int(args.pre_md_steps))
        dumps = []
        while done < md_steps:
            n_this = min(dump_every, md_steps - done)
            state_md = ts.integrate(
                state_md,
                md_model,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=n_this,
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                autobatcher=False,
                pbar=False,
            )
            done += n_this
            dumps.append(
                sim_state_to_batched_coordinates(state_md).detach().cpu().numpy()
            )
            print(f"[pre-md] step={done}", flush=True)
        if dumps:
            np.save(out / "pre_md_traj.npy", np.stack(dumps, axis=0))
            coordinate = torch.as_tensor(dumps[-1], dtype=dtype, device=device)
        np.save(out / "init_coords_from_pre_md.npy", coordinate.detach().cpu().numpy())

    collective0 = cv_fn(coordinate)
    bw0 = initial_bandwidth_from_collective_variable(collective0)
    cv0 = collective0.detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    print(
        f"[init] bw={bw0:.4g}  d1 med={np.median(cv0[:,0]):.3f}  "
        f"d2 med={np.median(cv0[:,1]):.3f}",
        flush=True,
    )

    rescale = PerReplicaRescaleConfig(
        enabled=bool(args.rescale_per_replica),
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=args.rescale_adaptive_ref_quantile,
        adaptive_band_low=args.rescale_adaptive_band_low,
        adaptive_band_high=args.rescale_adaptive_band_high,
        bandwidth_min=args.rescale_bw_min,
        bandwidth_max=args.rescale_bw_max,
        global_pull=0.02,
        fixed_row_sum_min=float(args.rescale_fixed_row_sum_min),
        fixed_row_sum_max=float(args.rescale_fixed_row_sum_max),
        use_normalized_kernel=False,
        normalize_row_sum_by_replicas=False,
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bw0,
        distance_function=pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )
    cv_clip = None if args.cv_grad_clip_norm <= 0 else float(args.cv_grad_clip_norm)
    fpt_bias = FPTBias(
        cv_fn,
        kernel,
        kernel_grad_beta=float(args.lambda_init) * float(args.beta0),
        device=device,
        dtype=dtype,
        collective_variable_grad_clip=cv_clip,
        update_pace=args.update_pace,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mlmm, fpt_bias, wall, softmin_wall)

    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=True,
        charge=float(args.charge),
        spin=float(args.spin),
    )
    extras = dict(state0.system_extras)
    atom_extras = dict(state0.atom_extras)
    atom_extras["partial_charges"] = topo.charges.repeat(int(coordinate.shape[0]))
    state0 = ts.SimState(
        positions=state0.positions,
        masses=state0.masses,
        cell=state0.row_vector_cell,
        pbc=True,
        atomic_numbers=state0.atomic_numbers,
        system_idx=state0.system_idx,
        _system_extras=extras,
        _atom_extras=atom_extras,
    )

    integrator = TorchSimFokkerPlanckIntegrator(
        model,
        fpt_bias,
        cv_fn,
        kernel,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        beta0=float(args.beta0),
        lambda_init=float(args.lambda_init),
        lambda_max=float(args.lambda_max),
        ess_threshold=float(args.ess_threshold),
        min_lambda_step=float(args.min_lambda_step),
        max_lambda_step=float(args.max_lambda_step),
        num_lambda_trials=int(args.num_lambda_trials),
        update_pace=int(args.update_pace),
        device=device,
        dtype=dtype,
    )

    def _tag(i, lam, phase):
        pref = "heal" if phase == "heal" else "stage"
        return f"{pref}{i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i, lam, st, phase="anneal"):
        coords = sim_state_to_batched_coordinates(st)
        cv = cv_fn(coords).detach().cpu().numpy()
        if cv.ndim == 3:
            cv = cv.reshape(cv.shape[0], -1)
        tag = _tag(stage_i, lam, phase)
        np.save(
            out / f"{tag}_bandwidth.npy",
            integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        print(
            f"[stage/{phase}] {tag}  d1 med={np.median(cv[:,0]):.3f}  "
            f"d2 med={np.median(cv[:,1]):.3f}",
            flush=True,
        )
        # full coords for analysis; XYZ uses solute-only when enabled
        np.save(out / f"{tag}_coords_full.npy", coords.detach().cpu().numpy().astype(np.float32))
        art_c, art_s = _art_coords_symbols(coords.detach().cpu().numpy())
        save_stage_artifacts(
            out,
            tag,
            art_c,
            cv,
            art_s,
            title=f"Claisen-water-FPT {phase} {tag}",
            max_xyz_frames=args.max_xyz_frames,
        )

    art0, art0s = _art_coords_symbols(coordinate.detach().cpu().numpy())
    save_stage_artifacts(
        out,
        "stage_init",
        art0,
        cv0,
        art0s,
        title="Claisen-water-FPT init",
        max_xyz_frames=args.max_xyz_frames,
    )
    np.save(
        out / "stage_init_coords_full.npy",
        coordinate.detach().cpu().numpy().astype(np.float32),
    )
    if args.heal_lambda_list:
        fpt_bias.set_kernel_grad_beta(
            float(args.heal_lambda_list[0]) * float(args.beta0)
        )

    final = integrator.run(
        state0,
        args.stage_steps,
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_stage_end=on_stage_end,
        heal_lambdas=args.heal_lambda_list or None,
        heal_steps=int(args.heal_steps) if args.heal_lambda_list else None,
    )
    fc = sim_state_to_batched_coordinates(final)
    fcv = cv_fn(fc).detach().cpu().numpy()
    if fcv.ndim == 3:
        fcv = fcv.reshape(fcv.shape[0], -1)
    art_f, art_fs = _art_coords_symbols(fc.detach().cpu().numpy())
    np.save(out / "stage_final_coords_full.npy", fc.detach().cpu().numpy().astype(np.float32))
    save_stage_artifacts(
        out,
        "stage_final",
        art_f,
        fcv,
        art_fs,
        title=f"Claisen-water-FPT final λ={integrator.lambda_current:.4g}",
        max_xyz_frames=args.max_xyz_frames,
    )
    print(f"done out={out} λ={integrator.lambda_current}", flush=True)


if __name__ == "__main__":
    main()
