#!/usr/bin/env python3
"""Build 2D US init coords for dialanine_water from MetaD solute trajectory.

1. Load solute-only MetaD ``trajectory.xyz``, compute φ/ψ for every frame.
2. Build an ``nφ × nψ`` umbrella grid on (−π, π].
3. For each window center, pick the MetaD frame with smallest periodic CV
   distance.
4. Re-solvate each selected solute with TIP3P (same box as ``mm_box``) and
   write ``init_coords.npy`` shaped ``(n_windows, N_atoms, 3)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WATER_DIR = Path(__file__).resolve().parents[1] / "dialanine_water"
MACE_EXP = Path(__file__).resolve().parents[1] / "rearrangement_mace"
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WATER_DIR))
sys.path.insert(0, str(MACE_EXP))

from fp_transformer_torch import Torsion  # noqa: E402
from prepare_mlmm_box import pack_tip3p_lattice  # noqa: E402

DEFAULT_BOX = WATER_DIR / "mm_box"
DEFAULT_META_TRAJ = (
    ROOT
    / "experiments/dialanine_water_meta/results"
    / "dialanine_water_metad_r64_s0_87241/trajectory.xyz"
)
N_SOLUTE = 22
TWO_PI = 2.0 * np.pi


def load_torsion_indices(path: Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        chunk: list[int] = []
        for part in parts:
            try:
                chunk.append(int(part) - 1)
            except ValueError:
                continue
            if len(chunk) == 4:
                rows.append(chunk)
                chunk = []
    if not rows:
        raise ValueError(f"no torsion indices in {path}")
    return np.asarray(rows, dtype=np.int64)


def load_xyz_traj(path: Path) -> tuple[list[str], np.ndarray]:
    """Return symbols and ``(n_frames, n_atoms, 3)``."""
    frames: list[np.ndarray] = []
    symbols: list[str] | None = None
    with Path(path).open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line.strip())
            f.readline()  # comment
            syms: list[str] = []
            xyz = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                parts = f.readline().split()
                syms.append(parts[0])
                xyz[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
            if symbols is None:
                symbols = syms
            frames.append(xyz)
    if not frames or symbols is None:
        raise ValueError(f"empty trajectory {path}")
    return symbols, np.stack(frames, axis=0)


def make_phi_psi_centers(n_phi: int, n_psi: int) -> np.ndarray:
    """Bin-center grid on (−π, π], shape ``(n_phi * n_psi, 2)``, φ varies fastest."""
    n_phi = int(n_phi)
    n_psi = int(n_psi)
    edges_phi = np.linspace(-np.pi, np.pi, n_phi + 1, dtype=np.float64)
    edges_psi = np.linspace(-np.pi, np.pi, n_psi + 1, dtype=np.float64)
    phi = 0.5 * (edges_phi[:-1] + edges_phi[1:])
    psi = 0.5 * (edges_psi[:-1] + edges_psi[1:])
    gg = np.stack(np.meshgrid(phi, psi, indexing="xy"), axis=-1)  # (n_psi, n_phi, 2)
    return gg.reshape(-1, 2)


def periodic_dist2(a: np.ndarray, b: np.ndarray, period: float = TWO_PI) -> np.ndarray:
    """Squared torus distance. ``a`` (N,d), ``b`` (M,d) → (N,M)."""
    d = a[:, None, :] - b[None, :, :]
    d = d - period * np.round(d / period)
    return np.sum(d * d, axis=-1)


def solvate_fixed_n(
    solute: np.ndarray,
    *,
    box: float,
    n_waters: int,
    density: float,
    min_dist: float,
    seed: int,
    ref_waters: np.ndarray,
) -> np.ndarray:
    """Pack TIP3P; force exactly ``n_waters`` to match topology.

    Prefer fresh lattice pack; if the water count undershoots (tight min_dist),
    fall back to centering the solute in the reference box and keeping the
    reference water coordinates.
    """
    sol_c, waters, n = pack_tip3p_lattice(
        solute,
        float(box),
        density=float(density),
        min_dist=float(min_dist),
        seed=int(seed),
    )
    target = int(n_waters)
    if n >= target:
        wat = waters[: target * 3]
        return np.concatenate([sol_c, wat], axis=0).astype(np.float32)

    # Undershoot → keep reference waters, recenter solute like pack does
    sol = np.asarray(solute, dtype=np.float64)
    shift = np.array([box / 2.0, box / 2.0, box / 2.0]) - sol.mean(axis=0)
    sol_c = sol + shift
    wat = np.asarray(ref_waters, dtype=np.float64).reshape(target * 3, 3)
    return np.concatenate([sol_c, wat], axis=0).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj-xyz", type=Path, default=DEFAULT_META_TRAJ)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=Path, default=None)
    p.add_argument("--n-phi", type=int, default=16)
    p.add_argument("--n-psi", type=int, default=16)
    p.add_argument("--box", type=float, default=18.0)
    p.add_argument("--density", type=float, default=0.033)
    p.add_argument("--min-dist", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument(
        "--output",
        type=Path,
        default=EXP / "init_us2d_r256.npy",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    box_dir = Path(args.box_dir)
    cv_file = Path(args.cv_file or (box_dir / "backbone_cv.txt"))
    _, ref_xyz = _load_single_xyz(box_dir / "box.xyz")
    n_atoms = int(ref_xyz.shape[0])
    n_waters = (n_atoms - N_SOLUTE) // 3
    ref_waters = ref_xyz[N_SOLUTE:]
    cell = float(np.loadtxt(box_dir / "cell.txt").reshape(3, 3)[0, 0])
    box_len = float(args.box or cell)

    symbols, traj = load_xyz_traj(Path(args.traj_xyz))
    if traj.shape[1] < N_SOLUTE:
        raise ValueError(f"traj atoms {traj.shape[1]} < N_SOLUTE={N_SOLUTE}")
    solute_traj = traj[:, :N_SOLUTE, :].astype(np.float64)
    print(
        f"[init] traj={args.traj_xyz}  frames={solute_traj.shape[0]}  "
        f"solute={N_SOLUTE}  target_waters={n_waters}  box={box_len}",
        flush=True,
    )

    tors = load_torsion_indices(cv_file)[:2]
    cv_fn = Torsion(tors)
    device = torch.device(args.device)
    cvs = []
    bs = int(args.batch)
    with torch.no_grad():
        for i0 in range(0, solute_traj.shape[0], bs):
            chunk = torch.as_tensor(
                solute_traj[i0 : i0 + bs], dtype=torch.float32, device=device
            )
            s = cv_fn(chunk).detach().cpu().numpy()
            if s.ndim == 3:
                s = s.reshape(s.shape[0], -1)
            cvs.append(s.astype(np.float64))
    pool_cv = np.concatenate(cvs, axis=0)
    print(
        f"[CV] φ/ψ torsions={tors.tolist()}  pool={pool_cv.shape}  "
        f"range φ=[{pool_cv[:, 0].min():.3f},{pool_cv[:, 0].max():.3f}]  "
        f"ψ=[{pool_cv[:, 1].min():.3f},{pool_cv[:, 1].max():.3f}]",
        flush=True,
    )

    centers = make_phi_psi_centers(int(args.n_phi), int(args.n_psi))
    d2 = periodic_dist2(centers, pool_cv)
    pick = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(centers.shape[0]), pick])
    print(
        f"[match] windows={centers.shape[0]}  "
        f"|Δ|_torus mean/max={dist.mean():.4f}/{dist.max():.4f} rad",
        flush=True,
    )

    init = np.zeros((centers.shape[0], n_atoms, 3), dtype=np.float32)
    for i, ifr in enumerate(pick):
        init[i] = solvate_fixed_n(
            solute_traj[int(ifr)],
            box=box_len,
            n_waters=n_waters,
            density=float(args.density),
            min_dist=float(args.min_dist),
            seed=int(args.seed) + i,
            ref_waters=ref_waters,
        )
        if (i + 1) % 32 == 0 or i + 1 == centers.shape[0]:
            print(f"[solvate] {i + 1}/{centers.shape[0]}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, init)
    meta = out.with_suffix(".npz")
    np.savez(
        meta,
        centers=centers,
        pick_frame=pick.astype(np.int64),
        torus_dist=dist.astype(np.float64),
        pool_cv=pool_cv.astype(np.float32),
        torsions=tors,
        n_phi=int(args.n_phi),
        n_psi=int(args.n_psi),
        traj=str(args.traj_xyz),
        box=box_len,
        n_waters=n_waters,
    )
    print(f"[write] {out}  shape={init.shape}", flush=True)
    print(f"[write] {meta}", flush=True)


def _load_single_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=np.float64)


if __name__ == "__main__":
    main()
