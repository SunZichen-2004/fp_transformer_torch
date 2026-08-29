"""Minimal I/O for solvated dialanine (GAFF + TIP3P)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8}
Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}


def load_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) != n:
        raise ValueError(f"xyz atom count mismatch in {path}")
    return symbols, np.asarray(coords, dtype=np.float64)


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "",
    append: bool = False,
    max_frames: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        frames = arr[None, ...]
    elif arr.ndim == 3:
        frames = arr
    else:
        raise ValueError(f"coords must be (N,3) or (B,N,3), got {arr.shape}")
    if max_frames is not None and frames.shape[0] > int(max_frames):
        frames = frames[: int(max_frames)]
    mode = "a" if append else "w"
    with path.open(mode) as f:
        for i, frame in enumerate(frames):
            f.write(f"{frame.shape[0]}\n{comment_prefix} frame={i}\n")
            for a in range(frame.shape[0]):
                s = symbols[a] if a < len(symbols) else "X"
                x, y, z = frame[a]
                f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")


def plot_phi_psi(cv: np.ndarray, save_path: Path, *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skip plot {save_path}: {exc}", flush=True)
        return
    arr = np.asarray(cv, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(arr[:, 0], arr[:, 1], c="C3", s=2.0, alpha=0.55, linewidths=0)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel(r"$\phi$ / rad")
    plt.ylabel(r"$\psi$ / rad")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)
