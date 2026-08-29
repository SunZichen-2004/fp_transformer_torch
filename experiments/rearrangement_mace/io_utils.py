"""I/O helpers for allyl phenyl ether Claisen rearrangement."""

from __future__ import annotations

from pathlib import Path

import numpy as np

Z_TO_SYMBOL = {1: "H", 6: "C", 8: "O"}
SYMBOL_TO_Z = {v: k for k, v in Z_TO_SYMBOL.items()}
SYMBOL_MASS = {"H": 1.008, "C": 12.011, "O": 15.999}


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


def atomic_numbers_from_symbols(symbols: list[str]) -> np.ndarray:
    zs = []
    for s in symbols:
        key = s.upper()
        if key not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element {s}")
        zs.append(SYMBOL_TO_Z[key])
    return np.asarray(zs, dtype=np.int64)


def masses_from_symbols(symbols: list[str]) -> np.ndarray:
    return np.asarray([SYMBOL_MASS[s.upper()] for s in symbols], dtype=np.float64)


def _scatter2d(x, y, save_path: Path, *, title: str, xlabel: str, ylabel: str, color="C0"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skip plot {save_path}: {exc}", flush=True)
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c=color, s=3.0, alpha=0.5, linewidths=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def plot_cv12(cv: np.ndarray, save_path: Path, *, title: str) -> None:
    arr = np.asarray(cv, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    _scatter2d(
        arr[:, 0],
        arr[:, 1],
        save_path,
        title=title,
        xlabel=r"$d_1$(O7–C8) / Å",
        ylabel=r"$d_2$(C1–C10) / Å",
    )


def plot_d12(d: np.ndarray, save_path: Path, *, title: str) -> None:
    plot_cv12(d, save_path, title=title)


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "Claisen",
    max_frames: int | None = None,
    append: bool = False,
) -> None:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if max_frames is not None:
        arr = arr[: int(max_frames)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.is_file() else "w"
    n_atoms = arr.shape[1]
    with path.open(mode, encoding="utf-8") as fh:
        for i, frame in enumerate(arr):
            fh.write(f"{n_atoms}\n")
            fh.write(f"{comment_prefix} frame={i}\n")
            for sym, (x, y, z) in zip(symbols, frame, strict=True):
                fh.write(f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")
    print(f"[xyz] wrote {path} frames={arr.shape[0]} append={append}", flush=True)


def save_stage_artifacts(
    out_dir: Path,
    tag: str,
    coords: np.ndarray,
    cv: np.ndarray,
    symbols: list[str],
    *,
    title: str,
    distances: np.ndarray | None = None,
    max_xyz_frames: int | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "d1_d2").mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=np.float32)
    cv = np.asarray(cv, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    np.save(out_dir / f"{tag}_coords.npy", coords)
    np.save(out_dir / f"{tag}_cv.npy", cv)
    write_xyz(out_dir / f"{tag}.xyz", coords, symbols, comment_prefix=title, max_frames=max_xyz_frames)
    plot_cv12(cv, out_dir / "d1_d2" / f"{tag}_d1_d2.png", title=title)
    if distances is not None:
        d = np.asarray(distances, dtype=np.float64)
        if d.ndim == 3:
            d = d.reshape(d.shape[0], -1)
        np.save(out_dir / f"{tag}_d.npy", d)
