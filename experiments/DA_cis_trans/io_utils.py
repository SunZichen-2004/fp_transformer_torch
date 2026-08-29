"""I/O and scatter plots for DA_cis_trans (3D CV + c8–c17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

Z_TO_SYMBOL = {1: "H", 6: "C", 8: "O"}
SYMBOL_TO_Z = {v: k for k, v in Z_TO_SYMBOL.items()}
SYMBOL_MASS = {"H": 1.008, "C": 12.011, "O": 15.999}

CV_LABELS = (
    r"$cv_1$ (HLDA)",
    r"$cv_2=c_8\,c_{17}$",
)


def symbols_from_z(atomic_numbers: np.ndarray) -> list[str]:
    out: list[str] = []
    for z in np.asarray(atomic_numbers, dtype=np.int64).reshape(-1):
        if int(z) not in Z_TO_SYMBOL:
            raise ValueError(f"unsupported Z={z}")
        out.append(Z_TO_SYMBOL[int(z)])
    return out


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
        raise ValueError(f"xyz atom count mismatch in {path}: header={n}, got {len(symbols)}")
    return symbols, np.asarray(coords, dtype=np.float64)


def atomic_numbers_from_symbols(symbols: list[str]) -> np.ndarray:
    zs = []
    for s in symbols:
        key = s.upper() if s.upper() in SYMBOL_TO_Z else s
        if key not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element {s}")
        zs.append(SYMBOL_TO_Z[key])
    return np.asarray(zs, dtype=np.int64)


def masses_from_symbols(symbols: list[str]) -> np.ndarray:
    masses = []
    for s in symbols:
        key = s.upper() if s.upper() in SYMBOL_MASS else s
        if key not in SYMBOL_MASS:
            raise ValueError(f"no mass for {s}")
        masses.append(SYMBOL_MASS[key])
    return np.asarray(masses, dtype=np.float64)


def _scatter2d(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "C0",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c=color, s=3.0, alpha=0.45, linewidths=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def plot_cv_pair(
    cv: np.ndarray,
    i: int,
    j: int,
    save_path: Path,
    *,
    title: str,
    color: str = "C0",
) -> None:
    arr = np.asarray(cv, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] <= max(i, j):
        print(f"[warn] CV dim {arr.shape[1]} too small for axes {i},{j}", flush=True)
        return
    _scatter2d(
        arr[:, i],
        arr[:, j],
        save_path,
        title=title,
        xlabel=CV_LABELS[i] if i < len(CV_LABELS) else f"cv{i+1}",
        ylabel=CV_LABELS[j] if j < len(CV_LABELS) else f"cv{j+1}",
        color=color,
    )


def plot_c8_c17(
    c8: np.ndarray,
    c17: np.ndarray,
    save_path: Path,
    *,
    title: str,
) -> None:
    x = np.asarray(c8, dtype=np.float64).reshape(-1)
    y = np.asarray(c17, dtype=np.float64).reshape(-1)
    _scatter2d(
        x,
        y,
        save_path,
        title=title,
        xlabel=r"$c_8$ (rad)",
        ylabel=r"$c_{17}$ (rad)",
        color="C3",
    )


def plot_live_scatters(
    out_dir: Path,
    cv: np.ndarray,
    c8: np.ndarray,
    c17: np.ndarray,
    *,
    title: str,
) -> None:
    """Overwrite fixed live scatter paths (all accumulated points)."""
    out_dir = Path(out_dir)
    plot_cv_pair(cv, 0, 1, out_dir / "live" / "cv1_cv2.png", title=title)
    plot_c8_c17(c8, c17, out_dir / "live" / "c8_c17.png", title=title)


def plot_fes_projection(
    grid_i: np.ndarray,
    grid_j: np.ndarray,
    F: np.ndarray,
    save_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    F0 = F - np.nanmin(F)
    im = ax.contourf(grid_i, grid_j, F0, levels=30, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="F / eV")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {save_path}", flush=True)


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "DA_cis_trans",
    max_frames: int | None = None,
    append: bool = False,
) -> None:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"coords must be (F,N,3) or (N,3), got {arr.shape}")
    n_atoms = int(arr.shape[1])
    if len(symbols) != n_atoms:
        raise ValueError(f"symbols len={len(symbols)} != n_atoms={n_atoms}")
    if max_frames is not None:
        arr = arr[: int(max_frames)]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.is_file() else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for i, frame in enumerate(arr):
            fh.write(f"{n_atoms}\n")
            fh.write(f"{comment_prefix} frame={i}\n")
            for sym, (x, y, z) in zip(symbols, frame, strict=True):
                fh.write(f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")
    print(
        f"[xyz] wrote {path}  frames={arr.shape[0]} atoms={n_atoms} append={append}",
        flush=True,
    )


def save_stage_artifacts(
    out_dir: Path,
    tag: str,
    coords: np.ndarray,
    cv: np.ndarray,
    symbols: list[str],
    *,
    title: str,
    c8: np.ndarray | None = None,
    c17: np.ndarray | None = None,
    max_xyz_frames: int | None = None,
    plot: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = np.asarray(coords, dtype=np.float32)
    cv = np.asarray(cv, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)

    np.save(out_dir / f"{tag}_coords.npy", coords)
    np.save(out_dir / f"{tag}_cv.npy", cv)
    write_xyz(
        out_dir / f"{tag}.xyz",
        coords,
        symbols,
        comment_prefix=title,
        max_frames=max_xyz_frames,
    )
    if plot:
        (out_dir / "cv_pairs").mkdir(parents=True, exist_ok=True)
        (out_dir / "c8_c17").mkdir(parents=True, exist_ok=True)
        plot_cv_pair(cv, 0, 1, out_dir / "cv_pairs" / f"{tag}_cv1_cv2.png", title=title)
        if c8 is not None and c17 is not None:
            c8_np = np.asarray(c8, dtype=np.float64).reshape(-1)
            c17_np = np.asarray(c17, dtype=np.float64).reshape(-1)
            np.save(out_dir / f"{tag}_c8_c17.npy", np.stack([c8_np, c17_np], axis=-1))
            plot_c8_c17(
                c8_np,
                c17_np,
                out_dir / "c8_c17" / f"{tag}_c8_c17.png",
                title=title,
            )
