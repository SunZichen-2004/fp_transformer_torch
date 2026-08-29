"""Shared DA I/O: CV scatter plots and multi-frame XYZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

Z_TO_SYMBOL = {1: "H", 6: "C", 8: "O"}
SYMBOL_TO_Z = {v: k for k, v in Z_TO_SYMBOL.items()}
SYMBOL_MASS = {"H": 1.008, "C": 12.011, "O": 15.999}


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
    plt.scatter(x, y, c=color, s=3.0, alpha=0.55, linewidths=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def plot_s1_s2(
    collective_variable: np.ndarray,
    save_path: Path,
    *,
    title: str,
) -> None:
    cv = np.asarray(collective_variable, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    if cv.shape[1] < 2:
        print(f"[warn] need ≥2 CVs for s1-s2, got shape {cv.shape}", flush=True)
        return
    _scatter2d(
        cv[:, 0],
        cv[:, 1],
        save_path,
        title=title,
        xlabel=r"$s_1 = c_1+c_2+c_3$",
        ylabel=r"$s_2 = c_2-c_3$",
        color="C0",
    )


def plot_c_pairs(
    c: np.ndarray,
    out_dir: Path,
    tag: str,
    *,
    title: str,
) -> None:
    """Scatter c1–c2, c1–c3, c2–c3 and derived s1–s2."""
    arr = np.asarray(c, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] < 3:
        print(f"[warn] need 3 soft-c for c-pair plots, got {arr.shape}", flush=True)
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = (
        (0, 1, "c1_c2", r"$c_1$", r"$c_2$", "C0"),
        (0, 2, "c1_c3", r"$c_1$", r"$c_3$", "C1"),
        (1, 2, "c2_c3", r"$c_2$", r"$c_3$", "C2"),
    )
    for i, j, name, xl, yl, col in pairs:
        _scatter2d(
            arr[:, i],
            arr[:, j],
            out_dir / f"{tag}_{name}.png",
            title=title,
            xlabel=xl,
            ylabel=yl,
            color=col,
        )
    s1 = arr[:, 0] + arr[:, 1] + arr[:, 2]
    s2 = arr[:, 1] - arr[:, 2]
    (out_dir.parent / "s1_s2").mkdir(parents=True, exist_ok=True)
    _scatter2d(
        s1,
        s2,
        out_dir.parent / "s1_s2" / f"{tag}_s1_s2.png",
        title=title + "  (from c)",
        xlabel=r"$s_1 = c_1+c_2+c_3$",
        ylabel=r"$s_2 = c_2-c_3$",
        color="C3",
    )


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "DA",
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
    cv_mode: str = "s12",
    max_xyz_frames: int | None = None,
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
    mode = str(cv_mode).lower()
    if mode in ("s12", "s1s2", "2d"):
        (out_dir / "s1_s2").mkdir(parents=True, exist_ok=True)
        plot_s1_s2(cv, out_dir / "s1_s2" / f"{tag}_s1_s2.png", title=title)
    else:
        # 3D soft-c: pairwise + derived s1–s2
        plot_c_pairs(cv, out_dir / "c_pairs", tag, title=title)
