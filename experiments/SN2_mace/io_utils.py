"""Shared SN2 I/O: d1–d2 scatter plots and multi-frame XYZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

Z_TO_SYMBOL = {1: "H", 6: "C", 17: "Cl"}


def symbols_from_z(atomic_numbers: np.ndarray) -> list[str]:
    out: list[str] = []
    for z in np.asarray(atomic_numbers, dtype=np.int64).reshape(-1):
        if int(z) not in Z_TO_SYMBOL:
            raise ValueError(f"unsupported Z={z}")
        out.append(Z_TO_SYMBOL[int(z)])
    return out


def plot_d1_d2(
    collective_variable: np.ndarray,
    save_path: Path,
    *,
    title: str,
    d_max: float = 3.5,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return

    cv = np.asarray(collective_variable, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    if cv.shape[1] < 2:
        print(f"[warn] need ≥2 CVs for d1-d2, got shape {cv.shape}", flush=True)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(cv[:, 0], cv[:, 1], c="C0", s=3.0, alpha=0.55, linewidths=0)
    plt.axvline(d_max, color="0.4", ls="--", lw=0.8, alpha=0.7)
    plt.axhline(d_max, color="0.4", ls="--", lw=0.8, alpha=0.7)
    plt.xlim(1.0, 4.0)
    plt.ylim(1.0, 4.0)
    plt.xlabel(r"$d(\mathrm{Cl_{nuc}},\mathrm{C})$ / Å")
    plt.ylabel(r"$d(\mathrm{Cl_{lg}},\mathrm{C})$ / Å")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "SN2",
    max_frames: int | None = None,
) -> None:
    """Write multi-frame XYZ. ``coords`` is ``(F, N, 3)`` or ``(N, 3)``."""
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
    with path.open("w", encoding="utf-8") as fh:
        for i, frame in enumerate(arr):
            fh.write(f"{n_atoms}\n")
            fh.write(f"{comment_prefix} frame={i}\n")
            for sym, (x, y, z) in zip(symbols, frame, strict=True):
                fh.write(f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")
    print(f"[xyz] wrote {path}  frames={arr.shape[0]} atoms={n_atoms}", flush=True)


def save_stage_artifacts(
    out_dir: Path,
    tag: str,
    coords: np.ndarray,
    cv: np.ndarray,
    symbols: list[str],
    *,
    title: str,
    d_max: float = 3.5,
    max_xyz_frames: int | None = None,
) -> None:
    """Save npy + xyz + d1-d2 scatter for one stage/snapshot."""
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
        comment_prefix=f"{title}",
        max_frames=max_xyz_frames,
    )
    plot_d1_d2(
        cv,
        out_dir / "d1_d2" / f"{tag}_d1_d2.png",
        title=title,
        d_max=d_max,
    )
