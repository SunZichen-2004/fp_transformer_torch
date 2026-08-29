"""Shared I/O / plotting for vacuum tetra-alanine sFPT."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CV_NAMES = ("phi1", "psi1", "phi2", "psi2", "phi3", "psi3")
PHI_NAMES = ("phi1", "phi2", "phi3")


def load_pdb_coords(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            rows.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    if not rows:
        raise ValueError(f"no ATOM records in {path}")
    return np.asarray(rows, dtype=np.float64)


def load_torsion_indices(path: Path) -> np.ndarray:
    """Load torsion quadruples; file is 1-based (same as dialanine)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split("#", 1)[0].replace(",", " ").split()
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


def as_cv_matrix(collective_variable: np.ndarray) -> np.ndarray:
    cv = np.asarray(collective_variable, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    elif cv.ndim == 1:
        cv = cv.reshape(1, -1)
    return cv


def plot_phi_psi(
    collective_variable: np.ndarray,
    save_path: Path,
    *,
    title: str,
) -> None:
    """Ramachandran (φi,ψi) if even ≥2 CVs; otherwise 3φ pairwise scatters."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return

    cv = as_cv_matrix(collective_variable)
    n_cv = cv.shape[1]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if n_cv == 3:
        pairs = ((0, 1), (1, 2), (0, 2))
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), squeeze=False)
        for ax, (i, j) in zip(axes[0], pairs):
            ax.scatter(cv[:, i], cv[:, j], c="C3", s=2.0, alpha=0.55, linewidths=0)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_xlabel(rf"$\phi_{i+1}$ (rad)")
            ax.set_ylabel(rf"$\phi_{j+1}$ (rad)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
        print(f"[plot] wrote {save_path}", flush=True)
        return

    n_res = n_cv // 2
    if n_res < 1:
        print(f"[warn] need ≥2 CVs for phi-psi, got shape {cv.shape}", flush=True)
        return
    fig, axes = plt.subplots(1, n_res, figsize=(4.2 * n_res, 4.0), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.scatter(cv[:, 2 * i], cv[:, 2 * i + 1], c="C3", s=2.0, alpha=0.55, linewidths=0)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(rf"$\phi_{i+1}$ (rad)")
        ax.set_ylabel(rf"$\psi_{i+1}$ (rad)")
        ax.set_title(f"ALA{i+1}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {save_path}", flush=True)
