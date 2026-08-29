"""Helpers for SPONGE text inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_sponge_coordinates(prefix: str | Path, num_atoms: int) -> np.ndarray:
    """Load ``prefix_coordinate.txt`` as ``(num_atoms, 3)``.

    Skips a leading atom-count line and ignores trailing box lines.
    """
    path = Path(str(prefix) + "_coordinate.txt")
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    start = 1 if len(lines[0].split()) == 1 else 0
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) >= int(num_atoms):
            break
    coords = np.asarray(rows, dtype=np.float64)
    if coords.shape != (int(num_atoms), 3):
        raise ValueError(
            f"{path}: expected ({num_atoms}, 3) coordinates, got {coords.shape}"
        )
    return coords
