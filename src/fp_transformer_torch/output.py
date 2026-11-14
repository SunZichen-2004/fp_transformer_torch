"""A complete, pickle-free trajectory in one atomically replaced NPZ archive."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


class TrajectoryArchive:
    """Store explicit frame/replica/CV axes, phase, time, lambda and final state.

    Coordinates are optional because solvent trajectories can be large. All
    frames include CVs; sFPT frames also include each replica's KDE bandwidth.
    NPZ strings are Unicode arrays and load safely with ``allow_pickle=False``.
    """

    def __init__(self, path, *, metadata=None, include_coordinates=True):
        self.path = Path(path)
        self.metadata = dict(metadata or {})
        self.include_coordinates = include_coordinates
        self.rows = []
        self.last_coordinates = None

    def append(
        self,
        phase,
        step,
        timestep_ps,
        lam,
        coordinates,
        cv,
        bandwidth=None,
        *,
        metrics=None,
        hills_count=0,
    ):
        coordinates = _numpy(coordinates)
        cv = _numpy(cv).reshape(coordinates.shape[0], -1)
        bw = (
            np.full((coordinates.shape[0],), np.nan)
            if bandwidth is None
            else _numpy(bandwidth).reshape(-1)
        )
        if bw.size == 1:
            bw = np.repeat(bw, coordinates.shape[0])
        if bw.shape != (coordinates.shape[0],):
            raise ValueError("bandwidth must be scalar or one value per replica")
        if not np.isfinite(cv).all() or not np.isfinite(coordinates).all():
            raise ValueError("nonfinite trajectory frame")
        if bandwidth is not None and (not np.isfinite(bw).all() or (bw <= 0).any()):
            raise ValueError("sFPT bandwidth must be finite and positive")
        if self.rows and cv.shape != self.rows[0]["cv"].shape:
            raise ValueError("replica/CV dimensions changed during trajectory")
        self.last_coordinates = coordinates
        self.rows.append(
            dict(
                phase=str(phase),
                step=int(step),
                time_ps=float(step) * float(timestep_ps),
                lambda_value=float(lam),
                cv=cv,
                bandwidth=bw,
                coordinates=coordinates.astype(np.float32)
                if self.include_coordinates
                else None,
                temperature_K=float(
                    (metrics or {}).get("kinetic_temperature_mean", np.nan)
                ),
                hills_count=int(hills_count),
            )
        )

    def save(self, *, complete=False, extra=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "schema_version": np.asarray(1),
            "complete": np.asarray(bool(complete)),
            "metadata_json": np.asarray(
                json.dumps(self.metadata, default=str, sort_keys=True)
            ),
        }
        if self.rows:
            for key in [
                "phase",
                "step",
                "time_ps",
                "lambda_value",
                "cv",
                "bandwidth",
                "temperature_K",
                "hills_count",
            ]:
                values[key] = np.asarray([row[key] for row in self.rows])
            if self.include_coordinates:
                values["coordinates"] = np.stack(
                    [row["coordinates"] for row in self.rows]
                )
            values["final_coordinates"] = self.last_coordinates
        values.update(extra or {})
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **values)
        tmp.replace(self.path)
