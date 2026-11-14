# Trajectory archive schema, version 1

All arrays are pickle-free. `T` is the number of recorded frames, `R` the replica count, `D` the CV dimension, and `N` the atom count.

| Key | Shape | Meaning |
| --- | --- | --- |
| schema_version | scalar | Integer 1 |
| complete | scalar | True only after all requested production steps finish |
| metadata_json | scalar Unicode | Resolved runner parameters and seed |
| phase | T | `heal` or `production` |
| step, time_ps | T | Step/time within each phase; production starts at zero |
| lambda_value | T | Prescribed sFPT lambda; NaN for metadynamics |
| cv | T,R,D | Full ensemble CV trajectory |
| bandwidth | T,R | Per-center sFPT sigma; NaN for metadynamics |
| temperature_K | T | Ensemble mean kinetic temperature, when available |
| coordinates | T,R,N,3 | Full coordinates, float32, unless explicitly disabled |
| final_coordinates | R,N,3 | Most recent coordinates at native precision |
| hills_count | T | Number of accumulated hills at a saved checkpoint |

WT-MetaD archives also contain `hills_centers`, `hills_heights`, `hills_sigma`, `bias_factor`, and, when interpolation is enabled, `bias_grid`, `bias_coefficients`, `bias_grid_origin`, and `bias_grid_period`. Hills use the eV/Å/ps convention of the TorchSim driver. CV angles are in radians and reaction distances in Å.

Healing time is separate from production time. Each healing frame reports the cumulative step count over prescribed healing stages. The final partial production interval is always recorded.

```python
import numpy as np
with np.load("trajectory.npz", allow_pickle=False) as data:
    assert data["complete"]
    use = (data["phase"] == "production") & (data["time_ps"] > 0)
    cv = data["cv"][use]
    bandwidth = data["bandwidth"][use]
```
