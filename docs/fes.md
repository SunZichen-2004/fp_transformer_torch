# FES reconstruction

The archive-native API is in `fp_transformer_torch.reconstruct_FES.archive`.

```python
import numpy as np
from fp_transformer_torch.reconstruct_FES.archive import reconstruct_sfpt
axis = np.linspace(-np.pi, np.pi, 72, endpoint=False)
fes = reconstruct_sfpt("trajectory.npz", [axis, axis],
                       start_ps=20, end_ps=25, periodic=True)
np.savez_compressed("fes.npz", **fes)
```

For three CVs, pass three axes. Joint density is reconstructed in three dimensions before calculating one- and two-dimensional marginal free energies. The flattened samples retain their individual saved bandwidths. Equal replica counts per frame make pooling equivalent to averaging the instantaneous joint KDEs.

Physical probability is proportional to the averaged joint KDE raised to `(1+lambda)/lambda`. Every returned FES is in kcal/mol and shifted to a minimum of zero. Angular grids should avoid duplicating the periodic endpoint. For nonperiodic distance CVs, use `periodic=False` and axes spanning both basins.

The convenience `delta_g_kcal_mol` follows the paper's sign convention, `kB*T*log(P1/P2)`. For two angular CVs, basin 1 has the first angle below zero; for the three-angle tetrapeptide, all three angles are below zero. For the two-distance reaction, basin 1 is the product region `d2-d1 < 0`. One-dimensional profiles split at zero. For other macrostate definitions, integrate the returned normalized `joint_log_probability` over the desired masks explicitly.

`reconstruct_metad` evaluates the saved final B-spline coefficients, or rebuilds the same bias from saved hills. `end_ps` can select an earlier saved hill count. It reconstructs the full joint physical probability from the cumulative bias before projecting. A two-dimensional slice of a three-dimensional bias is not a marginalized FES.

The retained neural fitter can be invoked with:

```bash
python -m fp_transformer_torch.reconstruct_FES.train_max_likelihood_torch --help
```

Its `--data_dir` accepts a directory containing `trajectory.npz`, an NPZ path, or legacy lambda-tagged arrays. Preparation frames and production step zero are excluded automatically. Select post-convergence data before fitting; a completeness flag only confirms trajectory completion, not statistical convergence.
