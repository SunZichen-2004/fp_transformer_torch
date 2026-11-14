# sFPT: subspace Fokker–Planck Transformer

PyTorch implementation of ensemble sampling in a collective-variable (CV) subspace. The core repository contains BAOAB integration, CVs and distances, Gaussian density/score estimation, molecular force-model adapters, well-tempered metadynamics, and free-energy reconstruction.

Paper inputs, benchmark runners, reference data and figures live in the separate [sFPT-paper repository](https://github.com/SunZichen-2004/sFPT-paper).

## Install

Python 3.10 or newer is required.

```bash
pip install -e ".[dev,analysis]"
# Explicit-solvent systems and TorchSim integration:
pip install -e ".[simulation,mlip]"
pytest -q
```

Molecular mechanics input tables are supplied by the paper repository. MACE checkpoints are loaded on demand; the model files are not bundled here.

## Prescribed healing and fixed production strength

```python
integrator = BaoabFokkerPlanckIntegrator(
    energy_system, collective_variable, gaussian_kernel,
    timestep=dt, langevin_friction=gamma, beta0=beta0,
    production_lambda=0.1,
)
final = integrator.run(
    production_steps=25_000,
    heal_lambdas=[0.5, 0.2], heal_steps=500,
    dump_every=100, seed=0, on_frame=record_frame,
)
```

The callback is `record_frame(phase, step, lambda_value, state)`. Production starts at step zero after healing. The final partial interval is saved when the production length is not divisible by `dump_every`. The TorchSim backend exposes the same protocol, with its initial simulation state as the first positional argument.

`production_lambda` remains constant throughout production. `heal_lambdas` is an optional, explicitly prescribed preparation sequence. There is no ESS search, production annealing, or replica resampling.

The native BAOAB backend uses the units of its force model (AKMA for the supplied SPONGE/Amber tables). TorchSim adapters use eV, Å and ps. See `units.py` for the conversion functions.

## One complete NPZ per trajectory

`fp_transformer_torch.output.TrajectoryArchive` writes `trajectory.npz`, readable with `np.load(path, allow_pickle=False)`. It contains explicit frame, replica and CV axes; phase, step, time, lambda, per-replica bandwidth, temperatures, full coordinates by default, final coordinates, metadata, and a completeness flag. WT-MetaD runners include all hills and the final bias interpolation grid. Files are replaced atomically. A failed calculation writes an archive marked `complete=False`.

See [the output schema](docs/output.md) and [FES reconstruction](docs/fes.md).

## Numerical conventions

The default kernel retains the joint replica gradient used by the paper. Angular CVs use the torus distance `2 - 2 mean(cos(delta))`; nonperiodic CVs use mean squared Euclidean distance. Bandwidths belong to kernel centers. The documented raw-row-sum bandwidth rescaling is preserved; see [bandwidth updates](docs/sfpt_bandwidth_update.md).

The neural maximum-likelihood FES fitter is retained and accepts both consolidated NPZ archives and legacy lambda-tagged arrays. Archive-native KDE reconstruction averages joint densities, applies the physical `(1+lambda)/lambda` power, and then marginalizes. WT-MetaD reconstruction uses `-gamma V/(gamma-1)`.

## Migration from 0.1

Replace `lambda_init`, `lambda_max`, ESS thresholds and lambda-step parameters with `production_lambda`. Replace the old annealing `.run(stage_steps, ...)` call with `.run(production_steps, dump_every=..., ...)`. Use `on_frame` to collect results. The experimental resident multi-GPU BAOAB driver, JAX FES fitter, alternative local bandwidth rules and frozen-center score option have been removed; force-model sharding remains available. The kernel retains the paper's joint replica gradient and row-sum bandwidth rescaling.
