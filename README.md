# fp_transformer_torch

Slim PyTorch port of Fokker–Planck + BAOAB sampling for:

- `particle_dimer` (custom Euclidean CV distance)
- alanine peptides via GAFF / SPONGE inputs (torus CV distance)

Keeps λ annealing, CV density estimation, and ESS-based λ selection from
`BAOABFPReweightAnneal`, but **does not** perform systematic replica resampling.

## Install

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```
