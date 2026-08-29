#!/bin/bash
# Dialanine GAFF BAOAB-FP smoke: JAX (uv) dump → torch (conda MACE) compare.
# Same init npy / SPONGE FF / shared OU noise → first N steps must match (default 200).

set -eo pipefail

TORCH_ROOT=/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch
JAX_ROOT=/share/home/sunzc/work/Fokker_Plank/fp_0429_final/fp-transformer
REF_DIR="$TORCH_ROOT/experiments/dialanine/parity_ref"
DUMP_PY="$TORCH_ROOT/scripts/dump_jax_dialanine_parity.py"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
unset PYTHONPATH
# Login nodes often lack system cuSPARSE; point at the uv venv NVIDIA wheels.
VENV_SITE="$JAX_ROOT/.venv/lib/python3.11/site-packages"
if [ -d "$VENV_SITE/nvidia/cusparse/lib" ]; then
  export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/nvjitlink/lib:${VENV_SITE}/nvidia/cusparse/lib:${VENV_SITE}/nvidia/cublas/lib:${VENV_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
fi
export CUDA_VISIBLE_DEVICES=""

echo "=== [1/2] JAX dump via uv ($(date)) ==="
cd "$JAX_ROOT"
uv run python "$DUMP_PY" \
  --sponge-prefix "$TORCH_ROOT/experiments/dialanine/dialanine" \
  --init-npy "$TORCH_ROOT/experiments/dialanine/init_coords_2048.npy" \
  --cv-file "$TORCH_ROOT/experiments/dialanine/backbone_cv.txt" \
  --out-dir "$REF_DIR" \
  --replicas 8 \
  --steps 200 \
  --seed 0 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --rew-lambda-init 0.05 \
  --bw-update-pace 20 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 1.0

echo "=== [2/2] torch compare via conda MACE ($(date)) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate MACE
set +u
cd "$TORCH_ROOT"
export PYTHONPATH="$TORCH_ROOT/src:$TORCH_ROOT/tests:${PYTHONPATH:-}"
python -u "$TORCH_ROOT/scripts/compare_torch_dialanine_parity.py" --ref-dir "$REF_DIR"

echo "=== smoke OK ($(date)) ==="
