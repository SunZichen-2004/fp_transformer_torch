#!/bin/bash
#SBATCH -J dialanine_gaff
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Dialanine (vacuum) GAFF/SPONGE + BAOAB-FP (torch)
# Flow: pre-MD (unbiased) → λ-heal → ESS anneal
# Bandwidth rescale: unnormalized K, metric=row_sum (no /B)
#
# Overrides:
#   SEED=0 REPLICAS=2048 STEPS=1000 \
#   HEAL_LAMBDAS=0.1,0.05 HEAL_STEPS=500 PRE_MD_STEPS=100 \
#   sbatch scripts/dialanine_gaff.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

SEED=${SEED:-0}
REPLICAS=${REPLICAS:-2048}
STEPS=${STEPS:-100}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5,0.2}
HEAL_STEPS=${HEAL_STEPS:-500}
PRE_MD_STEPS=${PRE_MD_STEPS:-0}
LAMBDA_INIT=${LAMBDA_INIT:-0.1}
LAMBDA_MAX=${LAMBDA_MAX:-0.11}

EXP="$ROOT/experiments/dialanine"
OUT_DIR="$EXP/results/dialanine_gaff_s${SEED}_${SLURM_JOB_ID}"
INIT_NPY="$EXP/init_coords_${REPLICAS}.npy"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "seed=$SEED R=$REPLICAS steps=$STEPS PRE_MD=$PRE_MD_STEPS HEAL_LAMBDAS=$HEAL_LAMBDAS HEAL_STEPS=$HEAL_STEPS λ0=$LAMBDA_INIT"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# Tile SPONGE coords only if MD init missing (prefer experiments/dialanine/init_coords_*.npy from MD)
if [[ ! -f "$INIT_NPY" ]]; then
  echo "[init] WARNING: $INIT_NPY missing — tiling SPONGE coords (run scripts/dialanine_md_init.slurm first for MD init)"
  python -u - <<PY
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}/src")
from fp_transformer_torch import GaffPotential
from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates

exp = Path("${EXP}")
prefix = exp / "dialanine"
B = int("${REPLICAS}")
pot = GaffPotential(str(prefix))
base = load_sponge_coordinates(prefix, num_atoms=pot.num_atoms)
coords = np.repeat(base[None, ...], B, axis=0)
out = Path("${INIT_NPY}")
np.save(out, coords.astype(np.float64))
print(f"wrote {out} shape={coords.shape}", flush=True)
PY
else
  echo "[init] using existing $INIT_NPY"
fi

mkdir -p "$OUT_DIR"

python -u "$EXP/sample.py" \
  --sponge-prefix "$EXP/dialanine" \
  --cv-file "$EXP/backbone_cv.txt" \
  --init-npy "$INIT_NPY" \
  --replicas "$REPLICAS" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --pre-md-steps "$PRE_MD_STEPS" \
  --heal-lambdas "$HEAL_LAMBDAS" \
  --heal-steps "$HEAL_STEPS" \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 1.0 \
  --compile \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init "$LAMBDA_INIT" \
  --rew-lambda-max "$LAMBDA_MAX" \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.000000002 \
  --rew-min-lambda-step 0.000000001 \
  --rew-debug-reweight \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 4.0 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
echo "plots: ${OUT_DIR}/phi_psi/"
ls -lt "${OUT_DIR}" | head -25
