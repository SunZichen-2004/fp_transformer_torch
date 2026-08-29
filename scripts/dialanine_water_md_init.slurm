#!/bin/bash
#SBATCH -J dial_wat_md
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Unbiased NVT MD (GAFF+TIP3P) → init_coords_{N_INIT}.npy for dialanine_water sFPT
#
#   STEPS=10000 DUMP_EVERY=3 DISCARD=2000 N_INIT=1024 sbatch scripts/dialanine_water_md_init.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

STEPS=${STEPS:-10000}
DUMP_EVERY=${DUMP_EVERY:-3}
DISCARD=${DISCARD:-2000}
N_INIT=${N_INIT:-1024}
SEED=${SEED:-0}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}

EXP="$ROOT/experiments/dialanine_water"
OUT="$EXP/results/md_init_s${SEED}_${SLURM_JOB_ID}"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "MD steps=$STEPS dump=$DUMP_EVERY discard=$DISCARD n_init=$N_INIT box=$BOX_DIR"
nvidia-smi || true
head -2 "$BOX_DIR/box.xyz" || true

python -u "$EXP/run_md_init.py" \
  --box-dir "$BOX_DIR" \
  --cv-file "$BOX_DIR/backbone_cv.txt" \
  --steps "$STEPS" \
  --dump-every "$DUMP_EVERY" \
  --discard-steps "$DISCARD" \
  --n-init "$N_INIT" \
  --seed "$SEED" \
  --pme-cutoff 8.0 \
  --device cuda \
  --out-dir "$OUT"

echo "done $(date)"
ls -lt "$OUT" "$EXP"/init_coords_${N_INIT}.npy
