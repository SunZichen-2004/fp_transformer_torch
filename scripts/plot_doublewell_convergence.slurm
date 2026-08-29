#!/bin/bash
#SBATCH -J dw_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/doublewell/job_conv.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/doublewell/job_conv.%j.err

# Double-well sFPT convergence (group_size=10), like dialanine_water convergence_320
#
#   FPT=.../doublewell_sfpt_r256_lam0.1_s0_88009 sbatch scripts/plot_doublewell_convergence.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

FPT=${FPT:-$ROOT/experiments/doublewell/results/doublewell_sfpt_r256_lam0.1_s0_88009}
OUT=${OUT:-$FPT/convergence}
GROUP_SIZE=${GROUP_SIZE:-10}
# each stage = 100 steps × dt=1e-3 → 0.1; group of 10 → dt_group=1.0
DT_GROUP=${DT_GROUP:-1.0}
BARRIER_H=${BARRIER_H:-10.0}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.1}
MAX_PANELS=${MAX_PANELS:-40}
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "FPT=$FPT"
echo "OUT=$OUT group=$GROUP_SIZE dt_group=$DT_GROUP h=$BARRIER_H λ=$LAMBDA_TARGET"
nvidia-smi || true
ls "$FPT"/stage*_x1.npy | wc -l

python -u "$ROOT/experiments/doublewell/plot_convergence_analysis.py" \
  --fpt-dir "$FPT" \
  --out-dir "$OUT" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --barrier-h "$BARRIER_H" \
  --temperature 1.0 \
  --kB 1.0 \
  --lambda-target "$LAMBDA_TARGET" \
  --max-panels "$MAX_PANELS" \
  --collage-vmax 20 \
  --rmse-vmax 12 \
  --device cuda

echo "done $(date)"
ls -lt "$OUT"/*.{png,pdf,npz} 2>/dev/null | head -30
