#!/bin/bash
#SBATCH -J dw_last50
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-01:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/doublewell/job_lastfrac.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/doublewell/job_lastfrac.%j.err

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1
source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

FPT=${FPT:-$ROOT/experiments/doublewell/results/doublewell_sfpt_r512_lam0.2_s0_88489}
OUT=${OUT:-$FPT/convergence}
LAST_FRAC=${LAST_FRAC:-0.5}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.2}
BARRIER_H=${BARRIER_H:-10.0}
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "FPT=$FPT OUT=$OUT last_frac=$LAST_FRAC λ=$LAMBDA_TARGET"
ls "$FPT"/stage*_x1.npy | wc -l

python -u "$ROOT/experiments/doublewell/plot_convergence_analysis.py" \
  --fpt-dir "$FPT" \
  --out-dir "$OUT" \
  --only-last-frac \
  --last-frac "$LAST_FRAC" \
  --barrier-h "$BARRIER_H" \
  --lambda-target "$LAMBDA_TARGET" \
  --collage-vmax 20 \
  --rmse-vmax 12 \
  --device cuda

echo "done $(date)"
ls -lt "$OUT"/fes_1d_last_frac.* 2>/dev/null
