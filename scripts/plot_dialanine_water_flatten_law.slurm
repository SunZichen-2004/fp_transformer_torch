#!/bin/bash
#SBATCH -J dial_flat_law
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/lambda_scan/flatten_law_validation/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/lambda_scan/flatten_law_validation/job.%j.err

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1
source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

OUT=$ROOT/experiments/dialanine_water/lambda_scan/flatten_law_validation
mkdir -p "$OUT"

python -u "$ROOT/experiments/dialanine_water/plot_flatten_law_validation.py" \
  --out-dir "$OUT" \
  --fes-npz "$ROOT/experiments/dialanine_water/lambda_scan/convergence_analysis/fes_last50/fes_rmse_by_lambda.npz" \
  --f-cut 16 \
  --last-frac 0.5

echo "done $(date)"
ls -lt "$OUT"/*.{png,pdf,npz} 2>/dev/null | head -40
