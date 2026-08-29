#!/bin/bash
#SBATCH -J dial_lam_fes
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/lambda_scan/convergence_analysis/fes_last50/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/lambda_scan/convergence_analysis/fes_last50/job.%j.err

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1
source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

OUT=$ROOT/experiments/dialanine_water/lambda_scan/convergence_analysis/fes_last50
mkdir -p "$OUT"

python -u "$ROOT/experiments/dialanine_water/plot_lambda_scan_fes_rmse.py" \
  --scan-dir "$ROOT/experiments/dialanine_water/lambda_scan" \
  --out-dir "$OUT" \
  --us-mbar "$ROOT/experiments/dialanine_water_us/results/dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz" \
  --last-frac 0.5 \
  --fes2d-vmax 20 \
  --fes1d-vmax 20 \
  --rmse-vmax 0 \
  --device cuda

echo "done $(date)"
ls -lt "$OUT"/*.{png,pdf,npz} 2>/dev/null | head -40
