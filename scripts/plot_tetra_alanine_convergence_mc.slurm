#!/bin/bash
#SBATCH -J tetra_conv_mc
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job_conv_mc.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job_conv_mc.%j.err

# 6D torus-KDE → sample P^{1/λ} (κ → κ/λ) → subspace hist vs REMD.
# CPU only. Writes $FPT/convergence_mc/

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
export CUDA_VISIBLE_DEVICES=""
export WARP_ENABLE_CUDA=0

FPT=${FPT:-$ROOT/experiments/tetra_alanine_sFPT/results/tetra_alanine_sfpt_r1024_s0_88658}
REMD=${REMD:-$ROOT/experiments/tetra_alanine_sFPT/remd_280_400_n64_90605}
OUT=${OUT:-$FPT/convergence_mc}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.05}
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "FPT=$FPT OUT=$OUT λ=$LAMBDA_TARGET"
ls "$FPT"/stage*_dihedral.npy | wc -l

python -u "$ROOT/experiments/tetra_alanine_sFPT/plot_convergence_analysis_mc.py" \
  --fpt-dir "$FPT" \
  --remd-dir "$REMD" \
  --out-dir "$OUT" \
  --lambda-target "$LAMBDA_TARGET" \
  --last-frac 0.5 \
  --group-size 5 \
  --dt-group 1.0 \
  --n-mc 1048576 \
  --n-mc-group 262144 \
  --n-plot-1d 72 \
  --n-plot-2d 64 \
  --vmax 20 \
  --rmse-vmax 8

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz,pdf} 2>/dev/null | head -50
