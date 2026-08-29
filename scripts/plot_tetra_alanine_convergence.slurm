#!/bin/bash
#SBATCH -J tetra_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job_conv.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job_conv.%j.err

# Tetra-alanine: 6D torus-KDE FES (average P in a group) → Boltzmann project to 1D/2D.
# REMD 300 K replica uses the same 6D grid and projection.

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

FPT=${FPT:-$ROOT/experiments/tetra_alanine_sFPT/results/tetra_alanine_sfpt_r1024_s0_88562}
REMD=${REMD:-$ROOT/experiments/tetra_alanine_sFPT/remd_280_400_n64_90605}
OUT=${OUT:-$FPT/convergence}
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "FPT=$FPT"
echo "REMD=$REMD"
ls "$FPT"/stage*_dihedral.npy | wc -l
nvidia-smi || true

python -u "$ROOT/experiments/tetra_alanine_sFPT/plot_convergence_analysis.py" \
  --fpt-dir "$FPT" \
  --remd-dir "$REMD" \
  --out-dir "$OUT" \
  --lambda-target 0.08 \
  --group-size 5 \
  --dt-group 1.0 \
  --last-frac 0.5 \
  --n-grid-6d 10 \
  --vmax 8 \
  --rmse-vmax 8 \
  --device cuda

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz,pdf} 2>/dev/null | head -40
