#!/bin/bash
#SBATCH -J tetra_conv3
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job_conv_phi3.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job_conv_phi3.%j.err

# Tetra-alanine 3φ convergence: 3 sFPT seeds vs 3 B-spline WT-MetaD seeds
# (94242/94243/94244). Last MetaD FES (seed-averaged) as reference.

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

OUT=${OUT:-$ROOT/experiments/tetra_alanine_sFPT/convergence_analysis}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "OUT=$OUT"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python -u "$ROOT/experiments/tetra_alanine_sFPT/plot_convergence_phi3.py" \
  --out "$OUT" \
  --device cuda \
  --dg-method hist \
  --n-3d 25 \
  --collage-every 2 \
  --collage-ncols 5 \
  --chunk 8192 \
  --meta-it-s 144

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz,pdf} 2>/dev/null | head -40
