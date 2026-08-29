#!/bin/bash
#SBATCH -J tetra_mainfig
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/main_picture/job_mainfig.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/main_picture/job_mainfig.%j.err

# Main figure a–d from 3 sFPT seeds (94120/94121/94122) + 3 B-spline WT-MetaD
# seeds (94242/94243/94244). ΔG from 3D hist / hills; 2D Boltzmann mean; 1D/ΔG mean±std.

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

OUT=$ROOT/experiments/tetra_alanine_sFPT/main_picture

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python -u "$OUT/plot_main_picture.py" \
  --device cuda \
  --n-3d 25 \
  --cache "$OUT/multi_seed_cache.npz"

echo "done $(date)"
ls -lt "$OUT"/tetra_alanine_main_picture.{png,pdf} "$OUT"/multi_seed_cache.npz
