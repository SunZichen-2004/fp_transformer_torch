#!/bin/bash
#SBATCH -J dial_main_pic
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/dialanine_main_picture/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/dialanine_main_picture/job.%j.err

# Water e/f/g: keep existing last-50% 2D cache + 122–140 ps 1D window.
# h: convergence_1024 ΔG (91354/91355), xlim = [0, MetaD wall-clock max].

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
OUT=$ROOT/experiments/dialanine_water/dialanine_main_picture
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$OUT"
echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na} cuda=$(python -c 'import torch; print(torch.cuda.is_available())')"

python -u "$OUT/plot_main_picture.py"

echo "done $(date)"
ls -lt "$OUT"/*.{png,pdf,npz} 2>/dev/null | head -20
