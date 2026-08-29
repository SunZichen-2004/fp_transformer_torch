#!/bin/bash
#SBATCH -J dial_R_2grp
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Two replica-scan ΔG figures:
#   equalN : R×Δt = 1024×2 ps  (R=128 → 16 ps / group)
#   dt3ps  : Δt = 3 ps for every R

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true

python -u "$ROOT/experiments/dialanine_water/replica_number_scan/compute_two_groupings.py" \
  --schemes equalN,dt3ps \
  --t-max 160 \
  --lambda-target 0.0625 \
  --device cuda

echo "done $(date)"
ls -lt "$ROOT/experiments/dialanine_water/replica_number_scan/replica_ab_equalN."{png,pdf} \
      "$ROOT/experiments/dialanine_water/replica_number_scan/replica_ab_dt3ps."{png,pdf}
