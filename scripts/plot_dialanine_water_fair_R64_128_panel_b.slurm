#!/bin/bash
#SBATCH -J dial_fair_R_ab
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH --dependency=afterok:94291:94292:94293:94294

# After fair-compete R=64/128 (94291–94294) finish: 1D KDE FES + replica_ab panel b.

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

python -u "$ROOT/experiments/dialanine_water/replica_number_scan/compute_fair_R_fes.py" \
  --group-size 20 \
  --dt-group 2.0 \
  --max-groups 80 \
  --lambda-target 0.0625 \
  --device cuda

echo "done $(date)"
ls -lt "$ROOT/experiments/dialanine_water/replica_number_scan/replica_ab."{png,pdf} \
      "$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/convergence_64/"*.npz \
      "$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/convergence_128/"*.npz
