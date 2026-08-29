#!/bin/bash
#SBATCH -J meta_rep_dG
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water_meta/convergence_replica/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water_meta/convergence_replica/job.%j.err

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1
source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
python -u "$ROOT/experiments/dialanine_water_meta/convergence_replica/plot_deltaG_vs_time.py"
echo "done $(date)"
ls -lt "$ROOT/experiments/dialanine_water_meta/convergence_replica"/*.{png,pdf,npz} 2>/dev/null | head -20
