#!/bin/bash
#SBATCH -J g5_1d_grid
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT/scripts" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

OUT=${OUT:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736/kde_fes/rmse_vs_time_g5}
echo "host=$(hostname) date=$(date) OUT=$OUT"
nvidia-smi || true

python -u "$ROOT/experiments/rearrangement_water_fpt/plot_g5_1d_grid_and_dg_zoom.py" \
  --out-dir "$OUT" \
  --group-size 5 \
  --dt-group 2.5 \
  --bounds 1.0,4.0,1.0,4.0 \
  --grid-n 120 \
  --device cuda \
  --ncols 4

echo "done $(date)"
