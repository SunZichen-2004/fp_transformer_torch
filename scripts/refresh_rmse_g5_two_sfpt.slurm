#!/bin/bash
#SBATCH -J g5_two_sfpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Two sFPT runs (86736 + 86938) + original MetaD live/ + US ΔG reference

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

FPT_A=$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736
FPT_B=$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86938
OUT=$FPT_A/kde_fes/rmse_vs_time_g5
METAD=$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_86711/live
US=$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g4_86936/mbar/fes_mbar.npz

echo "host=$(hostname) date=$(date)"
nvidia-smi || true
echo "FPT_A stages=$(ls "$FPT_A"/stage*_cv.npy | wc -l)"
echo "FPT_B stages=$(ls "$FPT_B"/stage*_cv.npy | wc -l)"

python -u "$ROOT/experiments/rearrangement_water_fpt/plot_g5_two_sfpt.py" \
  --fpt-a "$FPT_A" \
  --fpt-b "$FPT_B" \
  --out-dir "$OUT" \
  --metad-dir "$METAD" \
  --us-mbar "$US" \
  --reuse-f1d-a "$OUT/fes_1d_all_groups.npz" \
  --device cuda \
  --metad-stride-deposits 10

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} | head -20
