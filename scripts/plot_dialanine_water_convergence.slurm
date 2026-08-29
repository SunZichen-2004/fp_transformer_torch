#!/bin/bash
#SBATCH -J dial_wat_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Dialanine-water convergence
#   FPT=... OUT=.../convergence FPT_RATE=4.0 RMSE_VMAX_EXTRA=0 \
#   FES2D_VMAX=100 COLLAGE_VMAX=100 sbatch scripts/plot_dialanine_water_convergence.slurm

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

FPT=${FPT:-$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r1024_s0_87246}
METAD=${METAD:-$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s0_87241/live}
OUT=${OUT:-$FPT/convergence}
GROUP_SIZE=${GROUP_SIZE:-10}
DT_GROUP=${DT_GROUP:-1.0}
FPT_RATE=${FPT_RATE:-7.59}
METAD_RATE=${METAD_RATE:-50.0}
RMSE_VMAX_EXTRA=${RMSE_VMAX_EXTRA:-0}
COLLAGE_VMAX=${COLLAGE_VMAX:-0}
FES2D_VMAX=${FES2D_VMAX:-100}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.1}
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "FPT=$FPT"
echo "METAD=$METAD"
echo "OUT=$OUT group=$GROUP_SIZE dt_group=$DT_GROUP rate=$FPT_RATE λ=$LAMBDA_TARGET collage_vmax=$COLLAGE_VMAX"
nvidia-smi || true
ls "$FPT"/stage*_lambda*_dihedral.npy | wc -l
ls -lah "$METAD"/hills.npz "$METAD"/fes.npz

python -u "$ROOT/experiments/dialanine_water/plot_convergence_analysis.py" \
  --fpt-dir "$FPT" \
  --metad-dir "$METAD" \
  --out-dir "$OUT" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --dt-stage 0.1 \
  --device cuda \
  --fpt-rate "$FPT_RATE" \
  --metad-rate "$METAD_RATE" \
  --fpt-gpus 1 \
  --metad-gpus 1 \
  --bias-factor 15.0 \
  --sigma 0.35,0.35 \
  --metad-stride-deposits 10 \
  --metad-ref-last-ps 0 \
  --t-max 0 \
  --last-frac 0.2 \
  --fes2d-vmax "$FES2D_VMAX" \
  --collage-vmax "$COLLAGE_VMAX" \
  --rmse-vmax-extra "$RMSE_VMAX_EXTRA" \
  --vmax-kcal 0 \
  --lambda-target "$LAMBDA_TARGET" \
  --xi-lo -3.141592653589793 \
  --xi-hi 3.141592653589793 \
  --xi-cut 0

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -40
