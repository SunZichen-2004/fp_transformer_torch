#!/bin/bash
#SBATCH -J dial_conv_R
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# convergence_1024 (87297) + convergence_2048 (87264)
# same MetaD/US refs as convergence_320; then compare ΔG across R

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

US_MBAR=$ROOT/experiments/dialanine_water_us/results/dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz
META1=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s2026_87374/live
META2=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s2025_87375/live
METAD_DIRS="${META1},${META2}"

FPT1024=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r1024_s0_87297
OUT1024=$ROOT/experiments/dialanine_water/sFPT/convergence_1024
FPT2048=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r2048_s0_87264
OUT2048=$ROOT/experiments/dialanine_water/sFPT/convergence_2048
OUTCMP=$ROOT/experiments/dialanine_water/sFPT/convergence_R_compare

GROUP_SIZE=${GROUP_SIZE:-20}
DT_GROUP=${DT_GROUP:-2.0}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.0625}
COLLAGE_VMAX=${COLLAGE_VMAX:-20}
FES2D_VMAX=${FES2D_VMAX:-20}

# rates from earlier production (approx); 2048 uses λ=0.1
RATE1024=${RATE1024:-7.55}
RATE2048=${RATE2048:-4.0}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true
test -f "$US_MBAR"

echo "[1/3] convergence_1024 (λ=0.0625) ..."
mkdir -p "$OUT1024"
python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT1024" \
  --fpt-dirs "${FPT1024}" \
  --metad-dirs "${METAD_DIRS}" \
  --us-mbar "$US_MBAR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --lambda-target 0.0625 \
  --lambda-atol 1e-3 \
  --fpt-rate "$RATE1024" \
  --metad-rate 44.0 \
  --collage-vmax "$COLLAGE_VMAX" \
  --fes2d-vmax "$FES2D_VMAX" \
  --device cuda

echo "[2/3] convergence_2048 (λ=0.1) ..."
mkdir -p "$OUT2048"
python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT2048" \
  --fpt-dirs "${FPT2048}" \
  --metad-dirs "${METAD_DIRS}" \
  --us-mbar "$US_MBAR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --lambda-target 0.1 \
  --lambda-atol 1e-3 \
  --fpt-rate "$RATE2048" \
  --metad-rate 44.0 \
  --collage-vmax "$COLLAGE_VMAX" \
  --fes2d-vmax "$FES2D_VMAX" \
  --device cuda

echo "[3/3] ΔG compare across R ..."
mkdir -p "$OUTCMP"
python -u "$ROOT/experiments/dialanine_water/plot_deltaG_vs_R.py" \
  --out-dir "$OUTCMP" \
  --npz-320 "$ROOT/experiments/dialanine_water/sFPT/convergence_320/fes_rmse_vs_time.npz" \
  --npz-512 "$ROOT/experiments/dialanine_water/sFPT/convergence_512/fes_rmse_vs_time.npz" \
  --npz-1024 "$OUT1024/fes_rmse_vs_time.npz" \
  --npz-2048 "$OUT2048/fes_rmse_vs_time.npz"

echo "done $(date)"
ls -lt "$OUT1024"/deltaG*.png "$OUT2048"/deltaG*.png "$OUTCMP"/*.png 2>/dev/null | head -30
