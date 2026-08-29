#!/bin/bash
#SBATCH -J dial_fair_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Fair-compete sFPT (single seed) vs the same MetaD/US refs as
# experiments/dialanine_water/sFPT/convergence_320
#   90465 R=256 → fair_compete_result/convergence_256
#   90464 R=320 → fair_compete_result/convergence_320

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
META1=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s0_89387/live
META2=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s2_89392/live
METAD_DIRS="${META1},${META2}"

FPT256=$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/dialanine_water_sfpt_r256_g1_s1_90465
OUT256=$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/convergence_256
FPT320=$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/dialanine_water_sfpt_r320_g1_s1_90464
OUT320=$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/convergence_320

GROUP_SIZE=${GROUP_SIZE:-50}
DT_GROUP=${DT_GROUP:-5.0}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.0625}
COLLAGE_VMAX=${COLLAGE_VMAX:-20}
FES2D_VMAX=${FES2D_VMAX:-20}
RATE256=${RATE256:-23.3}
RATE320=${RATE320:-19.75}
METAD_RATE=${METAD_RATE:-44.0}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true
test -f "$US_MBAR"
echo "n_stage 256=$(ls "$FPT256"/stage*_lambda*_dihedral.npy | wc -l)"
echo "n_stage 320=$(ls "$FPT320"/stage*_lambda*_dihedral.npy | wc -l)"

echo "[1/2] convergence_256 (90465) ..."
mkdir -p "$OUT256"
python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT256" \
  --fpt-dirs "${FPT256}" \
  --metad-dirs "${METAD_DIRS}" \
  --us-mbar "$US_MBAR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --lambda-target "$LAMBDA_TARGET" \
  --lambda-atol 1e-3 \
  --fpt-rate "$RATE256" \
  --metad-rate "$METAD_RATE" \
  --collage-vmax "$COLLAGE_VMAX" \
  --fes2d-vmax "$FES2D_VMAX" \
  --device cuda

echo "[2/2] convergence_320 (90464) ..."
mkdir -p "$OUT320"
python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT320" \
  --fpt-dirs "${FPT320}" \
  --metad-dirs "${METAD_DIRS}" \
  --us-mbar "$US_MBAR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --lambda-target "$LAMBDA_TARGET" \
  --lambda-atol 1e-3 \
  --fpt-rate "$RATE320" \
  --metad-rate "$METAD_RATE" \
  --collage-vmax "$COLLAGE_VMAX" \
  --fes2d-vmax "$FES2D_VMAX" \
  --device cuda

echo "done $(date)"
ls -lt "$OUT256"/*.{png,npz} 2>/dev/null | head -20
ls -lt "$OUT320"/*.{png,npz} 2>/dev/null | head -20
