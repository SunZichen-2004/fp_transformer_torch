#!/bin/bash
#SBATCH -J dial_wat_convM
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# 1) US-MBAR 2D on 87367
# 2) multi-seed convergence_320 and (optional) convergence_512
#    MetaD mean±std over 2 seeds (md_init npy, R=64):
#      89387 (s0), 89392 (s2)
#    1D ref: US-MBAR F(φ)
#    ONLY=320 sbatch ...   # skip 512 (default for a quick 320 redraw)
#    ONLY=all  sbatch ...  # also redraw 512 with the same MetaD seeds

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

US_DIR=$ROOT/experiments/dialanine_water_us/results/dialanine_water_us2d_r576_k100_s0_87367
US_MBAR=$US_DIR/mbar/fes_mbar.npz
META1=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s0_89387/live
META2=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s2_89392/live
METAD_DIRS="${META1},${META2}"
ONLY=${ONLY:-320}

FPT320_0=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g1_s0_87371
FPT320_1=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g1_s1_87372
FPT320_2=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g1_s2_87373
OUT320=$ROOT/experiments/dialanine_water/sFPT/convergence_320

FPT512_0=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r512_g1_s0_87344
FPT512_1=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r512_g1_s2_87368
FPT512_2=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r512_g1_s4_87369
OUT512=$ROOT/experiments/dialanine_water/sFPT/convergence_512

GROUP_SIZE=${GROUP_SIZE:-20}
DT_GROUP=${DT_GROUP:-2.0}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.0625}
COLLAGE_VMAX=${COLLAGE_VMAX:-20}
FES2D_VMAX=${FES2D_VMAX:-20}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na} ONLY=$ONLY"
echo "METAD_DIRS=$METAD_DIRS"
nvidia-smi || true

if [[ ! -f "$US_MBAR" ]]; then
  echo "[1/3] US-MBAR 2D ..."
  python -u "$ROOT/experiments/dialanine_water_us/mbar_fes_2d.py" \
    --result-dir "$US_DIR" \
    --out-dir "$US_DIR/mbar" \
    --n-bins 72 \
    --n-bins-1d 72 \
    --discard-frac 0.2
else
  echo "[1/3] reuse $US_MBAR"
fi

if [[ "$ONLY" == "320" || "$ONLY" == "all" ]]; then
echo "[2/3] convergence_320 ..."
mkdir -p "$OUT320"
python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT320" \
  --fpt-dirs "${FPT320_0},${FPT320_1},${FPT320_2}" \
  --metad-dirs "${METAD_DIRS}" \
  --us-mbar "$US_MBAR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --lambda-target "$LAMBDA_TARGET" \
  --fpt-rate 35.0 \
  --metad-rate 44.0 \
  --fpt-gpus 4 \
  --collage-vmax "$COLLAGE_VMAX" \
  --fes2d-vmax "$FES2D_VMAX" \
  --device cuda
fi

if [[ "$ONLY" == "512" || "$ONLY" == "all" ]]; then
echo "[3/3] convergence_512 ..."
mkdir -p "$OUT512"
python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT512" \
  --fpt-dirs "${FPT512_0},${FPT512_1},${FPT512_2}" \
  --metad-dirs "${METAD_DIRS}" \
  --us-mbar "$US_MBAR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --lambda-target "$LAMBDA_TARGET" \
  --fpt-rate 13.8 \
  --metad-rate 44.0 \
  --collage-vmax "$COLLAGE_VMAX" \
  --fes2d-vmax "$FES2D_VMAX" \
  --device cuda
fi

echo "done $(date)"
ls -lt "$US_DIR/mbar" | head -15
ls -lt "$OUT320"/*.{png,npz} 2>/dev/null | head -25
ls -lt "$OUT512"/*.{png,npz} 2>/dev/null | head -25
