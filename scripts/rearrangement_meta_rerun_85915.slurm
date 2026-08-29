#!/bin/bash
#SBATCH -J claisen_meta_rr
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Resume Claisen gas-phase WT-MetaD from 85915 hills; write live_rerun/.
# --metad-max-hills 0 → unlimited hill buffer.

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
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

MACE_MODEL=${MACE_MODEL:-mh-0}
REPLICAS=${REPLICAS:-32}
STEPS=${STEPS:-500000}
PREV=$ROOT/experiments/rearrangement_metadynamic/results/claisen_metad_mh-0_r32_85915
# Bias state lives in hills.npz (fes.npz is only the plotted grid).
HILLS=${HILLS:-$PREV/live/hills.npz}
OUT_DIR=${OUT_DIR:-$PREV}

echo "host=$(hostname) date=$(date) Claisen-MetaD-RERUN from $HILLS → $OUT_DIR/live_rerun"
nvidia-smi || true
python - <<PY
import numpy as np
z=np.load("$HILLS")
print("resume n_hills=", int(z["n_hills"]), "centers", z["centers"].shape)
PY

EXP="$ROOT/experiments/rearrangement_metadynamic"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --xyz "$EXP/allyl_phenyl_ether.xyz" \
  --cv-file "$ROOT/experiments/rearrangement_mace/cv_pairs.txt" \
  --mass-file "$ROOT/experiments/rearrangement_mace/mass.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --resume-hills "${HILLS}" \
  --live-dir live_rerun \
  --metad-max-hills 0 \
  --charge 0.0 \
  --spin 1.0 \
  --wall-d-max 8.0 \
  --wall-k 10.0 \
  --softmin-a 10.0 \
  --softmin-d-wall 2.7 \
  --softmin-k 10.0 \
  --cell-length 40.0 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 350 \
  --metad-sigma 0.2,0.2 \
  --metad-height 0.02 \
  --metad-bias-factor 30.0 \
  --metad-deposit-pace 50 \
  --dump-every 500 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)  live=${OUT_DIR}/live_rerun/"
