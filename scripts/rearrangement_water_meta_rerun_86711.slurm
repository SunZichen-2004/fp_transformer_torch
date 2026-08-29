#!/bin/bash
#SBATCH -J claisen_wat_meta_rr
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 7-00:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Resume solvated Claisen WT-MetaD from 86711 hills → live_rerun/
# Prior run stopped ~156k steps / 199680 hills at max_hills=200000.
# --metad-max-hills 0 → unlimited.

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
REPLICAS=${REPLICAS:-64}
# finish original 5e6 steps (~156250 already done)
STEPS=${STEPS:-4843750}
TEMP=${TEMP:-300}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/mlmm_box}
PREV=$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_86711
HILLS=${HILLS:-$PREV/live/hills.npz}
OUT_DIR=${OUT_DIR:-$PREV}

echo "host=$(hostname) date=$(date) Claisen-water-MetaD-RERUN from $HILLS → $OUT_DIR/live_rerun"
nvidia-smi || true
python - <<PY
import numpy as np
z=np.load("$HILLS")
print("resume n_hills=", int(z["n_hills"]), "centers", z["centers"].shape)
PY

EXP="$ROOT/experiments/rearrangement_water_metadynamics"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --box-dir "${BOX_DIR}" \
  --cv-file "$EXP/cv_pairs.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --resume-hills "${HILLS}" \
  --live-dir live_rerun \
  --metad-max-hills 0 \
  --charge 0.0 \
  --spin 1.0 \
  --pme-cutoff 9.0 \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --wall-d-max 8.0 \
  --wall-k 10.0 \
  --softmin-a 10.0 \
  --softmin-d-wall 2.4 \
  --softmin-k 10.0 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature "${TEMP}" \
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
