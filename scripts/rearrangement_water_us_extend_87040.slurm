#!/bin/bash
#SBATCH -J claisen_us_ext
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Extra 1D US windows for 87040, same dx / κ / 100 ps.
#   SIDE=left   ξ < -3.5, steered from the ξ=−3.5 frame
#   SIDE=right  ξ > +2.5, steered from the ξ=+2.5 frame
#
#   SIDE=left  sbatch scripts/rearrangement_water_us_extend_87040.slurm
#   SIDE=right sbatch scripts/rearrangement_water_us_extend_87040.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

SIDE=${SIDE:?set SIDE=left or SIDE=right}
PARENT=${PARENT:-$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g1_87040}
EXT_DIR=${EXT_DIR:-$PARENT/extend}
MACE_MODEL=${MACE_MODEL:-mh-0}
STEPS=${STEPS:-100000}
HOP_PS=${HOP_PS:-0.5}
KAPPA_KCAL=${KAPPA_KCAL:-200.0}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/mlmm_box}
NGPUS=1

case "$SIDE" in
  left)
    CENTERS=$EXT_DIR/centers_left.npy
    STEER_START=$EXT_DIR/steer_start_left.npy
    ;;
  right)
    CENTERS=$EXT_DIR/centers_right.npy
    STEER_START=$EXT_DIR/steer_start_right.npy
    ;;
  *)
    echo "SIDE must be left or right, got: $SIDE" >&2
    exit 1
    ;;
esac
if [[ ! -f "$CENTERS" || ! -f "$STEER_START" ]]; then
  echo "missing $CENTERS or $STEER_START — run prepare_extend_87040.py first" >&2
  exit 1
fi
REPLICAS=$(python -c "import numpy as np; print(int(np.load('$CENTERS').reshape(-1).size))")

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na} SIDE=$SIDE R=$REPLICAS κ=${KAPPA_KCAL} parent=$(basename "$PARENT")"
nvidia-smi || true

EXP="$ROOT/experiments/rearrangement_water_US"
OUT_DIR="$EXP/results/claisen_water_us1d_${MACE_MODEL}_r${REPLICAS}_g${NGPUS}_${SIDE}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --box-dir "${BOX_DIR}" \
  --cv-file "$EXP/cv_pairs.txt" \
  --replicas "${REPLICAS}" \
  --centers-npy "${CENTERS}" \
  --steer-start-npy "${STEER_START}" \
  --steps "${STEPS}" \
  --relax-steps 0 \
  --hop-ps "${HOP_PS}" \
  --n-gpus "${NGPUS}" \
  --kappa-kcal "${KAPPA_KCAL}" \
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
  --update-pace 20 \
  --dump-every 500 \
  --timestep 0.001 \
  --langevin-friction 10.0 \
  --temperature 300 \
  --traj-solute-only \
  --max-xyz-frames 32 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "parent=$PARENT" >> "${OUT_DIR}/run_info.txt"
echo "extend_side=$SIDE" >> "${OUT_DIR}/run_info.txt"
echo "done $(date)  out=${OUT_DIR}"
