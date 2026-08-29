#!/bin/bash
#SBATCH -J claisen_wat_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Solvated Claisen WT-MetaD (ML/MM), compact ~22 Å TIP3P box
# Two seeds (same as 87059/87060 used in deltaG_vs_wallclock):
#   SEED=2 sbatch scripts/rearrangement_water_meta.slurm
#   SEED=3 sbatch scripts/rearrangement_water_meta.slurm
# --metad-max-hills 0 → unlimited (200000 used to abort ~156 ps / ~3 h).

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
SEED=${SEED:-3}
REPLICAS=${REPLICAS:-64}
STEPS=${STEPS:-10000000}
METAD_MAX_HILLS=${METAD_MAX_HILLS:-0}
TEMP=${TEMP:-300}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
DSF_ALPHA=${DSF_ALPHA:-0.2}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/mlmm_box}

echo "host=$(hostname) date=$(date) Claisen-water-MetaD MACE=$MACE_MODEL R=$REPLICAS T=$TEMP seed=$SEED max_hills=$METAD_MAX_HILLS box=$BOX_DIR elec=$ELECTROSTATICS mesh=$PME_MESH"
nvidia-smi || true
head -2 "$BOX_DIR/box.xyz" || true

EXP="$ROOT/experiments/rearrangement_water_metadynamics"
OUT_DIR="$EXP/results/claisen_water_metad_${MACE_MODEL}_r${REPLICAS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --box-dir "${BOX_DIR}" \
  --cv-file "$EXP/cv_pairs.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --charge 0.0 \
  --spin 1.0 \
  --pme-cutoff 9.0 \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --dsf-alpha "${DSF_ALPHA}" \
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
  --metad-max-hills "${METAD_MAX_HILLS}" \
  --dump-every 500 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)  live=${OUT_DIR}/live/"
