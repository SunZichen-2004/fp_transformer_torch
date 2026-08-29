#!/bin/bash
#SBATCH -J dial_wat_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Solvated dialanine (GAFF + TIP3P) φ/ψ WT-MetaD (multi-walker, period=2π)
# Same box as dialanine_water/sFPT: experiments/dialanine_water/mm_box
#
# Smoke:
#   STEPS=200 DUMP_EVERY=50 REPLICAS=8 sbatch scripts/dialanine_water_meta.slurm
# Production (2e6 steps @ dt=1 fs; 10 ps unbiased NVT before WT-MetaD):
#   PRE_MD_STEPS=10000 REPLICAS=256 SEED=1 INIT_NPY=/path/init_coords_1024.npy \
#     sbatch scripts/dialanine_water_meta.slurm
# PRE_MD_STEPS: unbiased NVT Langevin (no hills) before MetaD; default 10000 = 10 ps.

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

SEED=${SEED:-0}
REPLICAS=${REPLICAS:-320}
STEPS=${STEPS:-2000000}
PRE_MD_STEPS=${PRE_MD_STEPS:-10000}
DUMP_EVERY=${DUMP_EVERY:-500}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_CUTOFF=${PME_CUTOFF:-8.0}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}
INIT_NPY=${INIT_NPY:-}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=metad seed=$SEED R=$REPLICAS steps=$STEPS pre_md=$PRE_MD_STEPS elec=$ELECTROSTATICS box=$BOX_DIR init=${INIT_NPY:-box.xyz}"
nvidia-smi || true
head -2 "$BOX_DIR/box.xyz" || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

EXP="$ROOT/experiments/dialanine_water_meta"
OUT_DIR="$EXP/results/dialanine_water_metad_r${REPLICAS}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

INIT_ARGS=()
if [[ -n "$INIT_NPY" ]]; then
  INIT_ARGS+=(--init-npy "$INIT_NPY")
fi
if [[ "${PRE_MD_STEPS}" -gt 0 ]]; then
  INIT_ARGS+=(--pre-md-steps "${PRE_MD_STEPS}")
fi

python -u "$EXP/sample.py" \
  --box-dir "${BOX_DIR}" \
  --cv-file "${BOX_DIR}/backbone_cv.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --pme-cutoff "${PME_CUTOFF}" \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --metad-sigma 0.35,0.35 \
  --metad-height 0.02 \
  --metad-bias-factor 15.0 \
  --metad-deposit-pace 50 \
  --metad-max-hills 0 \
  --dump-every "${DUMP_EVERY}" \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}" \
  "${INIT_ARGS[@]}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
echo "live: ${OUT_DIR}/live/"
ls -lt "${OUT_DIR}" | head -25
ls -lt "${OUT_DIR}/live" | head -15
