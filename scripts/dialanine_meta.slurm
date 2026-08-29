#!/bin/bash
#SBATCH -J dialanine_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Dialanine (vacuum) GAFF + φ/ψ WT-MetaD (multi-walker, shared bias)
# Same FF/CV as experiments/dialanine; default R=64
#
# Overrides:
#   SEED=0 REPLICAS=64 STEPS=200000 sbatch scripts/dialanine_meta.slurm
# Smoke:
#   STEPS=200 DUMP_EVERY=50 sbatch scripts/dialanine_meta.slurm

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
REPLICAS=${REPLICAS:-64}
STEPS=${STEPS:-200000}
DUMP_EVERY=${DUMP_EVERY:-500}

DIAL="$ROOT/experiments/dialanine"
EXP="$ROOT/experiments/dialanine_meta"
OUT_DIR="$EXP/results/dialanine_metad_r${REPLICAS}_s${SEED}_${SLURM_JOB_ID}"
INIT_NPY=${INIT_NPY:-$DIAL/init_coords_2048.npy}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=metad seed=$SEED R=$REPLICAS steps=$STEPS"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

mkdir -p "$OUT_DIR"

python -u "$EXP/sample.py" \
  --sponge-prefix "$DIAL/dialanine" \
  --cv-file "$DIAL/backbone_cv.txt" \
  --pdb "$DIAL/dialanine.pdb" \
  --mass-file "$DIAL/dialanine_mass.txt" \
  --init-npy "$INIT_NPY" \
  --replicas "$REPLICAS" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --metad-sigma 0.35,0.35 \
  --metad-height 0.02 \
  --metad-bias-factor 15.0 \
  --metad-deposit-pace 50 \
  --metad-max-hills 0 \
  --dump-every "$DUMP_EVERY" \
  --dtype float64 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
echo "live: ${OUT_DIR}/live/"
ls -lt "${OUT_DIR}" | head -25
ls -lt "${OUT_DIR}/live" | head -15
