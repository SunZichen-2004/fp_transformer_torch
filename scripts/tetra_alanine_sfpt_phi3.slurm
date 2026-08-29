#!/bin/bash
#SBATCH -J tetra_phi3
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job.%j.err

# Vacuum tetra-alanine 3φ sFPT, fixed λ=0.02, R=4096.
# Kernel/rescale = vacuum dialanine (main-picture panel a).
# Dump coords/CV/bandwidth + scatter every 200 steps.
# Every 10 ps: torus-KDE 2D FES on last 10 ps vs REMD (φi, φj).

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

SEED=${SEED:-3}
REPLICAS=${REPLICAS:-1024}
TOTAL_TIME=${TOTAL_TIME:-10000}
DUMP_EVERY=${DUMP_EVERY:-200}
LAMBDA=${LAMBDA:-0.05}
UPDATE_PACE=${UPDATE_PACE:-20}

EXP="$ROOT/experiments/tetra_alanine_sFPT"
OUT_DIR="$EXP/results/tetra_alanine_phi3_r${REPLICAS}_s${SEED}_${SLURM_JOB_ID}"
INIT_NPY="$EXP/init_coords_${REPLICAS}.npy"
REMD="$EXP/remd_280_400_n64_90605"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "seed=$SEED R=$REPLICAS t=${TOTAL_TIME}ps dump_every=$DUMP_EVERY λ=$LAMBDA"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

if [[ ! -f "$INIT_NPY" ]]; then
  echo "[error] missing $INIT_NPY — load existing MD init npy first" >&2
  exit 1
fi
echo "[init] using $INIT_NPY"
mkdir -p "$OUT_DIR"

python -u "$EXP/sample.py" \
  --sponge-prefix "$EXP/tetra_alanine" \
  --cv-file "$EXP/backbone_phi.txt" \
  --init-npy "$INIT_NPY" \
  --replicas "$REPLICAS" \
  --total-time "$TOTAL_TIME" \
  --dump-every "$DUMP_EVERY" \
  --seed "$SEED" \
  --lambda "$LAMBDA" \
  --update-pace "$UPDATE_PACE" \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.0 \
  --compile \
  --temperature 300 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 4.0 \
  --plot-every 10 \
  --remd-dir "$REMD" \
  --fes-every-ps 10 \
  --fes-window-ps 10 \
  --fes-vmax 16 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
ls -lt "${OUT_DIR}" | head -15
ls -lt "${OUT_DIR}/fes_2d" 2>/dev/null | head -15
