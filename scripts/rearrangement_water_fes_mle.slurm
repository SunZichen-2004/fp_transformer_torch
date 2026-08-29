#!/bin/bash
#SBATCH -J claisen_water_fes_mle
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Train Torch MLE FES from water Claisen FPT CV npy, compare to WT-MetaD fes.npz.
#
# Defaults: DATA=86736 water FPT, METAD=86711 water MetaD, LAST_STAGES=3.
#
#   sbatch rearrangement_water_fes_mle.slurm
#   LAST_STAGES=10 sbatch --export=ALL,LAST_STAGES rearrangement_water_fes_mle.slurm
#   DATA=/path/to/run LAST_STAGES=5 sbatch --export=ALL,DATA,LAST_STAGES rearrangement_water_fes_mle.slurm
#   LAST_STAGES=0 MAX_STAGES=16  → use first 16 stages instead of last-N

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT/scripts" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

# --- water Claisen defaults ---
DATA=${DATA:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736}
METAD_FES=${METAD_FES:-$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_86711/live/fes.npz}

STEPS=${STEPS:-30000}
# LAST_STAGES>0 → take last N stage*_cv.npy; LAST_STAGES=0 → first MAX_STAGES via FILE_GLOB
LAST_STAGES=${LAST_STAGES:-12}
MAX_STAGES=${MAX_STAGES:-16}
FILE_GLOB=${FILE_GLOB:-stage[0-9]*_cv.npy}

# Training hparams: same as rearrangement_fes_mle_300K.slurm
GRID_N=${GRID_N:-200}
BOUNDS=${BOUNDS:-1.0,6.5,1.0,6.5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
SMOOTH_WEIGHT=${SMOOTH_WEIGHT:-0.1}
SMOOTH_GRAD_WEIGHT=${SMOOTH_GRAD_WEIGHT:-0.001}
SMOOTH_LAP_WEIGHT=${SMOOTH_LAP_WEIGHT:-0.1}
SMOOTH_GRID_N=${SMOOTH_GRID_N:-128}

if [[ ! -d "$DATA" ]]; then
  echo "ERROR: DATA dir missing: $DATA" >&2
  exit 1
fi
if [[ ! -f "$METAD_FES" ]]; then
  echo "ERROR: METAD_FES missing: $METAD_FES" >&2
  exit 1
fi

mapfile -t ALL_CV < <(python - "$DATA" "$FILE_GLOB" <<'PY'
import glob, re, sys
from pathlib import Path
data, pat = Path(sys.argv[1]), sys.argv[2]
paths = glob.glob(str(data / pat))
stage_re = re.compile(r"stage(\d+)", re.I)

def key(p):
    m = stage_re.search(Path(p).name)
    return (int(m.group(1)) if m else 10**9, Path(p).name)

for p in sorted(paths, key=key):
    print(p)
PY
)

if [[ ${#ALL_CV[@]} -eq 0 ]]; then
  echo "ERROR: no files matching FILE_GLOB='$FILE_GLOB' under $DATA" >&2
  exit 1
fi

if [[ "${LAST_STAGES}" -gt 0 ]]; then
  n=${LAST_STAGES}
  if [[ $n -gt ${#ALL_CV[@]} ]]; then
    n=${#ALL_CV[@]}
  fi
  TAG="last${n}"
  SAVE=${SAVE:-$DATA/fes_mle_${TAG}}
  TRAIN_DATA="$SAVE/mle_inputs"
  rm -rf "$TRAIN_DATA"
  mkdir -p "$TRAIN_DATA"
  start=$(( ${#ALL_CV[@]} - n ))
  for ((i=start; i<${#ALL_CV[@]}; i++)); do
    src="${ALL_CV[$i]}"
    ln -sfn "$src" "$TRAIN_DATA/$(basename "$src")"
  done
  TRAIN_GLOB='stage[0-9]*_cv.npy'
  TRAIN_MAX="$n"
else
  TAG="first${MAX_STAGES}"
  SAVE=${SAVE:-$DATA/fes_mle_${TAG}}
  TRAIN_DATA="$DATA"
  TRAIN_GLOB="$FILE_GLOB"
  TRAIN_MAX="$MAX_STAGES"
fi

mkdir -p "$SAVE"

echo "host=$(hostname) date=$(date)"
echo "DATA=$DATA  TRAIN_DATA=$TRAIN_DATA  SAVE=$SAVE  TAG=$TAG"
echo "METAD_FES=$METAD_FES"
echo "STEPS=$STEPS  TRAIN_MAX=$TRAIN_MAX  GRID_N=$GRID_N  WD=$WEIGHT_DECAY  SMOOTH=$SMOOTH_WEIGHT GRAD=$SMOOTH_GRAD_WEIGHT LAP=$SMOOTH_LAP_WEIGHT"
echo "selected cv files:"
ls -1 "$TRAIN_DATA"/stage*_cv.npy
nvidia-smi || true

python -u "$ROOT/src/fp_transformer_torch/reconstruct_FES/train_max_likelihood_torch.py" \
  --data_dir "$TRAIN_DATA" \
  --file-glob "${TRAIN_GLOB}" \
  --max-stages "${TRAIN_MAX}" \
  --embedding identity \
  --bounds "${BOUNDS}" \
  --steps "${STEPS}" \
  --lr 1e-3 \
  --z-method grid \
  --grid-n "${GRID_N}" \
  --hidden_sizes 64 64 64 \
  --optimizer adamw \
  --weight_decay "${WEIGHT_DECAY}" \
  --smooth-weight "${SMOOTH_WEIGHT}" \
  --smooth-grad-weight "${SMOOTH_GRAD_WEIGHT}" \
  --smooth-lap-weight "${SMOOTH_LAP_WEIGHT}" \
  --smooth-grid-n "${SMOOTH_GRID_N}" \
  --seed 10 \
  --device cuda \
  --temperature 300 \
  --save_dir "$SAVE" \
  --ckpt_name fes_mle.pt \
  --log_every 500

python -u "$ROOT/experiments/rearrangement_mace/plot_fes_compare.py" \
  --ckpt "$SAVE/fes_mle.pt" \
  --metad-fes "$METAD_FES" \
  --output "$SAVE/fes_compare_mle_vs_metad.png" \
  --vmax-kcal 60 \
  --device cuda

echo "done $(date)  ckpt=$SAVE/fes_mle.pt  plot=$SAVE/fes_compare_mle_vs_metad.png"
