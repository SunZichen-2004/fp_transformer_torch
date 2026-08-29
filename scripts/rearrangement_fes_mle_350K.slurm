#!/bin/bash
#SBATCH -J claisen_fes_mle
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Train Torch MLE FES from Claisen FPT CV npy, then plot vs MetaD FES.

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

DATA=${DATA:-$ROOT/experiments/rearrangement_mace/results/claisen_fpt_mh-0_g4_85914}
SAVE=${SAVE:-$DATA/fes_mle}
METAD_FES=${METAD_FES:-$ROOT/experiments/rearrangement_metadynamic/results/claisen_metad_mh-0_r32_85915/live/fes.npz}
STEPS=${STEPS:-30000}
MAX_STAGES=${MAX_STAGES:-16}
# 2D Z: dense midpoint grid (no MC). n_points = GRID_N^2
GRID_N=${GRID_N:-200}
BOUNDS=${BOUNDS:-1.0,6.5,1.0,6.5}

mkdir -p "$SAVE"

echo "host=$(hostname) date=$(date)  DATA=$DATA  SAVE=$SAVE  STEPS=$STEPS  MAX_STAGES=$MAX_STAGES  GRID_N=$GRID_N"
nvidia-smi || true

python -u "$ROOT/src/fp_transformer_torch/reconstruct_FES/train_max_likelihood_torch.py" \
  --data_dir "$DATA" \
  --file-glob 'stage[0-9]*_cv.npy' \
  --max-stages "${MAX_STAGES}" \
  --embedding identity \
  --bounds "${BOUNDS}" \
  --steps "${STEPS}" \
  --lr 1e-3 \
  --z-method grid \
  --grid-n "${GRID_N}" \
  --hidden_sizes 64 64 64 \
  --optimizer adamw \
  --weight_decay 1e-4 \
  --seed 0 \
  --device cuda \
  --temperature 350 \
  --save_dir "$SAVE" \
  --ckpt_name fes_mle.pt \
  --log_every 500

python -u "$ROOT/experiments/rearrangement_mace/plot_fes_compare.py" \
  --ckpt "$SAVE/fes_mle.pt" \
  --metad-fes "$METAD_FES" \
  --output "$SAVE/fes_compare_mle_vs_metad.png" \
  --device cuda

echo "done $(date)  ckpt=$SAVE/fes_mle.pt  plot=$SAVE/fes_compare_mle_vs_metad.png"
