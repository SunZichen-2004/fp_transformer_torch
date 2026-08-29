#!/bin/bash
#SBATCH -J tetra_sfpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job.%j.err

# Vacuum tetra-alanine (ACE-(ALA)3-NME) GAFF/SPONGE + BAOAB-FP
# CV: 3 φ + 3 ψ (torus kernel)
# Flow: MD init (1024 frames) → λ-heal 0.5,0.1 → ESS anneal from λ0=0.08
#
#   SEED=0 REPLICAS=1024 STEPS=1000 sbatch scripts/tetra_alanine_sfpt.slurm

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
REPLICAS=${REPLICAS:-1024}
STEPS=${STEPS:-200}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5,0.1}
HEAL_STEPS=${HEAL_STEPS:-500}
PRE_MD_STEPS=${PRE_MD_STEPS:-0}
LAMBDA_INIT=${LAMBDA_INIT:-0.05}
LAMBDA_MAX=${LAMBDA_MAX:-0.2}

EXP="$ROOT/experiments/tetra_alanine_sFPT"
OUT_DIR="$EXP/results/tetra_alanine_sfpt_r${REPLICAS}_s${SEED}_${SLURM_JOB_ID}"
INIT_NPY="$EXP/init_coords_${REPLICAS}.npy"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "seed=$SEED R=$REPLICAS steps=$STEPS PRE_MD=$PRE_MD_STEPS HEAL_LAMBDAS=$HEAL_LAMBDAS HEAL_STEPS=$HEAL_STEPS λ0=$LAMBDA_INIT"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

if [[ ! -f "$INIT_NPY" ]]; then
  echo "[error] missing $INIT_NPY — run scripts/tetra_alanine_md_init.slurm first" >&2
  exit 1
fi
echo "[init] using $INIT_NPY"

mkdir -p "$OUT_DIR"

python -u "$EXP/sample.py" \
  --sponge-prefix "$EXP/tetra_alanine" \
  --cv-file "$EXP/backbone_cv.txt" \
  --init-npy "$INIT_NPY" \
  --replicas "$REPLICAS" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --pre-md-steps "$PRE_MD_STEPS" \
  --heal-lambdas "$HEAL_LAMBDAS" \
  --heal-steps "$HEAL_STEPS" \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.0 \
  --compile \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init "$LAMBDA_INIT" \
  --rew-lambda-max "$LAMBDA_MAX" \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.00000002 \
  --rew-min-lambda-step 0.00000001 \
  --rew-debug-reweight \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min 20 \
  --rescale-fixed-row-sum-max 40 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
echo "plots: ${OUT_DIR}/phi_psi/"
ls -lt "${OUT_DIR}" | head -25
