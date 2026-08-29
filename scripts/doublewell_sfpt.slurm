#!/bin/bash
#SBATCH -J dw_sfpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/doublewell/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/doublewell/job.%j.err

# Classic 2D double-well sFPT (CV=x1), reduced units: kB=T=m=1, barrier h=5
#
#   REPLICAS=512 SEED=0 sbatch scripts/doublewell_sfpt.slurm
# Smoke:
#   STEPS=20 REPLICAS=64 sbatch scripts/doublewell_sfpt.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

SEED=${SEED:-0}
REPLICAS=${REPLICAS:-512}
STEPS=${STEPS:-1000}
PRE_MD_STEPS=${PRE_MD_STEPS:-1000}
LAMBDA_INIT=${LAMBDA_INIT:-0.4}
LAMBDA_MAX=${LAMBDA_MAX:-1.0}
BARRIER_H=${BARRIER_H:-10.0}
FAST_K=${FAST_K:-10.0}
TEMP=${TEMP:-2}
DT=${DT:-1e-3}
GAMMA=${GAMMA:-10.0}
CV_GRAD_CLIP=${CV_GRAD_CLIP:-10}

EXP="$ROOT/experiments/doublewell"
OUT_DIR="$EXP/results/doublewell_sfpt_r${REPLICAS}_lam${LAMBDA_INIT}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=sFPT seed=$SEED R=$REPLICAS pre_md=$PRE_MD_STEPS steps=$STEPS λ0=$LAMBDA_INIT→$LAMBDA_MAX h=$BARRIER_H k=$FAST_K T=$TEMP cv_clip=$CV_GRAD_CLIP"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python -u "$EXP/sample.py" \
  --replicas "${REPLICAS}" \
  --stage-steps "${STEPS}" \
  --pre-md-steps "${PRE_MD_STEPS}" \
  --seed "${SEED}" \
  --baoab-dt "${DT}" \
  --baoab-langevin-gamma "${GAMMA}" \
  --temperature "${TEMP}" \
  --kB 1.0 \
  --mass 1.0 \
  --barrier-h "${BARRIER_H}" \
  --fast-k "${FAST_K}" \
  --init-well left \
  --noise-scale 0.05 \
  --rew-lambda-init "${LAMBDA_INIT}" \
  --rew-lambda-max "${LAMBDA_MAX}" \
  --rew-ess-threshold 0.95 \
  --rew-num-lambda-trials 20 \
  --rew-max-lambda-step 0.0000001 \
  --rew-min-lambda-step 0.0000001 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 4.0 \
  --cv-grad-clip-norm "${CV_GRAD_CLIP}" \
  --verbose \
  --verbose-pace 50 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
ls -lt "$OUT_DIR" | head -30
