#!/bin/bash
#SBATCH -J pd_sfpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-08:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/particle_dimer/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/particle_dimer/job.%j.err

# Particle-dimer sFPT (CV = pair distance of dimer atoms 0–1).
#
# GPU queue:
#   sbatch scripts/particle_dimer_sfpt.slurm
#   STEPS=10000 REPLICAS=256 sbatch scripts/particle_dimer_sfpt.slurm
# Login node (CPU):
#   bash scripts/particle_dimer_sfpt.slurm

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
REPLICAS=${REPLICAS:-128}
LAMBDA_INIT=${LAMBDA_INIT:-0.05}
LAMBDA_MAX=${LAMBDA_MAX:-1.0}
DT=${DT:-1e-4}
GAMMA=${GAMMA:-10.0}
TEMP=${TEMP:-2}
NOISE=${NOISE:-0.05}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  DEVICE=${DEVICE:-cuda}
  STEPS=${STEPS:-10000}
  JOBTAG="${SLURM_JOB_ID}"
else
  DEVICE=${DEVICE:-cpu}
  STEPS=${STEPS:-200}
  JOBTAG="login"
fi

EXP="$ROOT/experiments/particle_dimer"
mkdir -p "$EXP"
OUT_DIR="$EXP/results/particle_dimer_sfpt_r${REPLICAS}_lam${LAMBDA_INIT}_s${SEED}_${JOBTAG}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-login}"
echo "METHOD=sFPT seed=$SEED R=$REPLICAS steps=$STEPS λ0=$LAMBDA_INIT→$LAMBDA_MAX T=$TEMP dt=$DT γ=$GAMMA device=$DEVICE"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python -u "$EXP/sample.py" \
  --replicas "${REPLICAS}" \
  --stage-steps "${STEPS}" \
  --seed "${SEED}" \
  --baoab-dt "${DT}" \
  --baoab-langevin-gamma "${GAMMA}" \
  --temperature "${TEMP}" \
  --kB 1.0 \
  --rew-anneal-ref-temperature "${TEMP}" \
  --dimer-init extended \
  --noise-scale "${NOISE}" \
  --rew-lambda-init "${LAMBDA_INIT}" \
  --rew-lambda-max "${LAMBDA_MAX}" \
  --rew-ess-threshold 0.96 \
  --rew-num-lambda-trials 20 \
  --rew-max-lambda-step 0.03 \
  --rew-min-lambda-step 0.01 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --verbose \
  --verbose-pace 50 \
  --device "${DEVICE}" \
  --output "${OUT_DIR}"

echo "done $(date)"
ls -lt "$OUT_DIR" | head -30
