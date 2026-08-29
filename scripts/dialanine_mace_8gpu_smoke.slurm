#!/bin/bash
#SBATCH -J dial_mace8
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# 8-GPU smoke: 1024 replicas × 40 steps
set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT"
source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u; conda activate MACE_torchsim; set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR=/storage/gaoyiqinLab/sunzicheng/tmp
mkdir -p "$TMPDIR"

NGPUS=8
REPLICAS=${REPLICAS:-1024}
STEPS=${STEPS:-40}
MACE_MODEL=${MACE_MODEL:-off-small}

echo "host=$(hostname) date=$(date) NGPUS=$NGPUS REPLICAS=$REPLICAS STEPS=$STEPS"
nvidia-smi
python -c "import torch; print('n_gpu', torch.cuda.device_count())"

EXP="$ROOT/experiments/dialanine_mace"
OUT_DIR="$EXP/results/dialanine_mace_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --pdb "$EXP/dialanine.pdb" \
  --cv-file "$EXP/backbone_cv.txt" \
  --mass-file "$EXP/dialanine_mass.txt" \
  --init-npy "$EXP/init_coords_1024.npy" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --n-gpus "${NGPUS}" \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.0 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init 0.2 \
  --rew-lambda-max 0.4 \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.02 \
  --rew-min-lambda-step 0.001 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 4.0 \
  --dtype float32 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
