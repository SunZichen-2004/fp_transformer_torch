#!/bin/bash
#SBATCH -J claisen_fpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Allyl phenyl ether Claisen FPT: (d1,d2)=d(O7-C8), d(C1-C10) in Å

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

MACE_MODEL=${MACE_MODEL:-mh-0}
REPLICAS=${REPLICAS:-1024}
STEPS=${STEPS:-2500}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5,0.1,0.08,0.05}
HEAL_STEPS=${HEAL_STEPS:-1000}
PRE_MD_STEPS=${PRE_MD_STEPS:-200}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-8}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname) date=$(date) Claisen-FPT MACE=$MACE_MODEL R=$REPLICAS NGPUS=$NGPUS"
nvidia-smi || true

EXP="$ROOT/experiments/rearrangement_mace"
OUT_DIR="$EXP/results/claisen_fpt_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --xyz "$EXP/allyl_phenyl_ether.xyz" \
  --cv-file "$EXP/cv_pairs.txt" \
  --mass-file "$EXP/mass.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --n-gpus "${NGPUS}" \
  --pre-md-steps "${PRE_MD_STEPS}" \
  --charge 0.0 \
  --spin 1.0 \
  --wall-d-max 6.0 \
  --wall-k 10.0 \
  --softmin-a 10.0 \
  --softmin-d-wall 2.7 \
  --softmin-k 10.0 \
  --cell-length 40.0 \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --cv-grad-clip-norm 2 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --heal-lambdas "${HEAL_LAMBDAS}" \
  --heal-steps "${HEAL_STEPS}" \
  --rew-lambda-init 0.025 \
  --rew-lambda-max 0.5 \
  --rew-ess-threshold 0.99 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.002 \
  --rew-min-lambda-step 0.001 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 4.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 5.0 \
  --dtype float32 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)  out=${OUT_DIR}"
