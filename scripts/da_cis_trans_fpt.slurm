#!/bin/bash
#SBATCH -J da_ct_fpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# IMDA cis/trans + MACE-MH-0 + paper 2D CV (HLDA, c8*c17) FPT
# Pre-MD (no noise) → FPT; λ starts at 0.1; R=1024

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
STEPS=${STEPS:-1000}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5}
HEAL_STEPS=${HEAL_STEPS:-100}
PRE_MD_STEPS=${PRE_MD_STEPS:-200}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-8}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname)  date=$(date)  MACE=${MACE_MODEL}  R=${REPLICAS}  STEPS=${STEPS}  NGPUS=${NGPUS}"
echo "PRE_MD_STEPS=${PRE_MD_STEPS}  HEAL_LAMBDAS=${HEAL_LAMBDAS}"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'n_gpu', torch.cuda.device_count())"

EXP="$ROOT/experiments/DA_cis_trans"
OUT_DIR="$EXP/results/da_cis_trans_fpt_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --xyz "$EXP/reactant.xyz" \
  --mass-file "$EXP/mass.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --n-gpus "${NGPUS}" \
  --pre-md-steps "${PRE_MD_STEPS}" \
  --pre-md-dump-every 50 \
  --charge 0.0 \
  --spin 1.0 \
  --wall-d-max 10.0 \
  --wall-k 10.0 \
  --cell-length 40.0 \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --cv-grad-clip-norm 1 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --heal-lambdas "${HEAL_LAMBDAS}" \
  --heal-steps "${HEAL_STEPS}" \
  --rew-lambda-init 0.2 \
  --rew-lambda-max 0.3 \
  --rew-ess-threshold 0.97 \
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
  --rescale-fixed-row-sum-max 4.0 \
  --dtype float32 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
