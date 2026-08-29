#!/bin/bash
#SBATCH -J da_mace
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# DA tropone+dimethylfulvene + MACE-MH-0 + soft-distance (s1,s2) FPT
# Heal λ schedule (hard-set, no ESS) → ESS anneal from λ_init to λ_max

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
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.1,0.05,0.02,}
HEAL_STEPS=${HEAL_STEPS:-100}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-8}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname)  date=$(date)  MACE_MODEL=${MACE_MODEL}  REPLICAS=${REPLICAS}  STEPS=${STEPS}  NGPUS=${NGPUS}"
echo "HEAL_LAMBDAS=${HEAL_LAMBDAS}  HEAL_STEPS=${HEAL_STEPS}"
nvidia-smi || true
python -c "import torch; import torch_sim as ts; print('torch', torch.__version__, 'n_gpu', torch.cuda.device_count(), 'torch_sim', ts.__version__)"

EXP="$ROOT/experiments/DA_mace"
OUT_DIR="$EXP/results/da_mace_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"
INIT_NPY=${INIT_NPY:-$EXP/results/da_mace_mh-0_g8_85632/stage0000_lambda0p050000_coords.npy}
if [[ ! -f "${INIT_NPY}" ]]; then
  echo "ERROR: missing init ${INIT_NPY}"
  exit 1
fi

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --xyz "$EXP/DA_tropone_dimethylfulvene.xyz" \
  --cv-file "$EXP/cv_pairs.txt" \
  --mass-file "$EXP/da_mass.txt" \
  --init-npy "${INIT_NPY}" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --n-gpus "${NGPUS}" \
  --charge 0.0 \
  --spin 1.0 \
  --d-ref 1.7 \
  --soft-n 6 \
  --soft-m 8 \
  --wall-d-max 8 \
  --wall-k 10.0 \
  --cell-length 40.0 \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 1 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --heal-lambdas "${HEAL_LAMBDAS}" \
  --heal-steps "${HEAL_STEPS}" \
  --rew-lambda-init 0.015 \
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
  --rescale-fixed-row-sum-max 5.0 \
  --dtype float32 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "plots: ${OUT_DIR}/s1_s2/"
