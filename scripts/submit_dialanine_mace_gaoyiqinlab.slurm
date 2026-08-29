#!/bin/bash
#SBATCH -J dialanine_mace
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Smoke / flexible submit. Override GPUs at submit time:
#   sbatch --gpus=8 --export=ALL,NGPUS=8,REPLICAS=1024,STEPS=40 scripts/submit_dialanine_mace_gaoyiqinlab.slurm

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

MACE_MODEL=${MACE_MODEL:-off-small}
REPLICAS=${REPLICAS:-64}
STEPS=${STEPS:-40}
NGPUS=${NGPUS:-1}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi
INIT_NPY=${INIT_NPY:-$ROOT/experiments/dialanine_mace/init_coords_1024.npy}

echo "host=$(hostname)  date=$(date)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MACE_MODEL=${MACE_MODEL}  REPLICAS=${REPLICAS}  STEPS=${STEPS}  NGPUS=${NGPUS}"
nvidia-smi || true
python -c "import torch, torch_sim as ts; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch_sim', ts.__version__)"

EXP="$ROOT/experiments/dialanine_mace"
OUT_DIR="$EXP/results/dialanine_mace_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --pdb "$EXP/dialanine.pdb" \
  --cv-file "$EXP/backbone_cv.txt" \
  --mass-file "$EXP/dialanine_mass.txt" \
  --init-npy "${INIT_NPY}" \
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
echo "plots: ${OUT_DIR}/phi_psi/"
ls -la "${OUT_DIR}/phi_psi/" || true
