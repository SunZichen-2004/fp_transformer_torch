#!/bin/bash
#SBATCH -J sn2_mace
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Vacuum SN2 Cl- + CH3Cl + MACE-MH-0 + 2D distance CV + FPT (8-GPU MACE shard)
#   sbatch scripts/sn2_mace.slurm
#   sbatch --export=ALL,REPLICAS=1024,STEPS=500 scripts/sn2_mace.slurm

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
STEPS=${STEPS:-2000}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-8}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname)  date=$(date)  MACE_MODEL=${MACE_MODEL}  REPLICAS=${REPLICAS}  STEPS=${STEPS}  NGPUS=${NGPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -c "import torch; import torch_sim as ts; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch_sim', ts.__version__)"

EXP="$ROOT/experiments/SN2_mace"
OUT_DIR="$EXP/results/sn2_mace_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"
INIT_NPY=${INIT_NPY:-$EXP/init_coords_md_1024.npy}
if [[ ! -f "${INIT_NPY}" ]]; then
  echo "ERROR: missing relaxed init ${INIT_NPY}"
  echo "Run scripts/sn2_prepare.slurm first (FIRE min + MD)."
  exit 1
fi

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --pdb "$EXP/sn2.pdb" \
  --cv-file "$EXP/cv_pairs.txt" \
  --mass-file "$EXP/sn2_mass.txt" \
  --init-npy "${INIT_NPY}" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --n-gpus "${NGPUS}" \
  --charge -1.0 \
  --spin 1.0 \
  --wall-d-max 3.5 \
  --wall-k 20.0 \
  --cell-length 30.0 \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.5 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init 0.1 \
  --rew-lambda-max 0.5 \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.02 \
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
  --pre-eq-steps ${PRE_EQ_STEPS:-200} \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "plots: ${OUT_DIR}/d1_d2/"
