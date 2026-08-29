#!/bin/bash
#SBATCH -J sn2_prep
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# SN2: FIRE min 20 → NVT MD → init_coords_md_1024.npy (+ xyz / d1-d2)

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
MIN_STEPS=${MIN_STEPS:-20}
MD_STEPS=${MD_STEPS:-2000}
MD_DUMP_EVERY=${MD_DUMP_EVERY:-200}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-8}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname)  date=$(date)  prepare SN2  NGPUS=$NGPUS REPLICAS=$REPLICAS"
nvidia-smi || true

EXP="$ROOT/experiments/SN2_mace"
OUT_DIR="$EXP/results/sn2_prepare_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/prepare_init.py" \
  --mace-model "${MACE_MODEL}" \
  --pdb "$EXP/sn2.pdb" \
  --cv-file "$EXP/cv_pairs.txt" \
  --mass-file "$EXP/sn2_mass.txt" \
  --init-npy "$EXP/init_coords_1024.npy" \
  --replicas "${REPLICAS}" \
  --min-steps "${MIN_STEPS}" \
  --md-steps "${MD_STEPS}" \
  --md-dump-every "${MD_DUMP_EVERY}" \
  --fpt-init-from-last-dumps 3 \
  --fpt-init-out "$EXP/init_coords_md_1024.npy" \
  --n-gpus "${NGPUS}" \
  --charge -1.0 \
  --spin 1.0 \
  --wall-d-max 3.5 \
  --wall-k 20.0 \
  --cell-length 30.0 \
  --timestep 0.001 \
  --langevin-friction 10.0 \
  --temperature 300 \
  --dtype float32 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "FPT init: $EXP/init_coords_md_1024.npy"
echo "artifacts: ${OUT_DIR}"
ls -lah "$OUT_DIR"/*.xyz "$EXP/init_coords_md_1024.npy" 2>/dev/null || true
