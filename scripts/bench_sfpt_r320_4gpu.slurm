#!/bin/bash
#SBATCH -J sfpt_r320_4g
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-00:40:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# R=320 sFPT throughput: original 1-GPU nvalchemiops PME vs 4-GPU batched torchpme.
#
#   sbatch scripts/bench_sfpt_r320_4gpu.slurm

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

REPLICAS=${REPLICAS:-320}
STEPS=${STEPS:-40}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}
EXP="$ROOT/experiments/dialanine_water/sFPT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "REPLICAS=$REPLICAS STEPS=$STEPS"
nvidia-smi -L || true

echo
echo "==== potential forward R=${REPLICAS} (pytest skipped) ===="
python -u "$ROOT/scripts/bench_torchpme_multigpu.py" \
  --box-dir "${BOX_DIR}" \
  --replicas "${REPLICAS}" \
  --check-replicas 4 \
  --n-gpus 4 \
  --pme-cutoff 8.0 \
  --pme-mesh-spacing 1.5 \
  --torchpme-method p3m \
  --warmup 2 \
  --repeats 4 \
  --skip-pytest

run_sfpt () {
  local label="$1"
  local elec="$2"
  local ngpus="$3"
  local out="$EXP/results/bench_r${REPLICAS}_${label}_${SLURM_JOB_ID}"
  mkdir -p "$out"
  echo
  echo "==== sFPT ${label}  elec=${elec}  n-gpus=${ngpus}  R=${REPLICAS} STEPS=${STEPS} ===="
  python -u "$EXP/sample.py" \
    --box-dir "${BOX_DIR}" \
    --cv-file "${BOX_DIR}/backbone_cv.txt" \
    --replicas "${REPLICAS}" \
    --steps "${STEPS}" \
    --seed 0 \
    --pre-md-steps 0 \
    --heal-lambdas "" \
    --heal-steps 0 \
    --update-pace 20 \
    --timestep 0.001 \
    --langevin-friction 10.0 \
    --temperature 300 \
    --lambda-init 0.08 \
    --lambda-max 0.2 \
    --pme-cutoff 8.0 \
    --pme-mesh-spacing 1.5 \
    --electrostatics "${elec}" \
    --torchpme-method p3m \
    --n-gpus "${ngpus}" \
    --dtype float32 \
    --verbose \
    --verbose-pace 20 \
    --no-progress \
    --device cuda \
    --output "${out}"
  echo "out: ${out}"
}

run_sfpt "pme_1gpu" pme 1
run_sfpt "torchpme_4gpu" torchpme 4

echo "done $(date)"
