#!/bin/bash
#SBATCH -J torchpme_mm
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-01:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# torchpme vs nvalchemiops PME: pytest + correctness + R=512 profile + short sFPT smoke.
#
#   sbatch scripts/bench_torchpme_multigpu.slurm
#   REPLICAS=256 STEPS=0 sbatch scripts/bench_torchpme_multigpu.slurm

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

REPLICAS=${REPLICAS:-512}
CHECK_R=${CHECK_R:-8}
NGPUS=${NGPUS:-4}
STEPS=${STEPS:-2}
WARMUP=${WARMUP:-2}
REPEATS=${REPEATS:-4}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}
EXP="$ROOT/experiments/dialanine_water/sFPT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "REPLICAS=$REPLICAS CHECK_R=$CHECK_R NGPUS=$NGPUS STEPS=$STEPS"
nvidia-smi -L || true
python -c "import torch, torchpme, vesin; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count()); print('torchpme', torchpme.__version__)"

echo
echo "==== bench: pytest + correctness + profile ===="
python -u "$ROOT/scripts/bench_torchpme_multigpu.py" \
  --box-dir "${BOX_DIR}" \
  --replicas "${REPLICAS}" \
  --check-replicas "${CHECK_R}" \
  --n-gpus "${NGPUS}" \
  --pme-cutoff 8.0 \
  --pme-mesh-spacing 1.5 \
  --torchpme-method p3m \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}"

if [[ "${STEPS}" -gt 0 ]]; then
  OUT_DIR="$EXP/results/torchpme_smoke_r${REPLICAS}_g${NGPUS}_${SLURM_JOB_ID}"
  mkdir -p "$OUT_DIR"
  echo
  echo "==== sFPT smoke STEPS=${STEPS} R=${REPLICAS} NGPUS=${NGPUS} torchpme ===="
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
    --electrostatics torchpme \
    --torchpme-method p3m \
    --n-gpus "${NGPUS}" \
    --dtype float32 \
    --verbose \
    --verbose-pace 1 \
    --no-progress \
    --device cuda \
    --output "${OUT_DIR}"
  echo "smoke out: ${OUT_DIR}"
  ls -lt "${OUT_DIR}" | head -20
fi

echo "done $(date)"
