#!/bin/bash
#SBATCH -J dw_l0.08_s0
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-08:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/lambda_scan/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/lambda_scan/job.%j.err

# λ-scan: LAMBDA_INIT=0.08  SEED=0
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

SEED=0
REPLICAS=${REPLICAS:-320}
STEPS=${STEPS:-100}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5,0.1}
HEAL_STEPS=${HEAL_STEPS:-100}
LAMBDA_INIT=0.08
LAMBDA_MAX=${LAMBDA_MAX:-0.2}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_CUTOFF=${PME_CUTOFF:-8.0}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}
WATER_EXP="$ROOT/experiments/dialanine_water"
NGPUS=${NGPUS:-1}
OUT_BASE=$ROOT/experiments/dialanine_water/lambda_scan

INIT_NPY=${INIT_NPY:-$WATER_EXP/init_coords_1024.npy}
if [[ -f "${INIT_NPY}" ]]; then
  PRE_MD_STEPS=${PRE_MD_STEPS:-0}
else
  echo "[warn] INIT_NPY missing: ${INIT_NPY} → fall back to box xyz + PRE_MD"
  PRE_MD_STEPS=${PRE_MD_STEPS:-100}
  INIT_NPY=""
fi

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=sFPT λ-scan seed=$SEED R=$REPLICAS steps=$STEPS λ0=$LAMBDA_INIT"
echo "INIT_NPY=${INIT_NPY:-<none>} PRE_MD_STEPS=$PRE_MD_STEPS"
nvidia-smi || true

EXP="$ROOT/experiments/dialanine_water/sFPT"
OUT_DIR="$OUT_BASE/dialanine_water_sfpt_lam${LAMBDA_INIT}_r${REPLICAS}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

INIT_ARGS=()
if [[ -n "${INIT_NPY}" && -f "${INIT_NPY}" ]]; then
  INIT_ARGS+=(--init-npy "${INIT_NPY}")
fi

python -u "$EXP/sample.py" \
  --box-dir "${BOX_DIR}" \
  --cv-file "${BOX_DIR}/backbone_cv.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  "${INIT_ARGS[@]}" \
  --pre-md-steps "${PRE_MD_STEPS}" \
  --heal-lambdas "${HEAL_LAMBDAS}" \
  --heal-steps "${HEAL_STEPS}" \
  --update-pace 20 \
  --timestep 0.001 \
  --langevin-friction 10.0 \
  --temperature 300 \
  --lambda-init "${LAMBDA_INIT}" \
  --lambda-max "${LAMBDA_MAX}" \
  --ess-threshold 0.97 \
  --num-lambda-trials 10 \
  --max-lambda-step 0.000000002 \
  --min-lambda-step 0.000000001 \
  --pme-cutoff "${PME_CUTOFF}" \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --n-gpus "${NGPUS}" \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
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
echo "out: ${OUT_DIR}"
ls -lt "${OUT_DIR}" | head -25
