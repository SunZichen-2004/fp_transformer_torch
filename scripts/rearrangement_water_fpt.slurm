#!/bin/bash
#SBATCH -J claisen_wat_fpt
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 0-06:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Solvated Claisen FPT (ML/MM), same box as water MetaD; λ_init=0.1

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

SEED=${SEED:-100}
MACE_MODEL=${MACE_MODEL:-mh-0}
REPLICAS=${REPLICAS:-512}
STEPS=${STEPS:-100}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.1,0.05}
HEAL_STEPS=${HEAL_STEPS:-500}
PRE_MD_STEPS=${PRE_MD_STEPS:-100}
LAMBDA_INIT=${LAMBDA_INIT:-0.025}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
DSF_ALPHA=${DSF_ALPHA:-0.2}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/mlmm_box}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-8}}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname) date=$(date) Claisen-water-FPT MACE=$MACE_MODEL R=$REPLICAS λ0=$LAMBDA_INIT seed=$SEED NGPUS=$NGPUS elec=$ELECTROSTATICS"
nvidia-smi || true
head -2 "$BOX_DIR/box.xyz" || true

EXP="$ROOT/experiments/rearrangement_water_fpt"
OUT_DIR="$EXP/results/claisen_water_fpt_${MACE_MODEL}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --box-dir "${BOX_DIR}" \
  --cv-file "$EXP/cv_pairs.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --n-gpus "${NGPUS}" \
  --pre-md-steps "${PRE_MD_STEPS}" \
  --charge 0.0 \
  --spin 1.0 \
  --pme-cutoff 9.0 \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --dsf-alpha "${DSF_ALPHA}" \
  --wall-d-max 8.0 \
  --wall-k 10.0 \
  --softmin-a 10.0 \
  --softmin-d-wall 2.4 \
  --softmin-k 10.0 \
  --update-pace 20 \
  --timestep 0.001 \
  --langevin-friction 10.0 \
  --cv-grad-clip-norm 2 \
  --temperature 300 \
  --heal-lambdas "${HEAL_LAMBDAS}" \
  --heal-steps "${HEAL_STEPS}" \
  --lambda-init "${LAMBDA_INIT}" \
  --lambda-max 0.3 \
  --ess-threshold 0.97 \
  --num-lambda-trials 10 \
  --max-lambda-step 0.00000000002 \
  --min-lambda-step 0.00000000001 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 4.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 5.0 \
  --traj-full \
  --max-xyz-frames 32 \
  --dtype float32 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)  out=${OUT_DIR}"
