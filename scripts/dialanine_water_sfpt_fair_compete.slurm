#!/bin/bash
#SBATCH -J dial_sfpt_fair
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-06:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Fair-compete sFPT vs MetaD 89603 init (full box, including water).
# Same protocol as 87372 except:
#   INIT = 89603 init_coords_from_pre_md.npy
#   heal λ = 0.5, 0.1, 0.04
#   anneal λ = 0.0625
#   wall-clock 6 h
#
#   REPLICAS=320 sbatch scripts/dialanine_water_sfpt_fair_compete.slurm
#   REPLICAS=256 sbatch scripts/dialanine_water_sfpt_fair_compete.slurm

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

SEED=${SEED:-1}
REPLICAS=${REPLICAS:-64}
STEPS=${STEPS:-100}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5,0.1,0.04}
HEAL_STEPS=${HEAL_STEPS:-100}
LAMBDA_INIT=${LAMBDA_INIT:-0.0625}
LAMBDA_MAX=${LAMBDA_MAX:-0.2}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_CUTOFF=${PME_CUTOFF:-8.0}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}
NGPUS=${NGPUS:-1}
PRE_MD_STEPS=${PRE_MD_STEPS:-0}
INIT_NPY=${INIT_NPY:-$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r256_s2_89603/init_coords_from_pre_md.npy}

EXP="$ROOT/experiments/dialanine_water/sFPT"
OUT_DIR="$EXP/fair_compete_result/dialanine_water_sfpt_r${REPLICAS}_g${NGPUS}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=sFPT-fair seed=$SEED R=$REPLICAS steps=$STEPS λ0=$LAMBDA_INIT heal=$HEAL_LAMBDAS NGPUS=$NGPUS elec=$ELECTROSTATICS"
echo "INIT_NPY=$INIT_NPY PRE_MD_STEPS=$PRE_MD_STEPS box=$BOX_DIR"
echo "OUT_DIR=$OUT_DIR"
nvidia-smi || true
head -2 "$BOX_DIR/box.xyz" || true
python -c "import numpy as np, torch; a=np.load('${INIT_NPY}'); print('init', a.shape, 'cuda', torch.cuda.is_available())"

python -u "$EXP/sample.py" \
  --box-dir "${BOX_DIR}" \
  --cv-file "${BOX_DIR}/backbone_cv.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --init-npy "${INIT_NPY}" \
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
ls -lt "${OUT_DIR}" | head -20
