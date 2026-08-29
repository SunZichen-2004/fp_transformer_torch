#!/bin/bash
#SBATCH -J pd_sfpt_cpu
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/particle_dimer/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/particle_dimer/job.%j.err

# CPU-only particle-dimer sFPT. MD up to ~1h55m; leftover walltime for
# convergence analysis. λ stays ≈0.1 (tiny Δλ). Kernel force F(1+λ)/λ.
#
#   TEMP=1 INIT_COORDS=/path/init_coords_from_pre_md.npy PRE_MD_STEPS=0 \
#     sbatch scripts/particle_dimer_sfpt_cpu.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=""

SEED=${SEED:-0}
REPLICAS=${REPLICAS:-128}
LAMBDA_INIT=${LAMBDA_INIT:-0.1}
LAMBDA_MAX=${LAMBDA_MAX:-1.0}
DT=${DT:-1e-4}
GAMMA=${GAMMA:-10.0}
TEMP=${TEMP:-1}
INIT_COORDS=${INIT_COORDS:-}
PRE_MD_STEPS=${PRE_MD_STEPS:-10000}
PRE_MD_CHUNK=${PRE_MD_CHUNK:-1000}
PRE_MD_TOL=${PRE_MD_TOL:-0.15}
HEAL_LAMBDAS=${HEAL_LAMBDAS:-0.5,0.1}
HEAL_STEPS=${HEAL_STEPS:-200}
STAGE_STEPS=${STAGE_STEPS:-100}
MAX_RUNTIME_SEC=${MAX_RUNTIME_SEC:-6900}
ROW_SUM_MIN=${ROW_SUM_MIN:-15.0}
ROW_SUM_MAX=${ROW_SUM_MAX:-25.0}

EXP="$ROOT/experiments/particle_dimer"
JOBTAG="${SLURM_JOB_ID:-cpu}"
OUT_DIR="$EXP/results/particle_dimer_sfpt_r${REPLICAS}_lam${LAMBDA_INIT}_T${TEMP}_s${SEED}_${JOBTAG}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=sFPT-CPU seed=$SEED R=$REPLICAS pre_md=$PRE_MD_STEPS chunk=$PRE_MD_CHUNK tol=$PRE_MD_TOL steps/stage=$STAGE_STEPS λ0=$LAMBDA_INIT heal=$HEAL_LAMBDAS T=$TEMP dt=$DT γ=$GAMMA row_sum=[$ROW_SUM_MIN,$ROW_SUM_MAX] max_runtime=${MAX_RUNTIME_SEC}s init=${INIT_COORDS:-none}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"

INIT_ARGS=()
if [[ -n "${INIT_COORDS}" ]]; then
  INIT_ARGS+=(--init-coords "${INIT_COORDS}")
fi

python -u "$EXP/sample.py" \
  --replicas "${REPLICAS}" \
  --stage-steps "${STAGE_STEPS}" \
  --seed "${SEED}" \
  --baoab-dt "${DT}" \
  --baoab-langevin-gamma "${GAMMA}" \
  --temperature "${TEMP}" \
  --kB 1.0 \
  --dimer-init extended \
  --noise-scale 0.05 \
  "${INIT_ARGS[@]}" \
  --rew-lambda-init "${LAMBDA_INIT}" \
  --rew-lambda-max "${LAMBDA_MAX}" \
  --rew-ess-threshold 0.96 \
  --rew-num-lambda-trials 20 \
  --rew-max-lambda-step 1e-12 \
  --rew-min-lambda-step 1e-12 \
  --heal-lambdas "${HEAL_LAMBDAS}" \
  --heal-steps "${HEAL_STEPS}" \
  --pre-md-steps "${PRE_MD_STEPS}" \
  --pre-md-chunk "${PRE_MD_CHUNK}" \
  --pre-md-tol "${PRE_MD_TOL}" \
  --max-runtime-sec "${MAX_RUNTIME_SEC}" \
  --kernel-grad-beta-mode lambda_over_one_plus_lambda \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min "${ROW_SUM_MIN}" \
  --rescale-fixed-row-sum-max "${ROW_SUM_MAX}" \
  --cv-grad-clip-norm 2.0 \
  --verbose \
  --verbose-pace 50 \
  --device cpu \
  --output "${OUT_DIR}"

echo "MD done $(date)  out=${OUT_DIR}"
ls -lt "$OUT_DIR" | head -40 || true

CONV="$OUT_DIR/convergence_analysis"
mkdir -p "$CONV"
python -u "$EXP/plot_convergence_analysis.py" \
  --fpt-dir "${OUT_DIR}" \
  --out-dir "${CONV}" \
  --group-size 10 \
  --stage-steps "${STAGE_STEPS}" \
  --dt "${DT}" \
  --temperature "${TEMP}" \
  --lambda-target "${LAMBDA_INIT}" \
  --device cpu

echo "analysis done $(date)"
ls -lt "$CONV" | head -30 || true
