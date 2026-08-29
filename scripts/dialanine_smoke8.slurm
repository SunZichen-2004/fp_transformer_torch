#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu1
#SBATCH --qos=temp
#SBATCH -J dial_smoke8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=60

# Smoke: dialanine GAFF BAOAB-FP, replicas=8, 200 steps/stage
# Check T_kin, bandwidth [bw_min,bw_max], phi-psi after each stage

set -eo pipefail
ROOT=/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch
cd "$ROOT" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate MACE
set +u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg

echo "host=$(hostname)  date=$(date)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

EXP="$ROOT/experiments/dialanine"
OUT_DIR="$EXP/results/dialanine_smoke8_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --sponge-prefix "$EXP/dialanine" \
  --cv-file "$EXP/backbone_cv.txt" \
  --init-npy "$EXP/init_coords_2048.npy" \
  --replicas 8 \
  --steps 200 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.05 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init 0.05 \
  --rew-lambda-max 0.5 \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.02 \
  --rew-min-lambda-step 0.001 \
  --rew-debug-reweight \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --rescale-bw-min 0.001 \
  --rescale-bw-max 1.0 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "plots: ${OUT_DIR}/phi_psi/"
ls -la "${OUT_DIR}/phi_psi/" || true
