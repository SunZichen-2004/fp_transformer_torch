#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu1
#SBATCH --qos=temp
#SBATCH -J dialanine_torch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=300

# Dialanine GAFF/SPONGE + BAOAB-FP (torch)
# Bandwidth rescale: unnormalized K, metric=row_sum (no /B),
# adaptive band = median×[0.8,1.2] clipped to fixed_row_sum [2,4]
# Each λ-stage end: coords + phi-psi under $OUT_DIR/phi_psi/

set -eo pipefail
ROOT=/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch
cd "$ROOT" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate MACE
set +u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MPLBACKEND=Agg

echo "host=$(hostname)  date=$(date)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

EXP="$ROOT/experiments/dialanine"
OUT_DIR="$EXP/results/dialanine_gaff_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --sponge-prefix "$EXP/dialanine" \
  --cv-file "$EXP/backbone_cv.txt" \
  --init-npy "$EXP/init_coords_2048.npy" \
  --replicas 2048 \
  --steps 1000 \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.0 \
  --compile \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init 0.1 \
  --rew-lambda-max 0.2 \
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
  --rescale-bw-max 8.0 \
  --rescale-fixed-row-sum-min 2.0 \
  --rescale-fixed-row-sum-max 4.0 \
  --verbose \
  --verbose-pace 20 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "plots: ${OUT_DIR}/phi_psi/"
