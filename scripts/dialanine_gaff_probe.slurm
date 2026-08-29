#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu1
#SBATCH --qos=temp
#SBATCH -J dia_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=30

set -eo pipefail
ROOT=/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch
cd "$ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate MACE; set +u
export PYTHONUNBUFFERED=1
EXP="$ROOT/experiments/dialanine"
python -u "$EXP/sample.py" \
  --sponge-prefix "$EXP/dialanine" \
  --cv-file "$EXP/backbone_cv.txt" \
  --replicas 2048 \
  --steps 500 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init 0.05 \
  --rew-lambda-max 0.5 \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.02 \
  --rew-min-lambda-step 0.001 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --verbose --verbose-pace 50 \
  --device cuda \
  --output "$EXP/results/dialanine_probe_${SLURM_JOB_ID}"
