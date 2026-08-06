#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu1
#SBATCH --qos=temp
#SBATCH -J pd_dist_torch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=60

# Torch port: same MD/anneal settings as scripts_gao/particle_dimer_distance.slurm
# (no systematic resampling)

set -eo pipefail
ROOT=/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch
cd "$ROOT" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate MACE
set +u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "host=$(hostname)  date=$(date)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

JAX_ROOT=/share/home/sunzc/work/Fokker_Plank/fp_0429_final/fp-transformer
OUT_DIR="$ROOT/experiments/particle_dimer/results/final_mbar_dist_torch_${SLURM_JOB_ID}"
INIT_COORDS="$JAX_ROOT/experiments/particle_dimer/results/control_onlymd_0710_82820/baoab_final_coords.npy"

python -u experiments/particle_dimer/sample.py \
  --init-coords "${INIT_COORDS}" \
  --replicas 512 \
  --steps 10000 \
  --verbose \
  --verbose-pace 50 \
  --dimer-init extended \
  --noise-scale 0.05 \
  --baoab-dt 1e-4 \
  --baoab-langevin-gamma 10.0 \
  --temperature 2 \
  --kB 1.0 \
  --rew-anneal-ref-temperature 2 \
  --rew-lambda-init 0.1 \
  --rew-lambda-max 1.0 \
  --rew-ess-threshold 0.96 \
  --rew-num-lambda-trials 20 \
  --rew-max-lambda-step 0.03 \
  --rew-min-lambda-step 0.01 \
  --rew-debug-reweight \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
