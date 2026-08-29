#!/bin/bash
#SBATCH -J tetra_md_init
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_sFPT/job.%j.err

# Unbiased MD from tetra_alanine.pdb → init_coords_1024.npy for sFPT

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg

STEPS=${STEPS:-10000}
DUMP_EVERY=${DUMP_EVERY:-3}
DISCARD=${DISCARD:-2000}
N_INIT=${N_INIT:-1024}
SEED=${SEED:-0}

EXP="$ROOT/experiments/tetra_alanine_sFPT"
OUT="$EXP/results/md_init_s${SEED}_${SLURM_JOB_ID}"

echo "host=$(hostname) date=$(date) MD steps=$STEPS dump=$DUMP_EVERY discard=$DISCARD n_init=$N_INIT"
nvidia-smi || true

python -u "$EXP/run_md_init.py" \
  --pdb "$EXP/tetra_alanine.pdb" \
  --sponge-prefix "$EXP/tetra_alanine" \
  --cv-file "$EXP/backbone_cv.txt" \
  --steps "$STEPS" \
  --dump-every "$DUMP_EVERY" \
  --discard-steps "$DISCARD" \
  --n-init "$N_INIT" \
  --seed "$SEED" \
  --device cuda \
  --out-dir "$OUT"

echo "done $(date)"
ls -lt "$OUT" "$EXP"/init_coords_${N_INIT}.npy
