#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpu1
#SBATCH --qos=temp
#SBATCH -J dialanine_mace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=300

# Vacuum dialanine + MACE foundation + torch-sim BAOAB-FP
# FPT settings match dialanine_gaff.slurm (force field is the only difference).
# Bandwidth rescale: unnormalized K, metric=row_sum (no /B),
# adaptive band = median×[0.8,1.2] clipped to fixed_row_sum [2,4]
# Each λ-stage end: coords + phi-psi under $OUT_DIR/phi_psi/

set -eo pipefail
ROOT=/share/home/sunzc/work/Fokker_Plank/fp_transformer_torch
cd "$ROOT" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate MACE_torchsim
set +u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MACE_MODEL=${MACE_MODEL:-off-small}  # or mh-0
# Same default as gaff; override if GPU OOM, e.g. REPLICAS=256 sbatch ...


echo "host=$(hostname)  date=$(date)  MACE_MODEL=${MACE_MODEL}  REPLICAS=${REPLICAS}"
python -c "import torch; import torch_sim as ts; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch_sim', ts.__version__)"

EXP="$ROOT/experiments/dialanine_mace"
OUT_DIR="$EXP/results/dialanine_mace_${MACE_MODEL}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --pdb "$EXP/dialanine.pdb" \
  --cv-file "$EXP/backbone_cv.txt" \
  --mass-file "$EXP/dialanine_mass.txt" \
  --init-npy "$EXP/init_coords_2048.npy" \
  --replicas 512 \
  --steps 500 \
  --update-pace 20 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --noise-scale 0.0 \
  --cv-grad-clip-norm 0.0 \
  --temperature 300 \
  --rew-anneal-ref-temperature 300 \
  --rew-lambda-init 0.2 \
  --rew-lambda-max 0.4 \
  --rew-ess-threshold 0.97 \
  --rew-num-lambda-trials 10 \
  --rew-max-lambda-step 0.02 \
  --rew-min-lambda-step 0.001 \
  --rescale-per-replica \
  --rescale-adaptive-band-low 0.8 \
  --rescale-adaptive-band-high 1.2 \
  --rescale-adaptive-ref-quantile 0.5 \
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
echo "plots: ${OUT_DIR}/phi_psi/"
