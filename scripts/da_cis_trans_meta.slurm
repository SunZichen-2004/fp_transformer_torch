#!/bin/bash
#SBATCH -J da_ct_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# IMDA cis/trans + MACE-MH-0 + 2D WT-MetaD (paper cv1, cv2=c8*c17)
# Trajectory + live scatter/FES in live/; single GPU; 4 walkers

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

MACE_MODEL=${MACE_MODEL:-mh-0}
REPLICAS=${REPLICAS:-16}
STEPS=${STEPS:-500000}

echo "host=$(hostname)  date=$(date)  METHOD=metad2d  MACE=$MACE_MODEL  R=$REPLICAS  STEPS=$STEPS"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'n_gpu', torch.cuda.device_count())"

EXP="$ROOT/experiments/DA_cis_trans"
OUT_DIR="$EXP/results/da_cis_trans_metad_${MACE_MODEL}_r${REPLICAS}_${SLURM_JOB_ID}"

python -u "$EXP/sample_meta.py" \
  --mace-model "${MACE_MODEL}" \
  --xyz "$EXP/reactant.xyz" \
  --mass-file "$EXP/mass.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --charge 0.0 \
  --spin 1.0 \
  --wall-d-max 8.0 \
  --wall-k 10.0 \
  --cell-length 40.0 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --metad-sigma 0.2,0.3 \
  --metad-height 0.02 \
  --metad-bias-factor 30.0 \
  --metad-deposit-pace 50 \
  --dump-every 200 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "live: ${OUT_DIR}/live/"
