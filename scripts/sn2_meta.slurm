#!/bin/bash
#SBATCH -J sn2_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# SN2 WT-MetaD: same FF/CV/walls as SN2_mace; R=32 walkers, T=300 K, 1 GPU

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT/scripts" || exit 1

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
REPLICAS=${REPLICAS:-32}
STEPS=${STEPS:-500000}

echo "host=$(hostname) date=$(date) SN2-MetaD MACE=$MACE_MODEL R=$REPLICAS T=300"
nvidia-smi || true

EXP="$ROOT/experiments/SN2_metadynamic"
SN2="$ROOT/experiments/SN2_mace"
OUT_DIR="$EXP/results/sn2_metad_${MACE_MODEL}_r${REPLICAS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --pdb "$SN2/sn2.pdb" \
  --cv-file "$SN2/cv_pairs.txt" \
  --mass-file "$SN2/sn2_mass.txt" \
  --init-npy "$SN2/init_coords_md_1024.npy" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --charge -1.0 \
  --spin 1.0 \
  --wall-d-max 3.5 \
  --wall-k 20.0 \
  --cell-length 30.0 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --metad-sigma 0.1,0.1 \
  --metad-height 0.02 \
  --metad-bias-factor 30.0 \
  --metad-deposit-pace 50 \
  --fes-d1-min 1.0 \
  --fes-d1-max 4.0 \
  --fes-d2-min 1.0 \
  --fes-d2-max 4.0 \
  --dump-every 500 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)  live=${OUT_DIR}/live/"
