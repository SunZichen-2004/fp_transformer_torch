#!/bin/bash
#SBATCH -J da_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# DA + MACE-MH-0 + SoftDistanceS12 + nonbond lower walls + multi-walker WT-MetaD
# 16 walkers, shared bias, single GPU
# σ=(0.05,0.05), γ=30, FES grid s1∈[0,3], s2∈[-1,1]

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
STEPS=${STEPS:-800000}

echo "host=$(hostname)  date=$(date)  METHOD=metad  MACE=$MACE_MODEL  R=$REPLICAS  STEPS=$STEPS"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'n_gpu', torch.cuda.device_count())"

EXP="$ROOT/experiments/DA_mace_meta"
OUT_DIR="$EXP/results/da_metad_${MACE_MODEL}_r${REPLICAS}_${SLURM_JOB_ID}"
INIT_NPY=${INIT_NPY:-$EXP/init_coords_md_1024.npy}

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --xyz "$EXP/DA_tropone_dimethylfulvene.xyz" \
  --cv-file "$EXP/cv_pairs.txt" \
  --nonbond-file "$EXP/nonbond_pairs.txt" \
  --mass-file "$EXP/da_mass.txt" \
  --init-npy "${INIT_NPY}" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --charge 0.0 \
  --spin 1.0 \
  --d-ref 1.7 \
  --soft-n 6 \
  --soft-m 8 \
  --wall-d-max 8.0 \
  --wall-k 10.0 \
  --nonbond-r-min 2.4 \
  --nonbond-k-kcal 100.0 \
  --cell-length 40.0 \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --metad-sigma 0.05,0.05 \
  --metad-height 0.02 \
  --metad-bias-factor 30.0 \
  --metad-deposit-pace 50 \
  --fes-s1-min 0.0 \
  --fes-s1-max 3.0 \
  --fes-s2-min -1.0 \
  --fes-s2-max 1.0 \
  --dump-every 500 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "live: ${OUT_DIR}/live/"
