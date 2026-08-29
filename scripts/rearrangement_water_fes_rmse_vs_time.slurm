#!/bin/bash
#SBATCH -J fes_rmse_t
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# 1D FES RMSE vs time: FPT-KDE (3 stages / 1.5 ps) and WT-MetaD (hills).

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT/scripts" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

FPT_DIR=${FPT_DIR:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736}
METAD_DIR=${METAD_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_86711/live}
GROUP_SIZE=${GROUP_SIZE:-5}
# 500 steps/stage × 0.001 ps × GROUP_SIZE
DT_GROUP=${DT_GROUP:-$(python -c "print(${GROUP_SIZE}*0.5)")}
OUT_DIR=${OUT_DIR:-$FPT_DIR/kde_fes/rmse_vs_time_g${GROUP_SIZE}}
DEVICE=${DEVICE:-cuda}
GRID_N=${GRID_N:-120}
METAD_STRIDE=${METAD_STRIDE:-10}

echo "host=$(hostname) date=$(date)  FPT=$FPT_DIR  METAD=$METAD_DIR  OUT=$OUT_DIR  GROUP=$GROUP_SIZE DT=$DT_GROUP"
nvidia-smi || true
mkdir -p "$OUT_DIR"

python -u "$ROOT/experiments/rearrangement_water_fpt/plot_fes_rmse_vs_time.py" \
  --fpt-dir "$FPT_DIR" \
  --metad-dir "$METAD_DIR" \
  --out-dir "$OUT_DIR" \
  --group-size "$GROUP_SIZE" \
  --dt-group "$DT_GROUP" \
  --bounds 1.0,4.0,1.0,4.0 \
  --grid-n "$GRID_N" \
  --n-bins-1d 80 \
  --vmax-kcal 60 \
  --device "$DEVICE" \
  --n-replicas 64 \
  --deposit-pace 50 \
  --timestep 0.001 \
  --bias-factor 30 \
  --sigma 0.2,0.2 \
  --metad-stride-deposits "$METAD_STRIDE" \
  --xi-cut 0.0

echo "done $(date)  out=$OUT_DIR"
