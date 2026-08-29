#!/bin/bash
#SBATCH -J kde_fes_groups
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# KDE FES (d1,d2 in [1,4]) for stage groups 1-4 / 5-8 / 9-12 / 13-16, vs WT-MetaD.

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

DATA=${DATA:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736}
OUTBASE=${OUTBASE:-$DATA/kde_fes}
METAD_FES=${METAD_FES:-$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_86711/live/fes.npz}
BOUNDS=${BOUNDS:-1.0,4.0,1.0,4.0}
GRID_N=${GRID_N:-200}
DEVICE=${DEVICE:-cuda}
KDE=$ROOT/experiments/rearrangement_water_fpt/plot_kde_fes.py
CMP=$ROOT/experiments/rearrangement_water_fpt/plot_kde_fes_compare.py

echo "host=$(hostname) date=$(date)  DATA=$DATA  BOUNDS=$BOUNDS  DEVICE=$DEVICE"
nvidia-smi || true

for spec in "1-4:1,2,3,4" "5-8:5,6,7,8" "9-12:9,10,11,12" "13-16:13,14,15,16"; do
  tag="${spec%%:*}"
  stages="${spec##*:}"
  outdir="$OUTBASE/stages${tag}"
  mkdir -p "$outdir"
  echo "===== stages $tag ($stages) ===== $(date)"
  python -u "$KDE" \
    --data-dir "$DATA" \
    --stages "$stages" \
    --bounds "$BOUNDS" \
    --grid-n "$GRID_N" \
    --device "$DEVICE" \
    --output "$outdir/kde_fes.png"
  python -u "$CMP" \
    --kde-fes "$outdir/kde_fes.npz" \
    --metad-fes "$METAD_FES" \
    --out-dir "$outdir" \
    --d1-lim 1,4 \
    --d2-lim 1,4 \
    --method-label "KDE"
done

echo "done $(date)  OUTBASE=$OUTBASE"
