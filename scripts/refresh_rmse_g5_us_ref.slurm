#!/bin/bash
#SBATCH -J us_ref_g5_plots
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Redraw rmse_vs_time_g5 dive-in / wall-clock / 1D collage with US-MBAR as ΔG reference.

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

FPT_DIR=${FPT_DIR:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736}
OUT_DIR=${OUT_DIR:-$FPT_DIR/kde_fes/rmse_vs_time_g5}
US_MBAR=${US_MBAR:-$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g4_86936/mbar/fes_mbar.npz}

echo "host=$(hostname) date=$(date)"
echo "OUT_DIR=$OUT_DIR"
echo "US_MBAR=$US_MBAR"

python -u "$ROOT/experiments/rearrangement_water_fpt/plot_g5_1d_grid_and_dg_zoom.py" \
  --out-dir "$OUT_DIR" \
  --fpt-dir "$FPT_DIR" \
  --us-mbar "$US_MBAR" \
  --device cpu \
  --xi-cut 0.0

echo "done $(date)"
ls -lt "$OUT_DIR"/deltaG_vs_time_divein.png \
       "$OUT_DIR"/deltaG_vs_wallclock_divein.png \
       "$OUT_DIR"/fes_1d_all_groups.png
