#!/bin/bash
#SBATCH -J conv_g5_plots
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# convergence_analysis full rerun on compute node:
# 3×sFPT (86736,86938,87055) + 2×WT-MetaD (87059,87060) + US-MBAR(87040)
# ΔG/RMSE: ξ∈[-2,2]; RMSE ref=US; also F≤30 suite; wall/cost plots

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

OUT=$ROOT/experiments/rearrangement_water_fpt/convergence_analysis
US_DIR=$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g1_87040
US_MBAR=$US_DIR/mbar/fes_mbar.npz
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true

# Archive previous npz so KDE is fully recomputed (script auto-reuses if present)
STAMP=$(date +%Y%m%d_%H%M%S)
BK="$OUT/backup_$STAMP"
mkdir -p "$BK"
for f in fes_1d_all_groups.npz fes_rmse_vs_time.npz fes_rmse_F30.npz; do
  if [[ -f "$OUT/$f" ]]; then
    mv "$OUT/$f" "$BK/"
    echo "[backup] $f → $BK/"
  fi
done

# 1) Fresh US MBAR on 87040
echo "[us] running MBAR on $US_DIR"
python -u "$ROOT/experiments/rearrangement_water_US/mbar_fes.py" \
  --result-dir "$US_DIR" \
  --out-dir "$US_DIR/mbar" \
  --discard-ps 10 \
  --n-bins 80

# 2) Full convergence plots (GPU KDE + MetaD incremental + F≤30)
python -u "$ROOT/experiments/rearrangement_water_fpt/plot_convergence_analysis.py" \
  --out-dir "$OUT" \
  --us-mbar "$US_MBAR" \
  --device cuda \
  --group-size 5 \
  --dt-group 2.5 \
  --xi-lo -2 --xi-hi 2 --xi-cut 0 \
  --fpt-rate 4.20 \
  --fpt-gpus 8 \
  --metad-gpus 1 \
  --rmse-vmax-extra 30 \
  --metad-stride-deposits 10

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -40
