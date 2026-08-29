#!/bin/bash
#SBATCH -J conv_cum_plots
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# convergence_analysis_2: same US/MetaD as analysis_1;
# sFPT = cumulative KDE with discard first 20% (eval every 5 stages)

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

OUT=$ROOT/experiments/rearrangement_water_fpt/convergence_analysis_2
US_DIR=$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g1_87040
US_MBAR=$US_DIR/mbar/fes_mbar.npz
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true

if [[ ! -f "$US_MBAR" ]]; then
  echo "[us] running MBAR on $US_DIR"
  python -u "$ROOT/experiments/rearrangement_water_US/mbar_fes.py" \
    --result-dir "$US_DIR" \
    --out-dir "$US_DIR/mbar" \
    --discard-ps 10 \
    --n-bins 80
else
  echo "[us] reuse $US_MBAR"
fi

python -u "$ROOT/experiments/rearrangement_water_fpt/plot_convergence_analysis.py" \
  --out-dir "$OUT" \
  --us-mbar "$US_MBAR" \
  --device cuda \
  --sfpt-mode cumulative \
  --discard-frac 0.2 \
  --dt-stage 0.5 \
  --eval-stride 5 \
  --xi-lo -2 --xi-hi 2 --xi-cut 0 \
  --fpt-rate 4.20 \
  --fpt-gpus 8 \
  --metad-gpus 1 \
  --rmse-vmax-extra 30 \
  --metad-stride-deposits 10

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -40
