#!/bin/bash
#SBATCH -J dial_90535_smooth
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/sFPT/convergence_320_90535/smooth/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/sFPT/convergence_320_90535/smooth/job.%j.err

# Rolling last-20% window: at t after N stages, P from stages [N-ceil(0.2 N), N).
# Same US / MetaD refs and t-grid (every 20 stages / 2 ps) as convergence_320_90535.

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

US_MBAR=$ROOT/experiments/dialanine_water_us/results/dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz
META1=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s0_89387/live
META2=$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s2_89392/live
FPT=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g4_s1_90535
OUT=$ROOT/experiments/dialanine_water/sFPT/convergence_320_90535/smooth

mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
n_stages=$(ls "$FPT"/stage*_lambda*_dihedral.npy 2>/dev/null | wc -l)
echo "[fpt] $FPT stages=$n_stages"

python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT" \
  --fpt-dirs "$FPT" \
  --metad-dirs "${META1},${META2}" \
  --us-mbar "$US_MBAR" \
  --group-size 20 \
  --dt-group 2.0 \
  --lambda-target 0.0625 \
  --fpt-rate 36.1 \
  --metad-rate 44.0 \
  --fpt-gpus 4 \
  --metad-gpus 1 \
  --collage-vmax 20 \
  --fes2d-vmax 20 \
  --device cpu \
  --sfpt-window lastfrac \
  --last-frac 0.2

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -30
