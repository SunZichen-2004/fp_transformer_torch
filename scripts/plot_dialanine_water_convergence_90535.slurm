#!/bin/bash
#SBATCH -J dial_90535_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/sFPT/convergence_320_90535/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/sFPT/convergence_320_90535/job.%j.err

# 3-seed mean±std for r320 4-GPU:
#   90535 seed=1, 90585 seed=2, 90586 seed=4
# Same US / MetaD refs and grouping as sFPT/convergence_320.

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
FPT1=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g4_s1_90535
FPT2=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g4_s2_90585
FPT4=$ROOT/experiments/dialanine_water/sFPT/results/dialanine_water_sfpt_r320_g4_s4_90586
OUT=$ROOT/experiments/dialanine_water/sFPT/convergence_320_90535

mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
for d in "$FPT1" "$FPT2" "$FPT4"; do
  n_stages=$(ls "$d"/stage*_lambda*_dihedral.npy 2>/dev/null | wc -l)
  echo "[fpt] $d stages=$n_stages"
done

python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT" \
  --fpt-dirs "${FPT1},${FPT2},${FPT4}" \
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
  --device cpu

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -30
