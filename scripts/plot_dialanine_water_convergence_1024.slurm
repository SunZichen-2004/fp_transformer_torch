#!/bin/bash
#SBATCH -J dial_1024_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/sFPT/convergence_1024/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/dialanine_water/sFPT/convergence_1024/job.%j.err

# Redo convergence_1024 with fair-compete 4-GPU seeds 91354 / 91355.
# WT-MetaD: same s0/s2 as dialanine_main_picture (89387 / 89392).
# Wall-clock from 4×H800 throughput of 91354 (~19.88 it/s); cost = wall-clock × 4.

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
FPT0=$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/dialanine_water_sfpt_r1024_g4_s0_91354
FPT1=$ROOT/experiments/dialanine_water/sFPT/fair_compete_result/dialanine_water_sfpt_r1024_g4_s2_91355
OUT=$ROOT/experiments/dialanine_water/sFPT/convergence_1024

mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
for d in "$FPT0" "$FPT1"; do
  n_stages=$(find "$d" -maxdepth 1 -name 'stage*_dihedral.npy' | wc -l)
  echo "[fpt] $d stages=$n_stages"
done

python -u "$ROOT/experiments/dialanine_water/plot_convergence_multi.py" \
  --out-dir "$OUT" \
  --fpt-dirs "${FPT0},${FPT1}" \
  --metad-dirs "${META1},${META2}" \
  --us-mbar "$US_MBAR" \
  --group-size 20 \
  --dt-group 2.0 \
  --lambda-target 0.0625 \
  --lambda-atol 1e-3 \
  --fpt-rate 19.88 \
  --metad-rate 44.0 \
  --fpt-gpus 4 \
  --metad-gpus 1 \
  --collage-vmax 20 \
  --fes2d-vmax 20 \
  --device cuda

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -30
