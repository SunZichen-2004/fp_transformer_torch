#!/bin/bash
#SBATCH -J conv_r3_g3
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/rearrangement_water_fpt/convergence_repeat3_gpu3/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/rearrangement_water_fpt/convergence_repeat3_gpu3/job.%j.err

# Same figure suite as convergence_analysis (87173 slurm), but 3× R=512 seed
# repeats on 4 GPU: 89108 (seed=10), 89109 (seed=0), 89110 (seed=100).
# MetaD 90440/90441 + US 87040.
# group=10 stages × 0.1 ps = 1.0 ps/group; wall rate ≈3.95 it/s (4 GPU logs).

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

EXP=$ROOT/experiments/rearrangement_water_fpt
OUT=$EXP/convergence_repeat3_gpu3
US_DIR=$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_merged_87040_90364_90365
US_MBAR=$US_DIR/mbar/fes_mbar.npz
FPT1=$EXP/results/claisen_water_fpt_mh-0_g4_89108
FPT2=$EXP/results/claisen_water_fpt_mh-0_g4_89109
FPT3=$EXP/results/claisen_water_fpt_mh-0_g4_89110
META1=$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_90440/live
META2=$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_90441/live

mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true

if [[ ! -f "$US_MBAR" ]]; then
  echo "[us] running MBAR on $US_DIR"
  python -u "$ROOT/experiments/rearrangement_water_US/mbar_fes.py" \
    --result-dir "$US_DIR" \
    --out-dir "$US_DIR/mbar" \
    --discard-ps 20 \
    --n-bins 80
else
  echo "[us] reuse $US_MBAR"
fi

for d in "$FPT1" "$FPT2" "$FPT3"; do
  n_stages=$(ls "$d"/stage*_cv.npy 2>/dev/null | wc -l)
  echo "[fpt] $d stages=$n_stages"
done

for f in fes_1d_all_groups.npz fes_rmse_vs_time.npz fes_rmse_F30.npz; do
  if [[ -f "$OUT/$f" ]]; then
    rm -f "$OUT/$f"
    echo "[clean] removed $OUT/$f"
  fi
done

python -u "$EXP/plot_convergence_analysis.py" \
  --out-dir "$OUT" \
  --fpt-dirs "$FPT1,$FPT2,$FPT3" \
  --metad-dirs "$META1,$META2" \
  --us-mbar "$US_MBAR" \
  --device cuda \
  --group-size 10 \
  --dt-group 1.0 \
  --bounds 1.0,8.5,1.0,8.5 \
  --xi-lo -2 --xi-hi 2 --xi-cut 0 \
  --fpt-rate 3.95 \
  --fpt-gpus 4 \
  --metad-gpus 1 \
  --rmse-vmax-extra 30 \
  --metad-stride-deposits 10

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -40
