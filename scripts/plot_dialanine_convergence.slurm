#!/bin/bash
#SBATCH -J dial_conv
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Dialanine vacuum convergence_analysis (3 seeds):
#   sFPT: 87220 (s0) + 90437 (s2) + 90438 (s4)
#   WT-MetaD: 87211 (s0) + 90425 (s2) + 90426 (s4)
#   λ=0.1, 100 steps/stage → 0.1 ps; groups of 5 → 0.5 ps
#   reference = last 200 ps MetaD hills (mean over seeds); t-axis ≤ 25 ps

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

FPT0=$ROOT/experiments/dialanine/results/dialanine_gaff_s0_87220
FPT2=$ROOT/experiments/dialanine/results/dialanine_gaff_s2_90437
FPT4=$ROOT/experiments/dialanine/results/dialanine_gaff_s4_90438
METAD0=$ROOT/experiments/dialanine_meta/results/dialanine_metad_r64_s0_87211/live
METAD2=$ROOT/experiments/dialanine_meta/results/dialanine_metad_r64_s2_90425/live
METAD4=$ROOT/experiments/dialanine_meta/results/dialanine_metad_r64_s4_90426/live
OUT=$ROOT/experiments/dialanine/convergence_analysis
mkdir -p "$OUT"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true
for d in "$FPT0" "$FPT2" "$FPT4"; do
  n=$(ls "$d"/stage*_lambda0p100*_dihedral.npy 2>/dev/null | wc -l)
  echo "[fpt] $d stages=$n"
done
for d in "$METAD0" "$METAD2" "$METAD4"; do
  ls -lah "$d"/hills.npz "$d"/fes.npz
done

python -u "$ROOT/experiments/dialanine/plot_convergence_multi.py" \
  --fpt-dirs "${FPT0},${FPT2},${FPT4}" \
  --metad-dirs "${METAD0},${METAD2},${METAD4}" \
  --out-dir "$OUT" \
  --group-size 5 \
  --dt-group 0.5 \
  --device cuda \
  --fpt-rate 64.4 \
  --metad-rate 139.0 \
  --fpt-gpus 1 \
  --metad-gpus 1 \
  --bias-factor 15.0 \
  --sigma 0.35,0.35 \
  --metad-stride-deposits 10 \
  --metad-ref-last-ps 200 \
  --t-max 25 \
  --rmse-vmax-extra 8 \
  --xi-lo -3.141592653589793 \
  --xi-hi 3.141592653589793 \
  --xi-cut 0

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -40
