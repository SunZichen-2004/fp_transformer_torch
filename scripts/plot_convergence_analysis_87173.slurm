#!/bin/bash
#SBATCH -J conv_87173
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# water convergence_analysis with sFPT 87173 only (+ MetaD 87059/87060 + US 87040)
# Prior convergence_analysis/ is moved aside before mkdir.

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
OUT=$EXP/convergence_analysis
US_DIR=$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g1_87040
US_MBAR=$US_DIR/mbar/fes_mbar.npz
FPT=$EXP/results/claisen_water_fpt_mh-0_g8_87173
META1=$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_87059/live
META2=$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_87060/live

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true

# Move old convergence_analysis aside, then fresh mkdir
if [[ -d "$OUT" ]]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  BK="${OUT}_before_87173_${STAMP}"
  mv "$OUT" "$BK"
  echo "[backup] $OUT → $BK"
fi
mkdir -p "$OUT"

# Reuse US MBAR if present; else compute
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

n_stages=$(ls "$FPT"/stage*_cv.npy 2>/dev/null | wc -l)
echo "[fpt] $FPT stages=$n_stages"

python -u "$EXP/plot_convergence_analysis.py" \
  --out-dir "$OUT" \
  --fpt-dirs "$FPT" \
  --metad-dirs "$META1,$META2" \
  --us-mbar "$US_MBAR" \
  --device cuda \
  --group-size 10 \
  --dt-group 1.0 \
  --xi-lo -2 --xi-hi 2 --xi-cut 0 \
  --fpt-rate 4.23 \
  --fpt-gpus 8 \
  --metad-gpus 1 \
  --rmse-vmax-extra 30 \
  --metad-stride-deposits 10

echo "done $(date)"
ls -lt "$OUT"/*.{png,npz} 2>/dev/null | head -40
