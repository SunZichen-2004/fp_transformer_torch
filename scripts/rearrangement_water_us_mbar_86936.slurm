#!/bin/bash
#SBATCH -J us_mbar_86936
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# CPU-only: pymbar reconstruct F(ξ) for water US job 86936

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
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

RESULT=${RESULT:-$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g4_86936}
OUT=${OUT:-$RESULT/mbar}
DISCARD_PS=${DISCARD_PS:-20}
N_BINS=${N_BINS:-80}

echo "host=$(hostname) date=$(date) MBAR US result=$RESULT out=$OUT"
python -c "import pymbar, numpy; print('pymbar', pymbar.__version__, 'numpy', numpy.__version__)"

python -u "$ROOT/experiments/rearrangement_water_US/mbar_fes.py" \
  --result-dir "$RESULT" \
  --out-dir "$OUT" \
  --discard-ps "$DISCARD_PS" \
  --n-bins "$N_BINS" \
  --verbose

echo "done $(date)"
ls -lah "$OUT"
