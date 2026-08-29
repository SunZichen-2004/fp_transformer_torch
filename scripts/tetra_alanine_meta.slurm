#!/bin/bash
#SBATCH -J tetra_meta
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_meta/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/tetra_alanine_meta/job.%j.err

# Vacuum tetra-alanine 3φ WT-MetaD, multi-walker R=64, 10 ns.
# Bias: periodic cubic B-spline grid (O(1) in n_hills). Live: Boltzmann-project
# 3D MetaD FES onto (φi, φj) every dump.

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
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

SEED=${SEED:-0}
REPLICAS=${REPLICAS:-64}
TOTAL_TIME=${TOTAL_TIME:-10000}
DUMP_EVERY=${DUMP_EVERY:-2000}
METAD_GRID_N=${METAD_GRID_N:-81}

SFP="$ROOT/experiments/tetra_alanine_sFPT"
EXP="$ROOT/experiments/tetra_alanine_meta"
OUT_DIR="$EXP/results/tetra_alanine_metad_bspline_r${REPLICAS}_s${SEED}_${SLURM_JOB_ID}"
INIT_NPY=${INIT_NPY:-$SFP/init_coords_1024.npy}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=WT-MetaD-3phi-bspline seed=$SEED R=$REPLICAS t=${TOTAL_TIME}ps dump_every=$DUMP_EVERY grid_n=$METAD_GRID_N"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

if [[ ! -f "$INIT_NPY" ]]; then
  echo "[error] missing $INIT_NPY" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

python -u "$EXP/sample.py" \
  --sponge-prefix "$SFP/tetra_alanine" \
  --cv-file "$SFP/backbone_phi.txt" \
  --pdb "$SFP/tetra_alanine.pdb" \
  --mass-file "$SFP/tetra_alanine_mass.txt" \
  --init-npy "$INIT_NPY" \
  --replicas "$REPLICAS" \
  --total-time "$TOTAL_TIME" \
  --seed "$SEED" \
  --baoab-dt 0.001 \
  --baoab-langevin-gamma 10.0 \
  --temperature 300 \
  --metad-sigma 0.35,0.35,0.35 \
  --metad-height 0.02 \
  --metad-bias-factor 15.0 \
  --metad-deposit-pace 50 \
  --metad-max-hills 0 \
  --metad-grid-n "$METAD_GRID_N" \
  --dump-every "$DUMP_EVERY" \
  --grid-n "$METAD_GRID_N" \
  --fes-vmax 16 \
  --dtype float64 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
echo "live: ${OUT_DIR}/live/"
ls -lt "${OUT_DIR}" | head -20
ls -lt "${OUT_DIR}/live" | head -15
