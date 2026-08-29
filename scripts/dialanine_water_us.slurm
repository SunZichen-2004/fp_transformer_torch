#!/bin/bash
#SBATCH -J dial_wat_us2d
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Dialanine-water 2D US on φ/ψ (batched parallel windows)
#   16×16 = 256 windows, κ = 100 kcal/mol/rad², 100 ps
# Init: MetaD solute traj → CV match → TIP3P re-solvate (prepare_init.py)

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
N_PHI=${N_PHI:-16}
N_PSI=${N_PSI:-16}
REPLICAS=$((N_PHI * N_PSI))
STEPS=${STEPS:-100000}
RELAX_STEPS=${RELAX_STEPS:-2000}
KAPPA_KCAL=${KAPPA_KCAL:-100.0}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_CUTOFF=${PME_CUTOFF:-8.0}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/dialanine_water/mm_box}
EXP="$ROOT/experiments/dialanine_water_us"
META_TRAJ=${META_TRAJ:-$ROOT/experiments/dialanine_water_meta/results/dialanine_water_metad_r64_s0_87241/trajectory.xyz}
INIT_NPY=${INIT_NPY:-$EXP/init_us2d_r${REPLICAS}.npy}

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=US-2D seed=$SEED R=$REPLICAS (${N_PHI}x${N_PSI}) steps=$STEPS κ=$KAPPA_KCAL"
echo "INIT_NPY=$INIT_NPY META_TRAJ=$META_TRAJ"
nvidia-smi || true

if [[ ! -f "${INIT_NPY}" ]]; then
  echo "[prep] building init from MetaD traj ..."
  python -u "$EXP/prepare_init.py" \
    --traj-xyz "${META_TRAJ}" \
    --box-dir "${BOX_DIR}" \
    --n-phi "${N_PHI}" \
    --n-psi "${N_PSI}" \
    --seed "${SEED}" \
    --device cpu \
    --output "${INIT_NPY}"
fi

OUT_DIR="$EXP/results/dialanine_water_us2d_r${REPLICAS}_k${KAPPA_KCAL%.*}_s${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -u "$EXP/sample.py" \
  --box-dir "${BOX_DIR}" \
  --cv-file "${BOX_DIR}/backbone_cv.txt" \
  --n-phi "${N_PHI}" \
  --n-psi "${N_PSI}" \
  --steps "${STEPS}" \
  --relax-steps "${RELAX_STEPS}" \
  --seed "${SEED}" \
  --init-npy "${INIT_NPY}" \
  --kappa-kcal "${KAPPA_KCAL}" \
  --timestep 0.001 \
  --langevin-friction 10.0 \
  --temperature 300 \
  --pme-cutoff "${PME_CUTOFF}" \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --dump-every 500 \
  --update-pace 20 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)"
echo "out: ${OUT_DIR}"
ls -lt "${OUT_DIR}" | head -25
