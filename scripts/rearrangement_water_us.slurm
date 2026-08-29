#!/bin/bash
#SBATCH -J claisen_wat_us
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 7-00:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Solvated Claisen 1D US on ξ = d1 − d2 (ML/MM):
#   1) 1-replica plain MD relax
#   2) steered hop along ξ windows (0.2 ps / window)
#   3) 64-window production (100 ps), κ = 100 kcal/mol/Å²

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=${TMPDIR:-/storage/gaoyiqinLab/sunzicheng/tmp}
mkdir -p "$TMPDIR"

MACE_MODEL=${MACE_MODEL:-mh-0}
REPLICAS=${REPLICAS:-64}
STEPS=${STEPS:-100000}
RELAX_STEPS=${RELAX_STEPS:-2000}
HOP_PS=${HOP_PS:-0.5}
KAPPA_KCAL=${KAPPA_KCAL:-200.0}
XI_MIN=${XI_MIN:--3.5}
XI_MAX=${XI_MAX:-2.5}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_MESH=${PME_MESH:-1.5}
BOX_DIR=${BOX_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/mlmm_box}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-4}}}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#_cvd[@]}
fi

echo "host=$(hostname) date=$(date) Claisen-water-US-1D MACE=$MACE_MODEL R=$REPLICAS κ=${KAPPA_KCAL}kcal ξ∈[$XI_MIN,$XI_MAX] NGPUS=$NGPUS"
nvidia-smi || true
head -2 "$BOX_DIR/box.xyz" || true

EXP="$ROOT/experiments/rearrangement_water_US"
OUT_DIR="$EXP/results/claisen_water_us1d_${MACE_MODEL}_r${REPLICAS}_g${NGPUS}_${SLURM_JOB_ID}"

python -u "$EXP/sample.py" \
  --mace-model "${MACE_MODEL}" \
  --box-dir "${BOX_DIR}" \
  --cv-file "$EXP/cv_pairs.txt" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --relax-steps "${RELAX_STEPS}" \
  --hop-ps "${HOP_PS}" \
  --xi-min "${XI_MIN}" \
  --xi-max "${XI_MAX}" \
  --n-gpus "${NGPUS}" \
  --kappa-kcal "${KAPPA_KCAL}" \
  --charge 0.0 \
  --spin 1.0 \
  --pme-cutoff 9.0 \
  --pme-mesh-spacing "${PME_MESH}" \
  --electrostatics "${ELECTROSTATICS}" \
  --wall-d-max 8.0 \
  --wall-k 10.0 \
  --softmin-a 10.0 \
  --softmin-d-wall 2.4 \
  --softmin-k 10.0 \
  --update-pace 20 \
  --dump-every 500 \
  --timestep 0.001 \
  --langevin-friction 10.0 \
  --temperature 300 \
  --traj-solute-only \
  --max-xyz-frames 32 \
  --dtype float32 \
  --verbose \
  --verbose-pace 100 \
  --device cuda \
  --output "${OUT_DIR}"

echo "done $(date)  out=${OUT_DIR}"
