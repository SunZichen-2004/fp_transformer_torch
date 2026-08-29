#!/bin/bash
#SBATCH -J profile_mlmm
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPLICAS=${REPLICAS:-1024}
NGPUS=${NGPUS:-8}
ELECTROSTATICS=${ELECTROSTATICS:-pme}
PME_MESH=${PME_MESH:-1.5}

echo "host=$(hostname) date=$(date) profile R=$REPLICAS NGPUS=$NGPUS elec=$ELECTROSTATICS mesh=$PME_MESH"
nvidia-smi -L || true

python -u "$ROOT/scripts/profile_mlmm_components.py" \
  --replicas "${REPLICAS}" \
  --n-gpus "${NGPUS}" \
  --electrostatics "${ELECTROSTATICS}" \
  --pme-cutoff 9.0 \
  --pme-mesh-spacing "${PME_MESH}" \
  --warmup 3 \
  --repeats 8 \
  --dtype float32

echo "done $(date)"
