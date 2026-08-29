#!/bin/bash
#SBATCH -J mace_parity
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Parity (2gpu) + throughput at B=1024 for 1/2/8 GPUs
set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT"
source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u; conda activate MACE_torchsim; set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export TMPDIR=/storage/gaoyiqinLab/sunzicheng/tmp
mkdir -p "$TMPDIR"

nvidia-smi -L
echo "==== 2-GPU parity @128 ===="
python -u scripts/parity_sharded_mace.py --n-gpus 2 --replicas 128 --bench-iters 20
echo "==== 2-GPU bench @1024 ===="
python -u scripts/parity_sharded_mace.py --n-gpus 2 --replicas 1024 --bench-iters 20
echo "==== 8-GPU bench @1024 ===="
python -u scripts/parity_sharded_mace.py --n-gpus 8 --replicas 1024 --bench-iters 20
