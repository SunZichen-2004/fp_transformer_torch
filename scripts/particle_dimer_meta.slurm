#!/bin/bash
#SBATCH -J pd_meta_cpu
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
#SBATCH -o /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/particle_dimer_meta/job.%j.out
#SBATCH -e /storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master/experiments/particle_dimer_meta/job.%j.err

# CPU-only particle-dimer WT-MetaD. Init = sFPT pre-MD snapshot.
#
#   TEMP=1 REPLICAS=64 sbatch scripts/particle_dimer_meta.slurm

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=""

SEED=${SEED:-0}
REPLICAS=${REPLICAS:-64}
STEPS=${STEPS:-100000}
DT=${DT:-1e-4}
GAMMA=${GAMMA:-10.0}
TEMP=${TEMP:-1}
METAD_SIGMA=${METAD_SIGMA:-0.1}
METAD_HEIGHT=${METAD_HEIGHT:-0.1}
METAD_BIAS_FACTOR=${METAD_BIAS_FACTOR:-15.0}
DEPOSIT_PACE=${DEPOSIT_PACE:-50}
DUMP_EVERY=${DUMP_EVERY:-500}

SFPT_ROOT="$ROOT/experiments/particle_dimer/results"
case "${TEMP}" in
  1|1.0)
    INIT_NPY=${INIT_NPY:-$SFPT_ROOT/particle_dimer_sfpt_r512_lam0.05_T1_s0_89298/init_coords_from_pre_md.npy}
    ;;
  1.5)
    INIT_NPY=${INIT_NPY:-$SFPT_ROOT/particle_dimer_sfpt_r512_lam0.05_T1.5_s0_89299/init_coords_from_pre_md.npy}
    ;;
  2|2.0)
    INIT_NPY=${INIT_NPY:-$SFPT_ROOT/particle_dimer_sfpt_r512_lam0.05_T2_s0_89300/init_coords_from_pre_md.npy}
    ;;
  *)
    echo "unknown TEMP=${TEMP}; pass INIT_NPY=..." >&2
    exit 1
    ;;
esac

if [[ ! -f "$INIT_NPY" ]]; then
  echo "missing init npy: $INIT_NPY" >&2
  exit 1
fi

EXP="$ROOT/experiments/particle_dimer_meta"
JOBTAG="${SLURM_JOB_ID:-cpu}"
OUT_DIR="$EXP/results/particle_dimer_meta_r${REPLICAS}_T${TEMP}_s${SEED}_${JOBTAG}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-na}"
echo "METHOD=WT-MetaD-CPU seed=$SEED R=$REPLICAS steps=$STEPS T=$TEMP dt=$DT γ=$GAMMA σ=$METAD_SIGMA w0=$METAD_HEIGHT bias_factor=$METAD_BIAS_FACTOR init=$INIT_NPY"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"

python -u "$EXP/sample.py" \
  --replicas "${REPLICAS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --baoab-dt "${DT}" \
  --baoab-langevin-gamma "${GAMMA}" \
  --temperature "${TEMP}" \
  --kB 1.0 \
  --init-coords "${INIT_NPY}" \
  --metad-sigma "${METAD_SIGMA}" \
  --metad-height "${METAD_HEIGHT}" \
  --metad-bias-factor "${METAD_BIAS_FACTOR}" \
  --metad-deposit-pace "${DEPOSIT_PACE}" \
  --metad-max-hills 0 \
  --dump-every "${DUMP_EVERY}" \
  --verbose \
  --verbose-pace 50 \
  --device cpu \
  --output "${OUT_DIR}"

echo "done $(date)  out=${OUT_DIR}"
ls -lt "$OUT_DIR" | head -40
