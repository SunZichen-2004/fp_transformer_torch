#!/bin/bash
# Refresh rmse_vs_time_g5 figures with the *latest* MetaD hills/FES,
# keeping existing sFPT/KDE curves unchanged.
#
# Usage (login node is fine; MetaD recompute is CPU-only-ish but numpy-heavy):
#   bash scripts/refresh_rmse_g5_metad_latest.sh
#
# Optional env:
#   METAD_DIR   explicit MetaD live/ dir (default: newest */live with hills+fes)
#   OUT_DIR     plot output dir (default: .../kde_fes/rmse_vs_time_g5)
#   FPT_DIR     FPT result dir (for wall-clock mtimes)
#   DEVICE      cpu|cuda (default cpu)
#   METAD_RATE  MetaD it/s for wall-clock axis (default 13.74)
#   NGPUS_META / NGPUS_FPT  legend labels (default 1 / 8)

set -eo pipefail
ROOT=/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master
cd "$ROOT" || exit 1

source /storage/gaoyiqinLab/sunzicheng/software/miniconda3/etc/profile.d/conda.sh
set +u
conda activate MACE_torchsim
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

FPT_DIR=${FPT_DIR:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736}
OUT_DIR=${OUT_DIR:-$FPT_DIR/kde_fes/rmse_vs_time_g5}
METAD_ROOT=${METAD_ROOT:-$ROOT/experiments/rearrangement_water_metadynamics/results}
DEVICE=${DEVICE:-cpu}
METAD_STRIDE=${METAD_STRIDE:-10}
METAD_RATE=${METAD_RATE:-13.74}
NGPUS_META=${NGPUS_META:-1}
NGPUS_FPT=${NGPUS_FPT:-8}

# Pick latest MetaD live/ that has both hills.npz and fes.npz
if [[ -z "${METAD_DIR:-}" ]]; then
  METAD_DIR=$(
    METAD_ROOT="$METAD_ROOT" python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["METAD_ROOT"])
cands = []
for live in root.glob("*/live"):
    if (live / "hills.npz").is_file() and (live / "fes.npz").is_file():
        cands.append(live)
if not cands:
    raise SystemExit("no MetaD live/ with hills.npz+fes.npz under " + str(root))
best = max(cands, key=lambda p: (p / "hills.npz").stat().st_mtime)
print(best)
PY
  )
fi

REUSE_NPZ=${REUSE_NPZ:-$OUT_DIR/fes_rmse_vs_time.npz}
if [[ ! -f "$REUSE_NPZ" ]]; then
  echo "ERROR: missing sFPT cache $REUSE_NPZ" >&2
  exit 1
fi

# Backup before overwrite
stamp=$(date +%Y%m%d_%H%M%S)
cp -a "$REUSE_NPZ" "$OUT_DIR/fes_rmse_vs_time.before_metad_${stamp}.npz"

echo "host=$(hostname) date=$(date)"
echo "METAD_DIR=$METAD_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "REUSE_NPZ=$REUSE_NPZ  (sFPT kept)"
python - <<PY
import numpy as np
from pathlib import Path
h = np.load(Path("$METAD_DIR") / "hills.npz")
nh = int(h["n_hills"]) if "n_hills" in h.files else int(h["centers"].shape[0])
print(f"[metad] n_hills={nh}  hills mtime={Path('$METAD_DIR/hills.npz').stat().st_mtime}")
old = np.load("$REUSE_NPZ")
print(f"[sFPT] n_groups={len(old['t_kde_ps'])}  t∈[{old['t_kde_ps'][0]},{old['t_kde_ps'][-1]}] ps")
PY

# 1) Recompute MetaD curves only; keep KDE arrays from REUSE_NPZ
python -u "$ROOT/experiments/rearrangement_water_fpt/plot_fes_rmse_vs_time.py" \
  --fpt-dir "$FPT_DIR" \
  --metad-dir "$METAD_DIR" \
  --out-dir "$OUT_DIR" \
  --reuse-kde-npz "$REUSE_NPZ" \
  --group-size 5 \
  --dt-group 2.5 \
  --bounds 1.0,4.0,1.0,4.0 \
  --grid-n 120 \
  --n-bins-1d 80 \
  --vmax-kcal 60 \
  --device "$DEVICE" \
  --n-replicas 64 \
  --deposit-pace 50 \
  --timestep 0.001 \
  --bias-factor 30 \
  --sigma 0.2,0.2 \
  --metad-stride-deposits "$METAD_STRIDE" \
  --xi-cut 0.0

# 2) Redraw dive-in / wall-clock / 1D collage (MetaD updated, sFPT reused)
python -u - <<PY
from pathlib import Path
import sys

ROOT = Path("$ROOT")
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))

from plot_g5_1d_grid_and_dg_zoom import (
    plot_deltaG_dive_in,
    plot_deltaG_vs_wall_clock,
    refresh_1d_collage_metad_only,
)

out = Path("$OUT_DIR")
fpt = Path("$FPT_DIR")
npz = out / "fes_rmse_vs_time.npz"
metad_fes = Path("$METAD_DIR") / "fes.npz"

plot_deltaG_dive_in(npz, out)
plot_deltaG_vs_wall_clock(
    npz,
    out,
    fpt_dir=fpt,
    metad_rate_it_s=float("$METAD_RATE"),
    metad_gpus=int("$NGPUS_META"),
    fpt_gpus=int("$NGPUS_FPT"),
)
plot_deltaG_vs_wall_clock(
    npz,
    out,
    fpt_dir=fpt,
    metad_rate_it_s=float("$METAD_RATE"),
    metad_gpus=int("$NGPUS_META"),
    fpt_gpus=int("$NGPUS_FPT"),
    ylim=(-20.0, 0.0),
    stem="deltaG_vs_wallclock_divein",
)
refresh_1d_collage_metad_only(
    out_dir=out,
    metad_fes=metad_fes,
    bounds=(1.0, 4.0, 1.0, 4.0),
    n_bins=80,
    temperature=300.0,
    vmax=60.0,
    ncols=4,
)
print("all refresh plots done")
PY

echo "done $(date)  out=$OUT_DIR"
ls -lt "$OUT_DIR"/*.{png,pdf,npz} 2>/dev/null | head -30
