#!/bin/bash
#SBATCH -J metad_rerun_g5
#SBATCH -A gaoyiqin
#SBATCH -p gaoyiqinlab
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

# Refresh g5 plots using MetaD hills/FES from live_rerun (86711 resume).
# Keep sFPT/KDE curves; recompute MetaD time series; US remains ΔG reference.

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

FPT_DIR=${FPT_DIR:-$ROOT/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736}
OUT_DIR=${OUT_DIR:-$FPT_DIR/kde_fes/rmse_vs_time_g5}
METAD_DIR=${METAD_DIR:-$ROOT/experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_86711/live_rerun}
US_MBAR=${US_MBAR:-$ROOT/experiments/rearrangement_water_US/results/claisen_water_us1d_mh-0_r64_g4_86936/mbar/fes_mbar.npz}
REUSE_NPZ=${REUSE_NPZ:-$OUT_DIR/fes_rmse_vs_time.npz}
METAD_STRIDE=${METAD_STRIDE:-10}
METAD_RATE=${METAD_RATE:-13.74}

echo "host=$(hostname) date=$(date)"
echo "METAD_DIR=$METAD_DIR"
echo "OUT_DIR=$OUT_DIR"
python - <<PY
import numpy as np
from pathlib import Path
h=np.load(Path("$METAD_DIR")/"hills.npz")
n=int(h["n_hills"]) if "n_hills" in h.files else int(h["centers"].shape[0])
print(f"live_rerun n_hills={n}  deposits≈{n//64}  t≈{n/64*50*0.001:.1f} ps")
PY

stamp=$(date +%Y%m%d_%H%M%S)
cp -a "$REUSE_NPZ" "$OUT_DIR/fes_rmse_vs_time.before_rerun_${stamp}.npz"

# 1) MetaD curves from live_rerun; keep sFPT arrays
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
  --device cpu \
  --n-replicas 64 \
  --deposit-pace 50 \
  --timestep 0.001 \
  --bias-factor 30 \
  --sigma 0.2,0.2 \
  --metad-stride-deposits "$METAD_STRIDE" \
  --xi-cut 0.0

# 2) Fix KDE RMSE vs new MetaD 1D ref (from cached F1d_kde); restore US ΔG ref; redraw
export OUT_DIR FPT_DIR METAD_DIR US_MBAR METAD_RATE
python -u - <<'PY'
from pathlib import Path
import os
import sys
import numpy as np

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))

from plot_fes_rmse_vs_time import rmse_1d
from plot_g5_1d_grid_and_dg_zoom import (
    plot_deltaG_dive_in,
    plot_deltaG_vs_wall_clock,
    refresh_1d_collage_metad_only,
    us_delta_g_ref,
)

out = Path(os.environ["OUT_DIR"])
fpt = Path(os.environ["FPT_DIR"])
metad_fes = Path(os.environ["METAD_DIR"]) / "fes.npz"
us_mbar = Path(os.environ["US_MBAR"])
metad_rate = float(os.environ.get("METAD_RATE", "13.74"))
npz = out / "fes_rmse_vs_time.npz"
cache = out / "fes_1d_all_groups.npz"

z = dict(np.load(npz))
# recompute KDE RMSE against new MetaD ref if 1D cache matches
if cache.is_file():
    c = np.load(cache, allow_pickle=True)
    xi_c = np.asarray(c["xi"], dtype=np.float64)
    xi_z = np.asarray(z["xi"], dtype=np.float64)
    f_ref = np.asarray(z["F1d_metad_ref"], dtype=np.float64)
    f_kde = np.asarray(c["F1d_kde"], dtype=np.float64)
    if len(xi_c) == len(xi_z) and np.allclose(xi_c, xi_z):
        rmse = np.array(
            [rmse_1d(f_kde[i], f_ref, vmax=60.0) for i in range(f_kde.shape[0])],
            dtype=np.float64,
        )
        z["rmse_kde"] = rmse
        print(f"[fix] rmse_kde vs live_rerun MetaD  n={len(rmse)}  last={rmse[-1]:.4f}")
    else:
        print("[warn] xi mismatch; keep old rmse_kde")

dG_us, _, f1d_us = us_delta_g_ref(
    us_mbar, temperature=300.0, xi_cut=0.0, xi_query=np.asarray(z["xi"])
)
z["dG_us_ref"] = np.asarray(dG_us, dtype=np.float64)
z["F1d_us_ref"] = np.asarray(f1d_us, dtype=np.float64)
np.savez_compressed(npz, **z)
print(f"[us] ΔG_ref={dG_us:.4f}  MetaD_ref ΔG={float(z['dG_metad_ref']):.4f}")

# redraw RMSE plot with updated rmse_kde
from plot_fes_compare import nature_rcparams
import matplotlib.pyplot as plt

plt.rcParams.update(nature_rcparams())
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
ax.plot(z["t_metad_ps"], z["rmse_metad"], color="#2166ac", lw=1.3, label="WT-MetaD", zorder=2)
ax.plot(
    z["t_kde_ps"], z["rmse_kde"], color="#e67e22", lw=1.4, marker="o", ms=4.5,
    label="sFPT-KDE", zorder=3,
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"1D FES RMSE / kcal mol$^{-1}$")
ax.set_xlim(left=0.0)
ymax = float(np.nanmax(np.concatenate([z["rmse_kde"], z["rmse_metad"]])))
ax.set_ylim(bottom=-0.02 * max(ymax, 1.0), top=ymax * 1.08 if ymax > 0 else 1.0)
ax.legend(frameon=False, loc="best")
ax.tick_params(which="both", top=False, right=False)
fig.tight_layout()
fig.savefig(out / "fes_rmse_vs_time.png", dpi=200)
fig.savefig(out / "fes_rmse_vs_time.pdf")
plt.close(fig)

# ΔG vs t (full) — keep MetaD/sFPT curves; US ref line via dive-in helpers
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
ax.plot(z["t_metad_ps"], z["dG_metad"], color="#2166ac", lw=1.3, label="WT-MetaD", zorder=2)
ax.plot(
    z["t_kde_ps"], z["dG_kde"], color="#e67e22", lw=1.4, marker="o", ms=4.5,
    label="sFPT-KDE", zorder=3,
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$\Delta G = k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$")
ax.set_xlim(left=0.0)
ax.legend(frameon=False, loc="best")
ax.tick_params(which="both", top=False, right=False)
fig.tight_layout()
fig.savefig(out / "deltaG_vs_time.png", dpi=200)
fig.savefig(out / "deltaG_vs_time.pdf")
plt.close(fig)

plot_deltaG_dive_in(npz, out, dG_ref=float(dG_us), ref_label="US")
plot_deltaG_vs_wall_clock(
    npz, out, fpt_dir=fpt, metad_rate_it_s=metad_rate,
    metad_gpus=1, fpt_gpus=8, dG_ref=float(dG_us), ref_label="US",
)
plot_deltaG_vs_wall_clock(
    npz, out, fpt_dir=fpt, metad_rate_it_s=metad_rate,
    metad_gpus=1, fpt_gpus=8, ylim=(-20.0, 0.0),
    stem="deltaG_vs_wallclock_divein",
    dG_ref=float(dG_us), ref_label="US",
)
refresh_1d_collage_metad_only(
    out_dir=out,
    metad_fes=metad_fes,
    bounds=(1.0, 4.0, 1.0, 4.0),
    n_bins=80,
    temperature=300.0,
    vmax=60.0,
    ncols=4,
    us_mbar=us_mbar,
)
print("all plots refreshed with live_rerun MetaD + US ΔG ref")
PY

echo "done $(date)"
ls -lt "$OUT_DIR"/*.{png,npz} | head -20
