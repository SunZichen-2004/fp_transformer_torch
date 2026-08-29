"""Smoke: dialanine GAFF + BAOAB-FP — torch vs pre-dumped JAX reference (50 steps).

Generate the reference first (uv / JAX venv)::

  bash scripts/smoke_dialanine_jax_torch_parity.sh

Or only the dump step::

  cd ../fp_0429_final/fp-transformer && uv run python \\
    ../../fp_transformer_torch/scripts/dump_jax_dialanine_parity.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "experiments" / "dialanine" / "parity_ref"
COMPARE = ROOT / "scripts" / "compare_torch_dialanine_parity.py"
SMOKE_SH = ROOT / "scripts" / "smoke_dialanine_jax_torch_parity.sh"


@pytest.mark.skipif(not COMPARE.is_file(), reason="compare script missing")
def test_dialanine_torch_matches_jax_parity_ref():
    if not (REF / "coordinates.npy").is_file():
        pytest.skip(
            f"JAX parity_ref missing at {REF}; run "
            f"`bash {SMOKE_SH}` (uv dump + MACE compare) first"
        )
    proc = subprocess.run(
        [sys.executable, str(COMPARE), "--ref-dir", str(REF)],
        cwd=str(ROOT),
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
    assert proc.returncode == 0, proc.stderr or proc.stdout
