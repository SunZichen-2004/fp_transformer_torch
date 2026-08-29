"""Mechanical embedding ML/MM potential (MACE + classical MM, PBC).

Energy (metal units: Å, eV)::

    E = E_MACE(R_ML)
      + E_bonded^MM(R_MM)
      + E_LJ(R_MM-MM, R_ML-MM)
      + E_elec(R_all; q^MM)
      - E_elec(R_ML; q_ML)

``E_elec`` is **PME** (default, nvalchemiops), **DSF**, or **torchpme**
(lab-cosmo ``torchpme`` P3M/PME; Warp-free, multi-GPU replica sharding).

Notes
-----
* All atoms carry MM parameters (charges, LJ σ/ε, bonded, exclusions).
* ML intramolecular interactions come from MACE only (no ML–ML LJ / MM bonded).
* Electrostatics: full-system MM charges (ML atoms as point charges); classical
  ML–ML electrostatics removed by subtracting the same model on the ML subset.
* PME: Amber exclusions via bare-Coulomb correction for MM-touching pairs.
* DSF: real-space only; MM-touching excluded pairs are masked in the neighbor
  list (ML–ML exclusions stay and are removed by the ML correction, as in PME).
* Heterogeneous LJ uses Lorentz–Berthelot mixing with a PBC neighbor list.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.units import UnitConversion

from .gaff_utils import convert_sponge_input_to_dict

# nvalchemiops PME custom-op registration is not thread-safe on first call.
_ELEC_OPS_LOCK = threading.Lock()
_ELEC_OPS_READY = False


def _ensure_electrostatics_ops() -> None:
    """Register PME/Ewald Torch custom ops once (safe under parallel_apply)."""
    global _ELEC_OPS_READY
    if _ELEC_OPS_READY:
        return
    with _ELEC_OPS_LOCK:
        if _ELEC_OPS_READY:
            return
        from nvalchemiops.torch.interactions.electrostatics._registration import (
            ensure_electrostatics_ops_registered,
        )

        ensure_electrostatics_ops_registered()
        _ELEC_OPS_READY = True

# ---------------------------------------------------------------------------
# Unit conversions (Amber/SPONGE kcal·mol⁻¹ → metal eV)
# ---------------------------------------------------------------------------

# SPONGE stores charges such that E_kcal = q_i q_j / r  (Coulomb constant absorbed).
SPONGE_COULOMB_K_KCAL = 332.0637130
SPONGE_CHARGE_SCALE = math.sqrt(SPONGE_COULOMB_K_KCAL)

# 1 eV = 23.060547830619027 kcal/mol (consistent with ke_eV ≈ ke_kcal / 23.06)
KCAL_PER_MOL_TO_EV = float(UnitConversion.e2_per_Ang_to_eV) / SPONGE_COULOMB_K_KCAL
KE_EV = float(UnitConversion.e2_per_Ang_to_eV)  # ≈ 14.399645 eV·Å / e²

# ---------------------------------------------------------------------------
# TIP3P (Amber) — flexible (no SETTLE)
# ---------------------------------------------------------------------------

TIP3P_PARAMS: dict[str, Any] = {
    "OW_charge": -0.834,
    "HW_charge": 0.417,
    "OW_sigma": 3.1507,  # Å
    "OW_epsilon_kcal": 0.1521,
    "HW_sigma": 0.0,
    "HW_epsilon_kcal": 0.0,
    "OH_k_kcal": 553.0,  # kcal/mol/Å²
    "OH_r0": 0.9572,  # Å
    "HOH_k_kcal": 100.0,  # kcal/mol/rad²
    "HOH_theta0_deg": 104.52,
    "OW_mass": 16.0,
    "HW_mass": 1.008,
    "OW_Z": 8,
    "HW_Z": 1,
}


def tip3p_epsilon_ev() -> tuple[float, float]:
    ow = float(TIP3P_PARAMS["OW_epsilon_kcal"]) * KCAL_PER_MOL_TO_EV
    hw = float(TIP3P_PARAMS["HW_epsilon_kcal"]) * KCAL_PER_MOL_TO_EV
    return ow, hw


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass
class MLMMTopology:
    """Per-system classical topology for mechanical-embedding ML/MM.

    All tensors describe a single system of ``n_atoms`` atoms. Batched MD
    replicas reuse the same topology via local index ``atom % n_atoms``.
    """

    ml_mask: torch.Tensor  # [N] bool
    charges: torch.Tensor  # [N] elementary charge e
    sigma: torch.Tensor  # [N] Å
    epsilon: torch.Tensor  # [N] eV
    atomic_numbers: torch.Tensor  # [N] long
    masses: torch.Tensor  # [N]

    bond_idx: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2, dtype=torch.long))
    bond_k: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    bond_r0: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    angle_idx: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 3, dtype=torch.long))
    angle_k: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    angle_theta0: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    dihedral_idx: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 4, dtype=torch.long))
    dihedral_k: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    dihedral_n: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    dihedral_phi0: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    # Excluded pairs for Coulomb / LJ (local atom indices, unordered i<j preferred)
    exclude_i: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    exclude_j: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))

    # Optional 1–4 scaling (MM region only); factors multiply full LJ / Coulomb pair
    nb14_i: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    nb14_j: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    nb14_lj_factor: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    nb14_q_factor: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    cell: torch.Tensor | None = None  # optional default [3,3] Å

    def __post_init__(self) -> None:
        n = int(self.ml_mask.numel())
        for name in ("charges", "sigma", "epsilon", "atomic_numbers", "masses"):
            if int(getattr(self, name).numel()) != n:
                raise ValueError(f"{name} length must equal n_atoms={n}")
        self.ml_mask = self.ml_mask.bool().reshape(n)
        self.atomic_numbers = self.atomic_numbers.long().reshape(n)

    @property
    def n_atoms(self) -> int:
        return int(self.ml_mask.numel())

    @property
    def ml_indices(self) -> torch.Tensor:
        return torch.nonzero(self.ml_mask, as_tuple=False).reshape(-1)

    @property
    def mm_indices(self) -> torch.Tensor:
        return torch.nonzero(~self.ml_mask, as_tuple=False).reshape(-1)

    def to(self, device: torch.device, dtype: torch.dtype) -> "MLMMTopology":
        def _f(t: torch.Tensor) -> torch.Tensor:
            if t.dtype in (torch.long, torch.int32, torch.int64, torch.bool):
                return t.to(device=device)
            return t.to(device=device, dtype=dtype)

        return MLMMTopology(
            ml_mask=self.ml_mask.to(device=device),
            charges=_f(self.charges),
            sigma=_f(self.sigma),
            epsilon=_f(self.epsilon),
            atomic_numbers=self.atomic_numbers.to(device=device),
            masses=_f(self.masses),
            bond_idx=self.bond_idx.to(device=device, dtype=torch.long),
            bond_k=_f(self.bond_k),
            bond_r0=_f(self.bond_r0),
            angle_idx=self.angle_idx.to(device=device, dtype=torch.long),
            angle_k=_f(self.angle_k),
            angle_theta0=_f(self.angle_theta0),
            dihedral_idx=self.dihedral_idx.to(device=device, dtype=torch.long),
            dihedral_k=_f(self.dihedral_k),
            dihedral_n=self.dihedral_n.to(device=device, dtype=torch.long),
            dihedral_phi0=_f(self.dihedral_phi0),
            exclude_i=self.exclude_i.to(device=device, dtype=torch.long),
            exclude_j=self.exclude_j.to(device=device, dtype=torch.long),
            nb14_i=self.nb14_i.to(device=device, dtype=torch.long),
            nb14_j=self.nb14_j.to(device=device, dtype=torch.long),
            nb14_lj_factor=_f(self.nb14_lj_factor),
            nb14_q_factor=_f(self.nb14_q_factor),
            cell=None if self.cell is None else _f(self.cell),
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "ml_mask": self.ml_mask.cpu().numpy().astype(np.bool_),
            "charges": self.charges.cpu().numpy(),
            "sigma": self.sigma.cpu().numpy(),
            "epsilon": self.epsilon.cpu().numpy(),
            "atomic_numbers": self.atomic_numbers.cpu().numpy().astype(np.int64),
            "masses": self.masses.cpu().numpy(),
            "bond_idx": self.bond_idx.cpu().numpy().astype(np.int64),
            "bond_k": self.bond_k.cpu().numpy(),
            "bond_r0": self.bond_r0.cpu().numpy(),
            "angle_idx": self.angle_idx.cpu().numpy().astype(np.int64),
            "angle_k": self.angle_k.cpu().numpy(),
            "angle_theta0": self.angle_theta0.cpu().numpy(),
            "dihedral_idx": self.dihedral_idx.cpu().numpy().astype(np.int64),
            "dihedral_k": self.dihedral_k.cpu().numpy(),
            "dihedral_n": self.dihedral_n.cpu().numpy().astype(np.int64),
            "dihedral_phi0": self.dihedral_phi0.cpu().numpy(),
            "exclude_i": self.exclude_i.cpu().numpy().astype(np.int64),
            "exclude_j": self.exclude_j.cpu().numpy().astype(np.int64),
            "nb14_i": self.nb14_i.cpu().numpy().astype(np.int64),
            "nb14_j": self.nb14_j.cpu().numpy().astype(np.int64),
            "nb14_lj_factor": self.nb14_lj_factor.cpu().numpy(),
            "nb14_q_factor": self.nb14_q_factor.cpu().numpy(),
        }
        if self.cell is not None:
            payload["cell"] = self.cell.cpu().numpy()
        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: str | Path) -> "MLMMTopology":
        data = np.load(Path(path), allow_pickle=False)

        def t(key: str, dtype=None) -> torch.Tensor:
            arr = data[key]
            ten = torch.as_tensor(arr)
            return ten if dtype is None else ten.to(dtype=dtype)

        cell = torch.as_tensor(data["cell"]) if "cell" in data.files else None
        return MLMMTopology(
            ml_mask=t("ml_mask", torch.bool),
            charges=t("charges"),
            sigma=t("sigma"),
            epsilon=t("epsilon"),
            atomic_numbers=t("atomic_numbers", torch.long),
            masses=t("masses"),
            bond_idx=t("bond_idx", torch.long).reshape(-1, 2)
            if data["bond_idx"].size
            else torch.zeros(0, 2, dtype=torch.long),
            bond_k=t("bond_k"),
            bond_r0=t("bond_r0"),
            angle_idx=t("angle_idx", torch.long).reshape(-1, 3)
            if data["angle_idx"].size
            else torch.zeros(0, 3, dtype=torch.long),
            angle_k=t("angle_k"),
            angle_theta0=t("angle_theta0"),
            dihedral_idx=t("dihedral_idx", torch.long).reshape(-1, 4)
            if data["dihedral_idx"].size
            else torch.zeros(0, 4, dtype=torch.long),
            dihedral_k=t("dihedral_k"),
            dihedral_n=t("dihedral_n", torch.long),
            dihedral_phi0=t("dihedral_phi0"),
            exclude_i=t("exclude_i", torch.long),
            exclude_j=t("exclude_j", torch.long),
            nb14_i=t("nb14_i", torch.long),
            nb14_j=t("nb14_j", torch.long),
            nb14_lj_factor=t("nb14_lj_factor"),
            nb14_q_factor=t("nb14_q_factor"),
            cell=cell,
        )


def sponge_charges_to_elementary(q_sponge: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert SPONGE absorbed-constant charges → elementary charge ``e``."""
    q = np.asarray(q_sponge, dtype=np.float64)
    return q / SPONGE_CHARGE_SCALE


def _empty_long(n: int = 0, cols: int | None = None) -> torch.Tensor:
    if cols is None:
        return torch.zeros(n, dtype=torch.long)
    return torch.zeros(n, cols, dtype=torch.long)


def _filter_bonded_mm_only(
    idx: np.ndarray,
    k: np.ndarray,
    extra: list[np.ndarray],
    ml_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Keep terms whose *all* atoms lie in the MM region."""
    if idx.size == 0:
        return idx, k, extra
    mm = ~ml_mask.astype(bool)
    keep = np.all(mm[idx], axis=1)
    return idx[keep], k[keep], [e[keep] for e in extra]


def _exclude_pairs_from_matrix(ex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense exclude matrix (1=interact, 0=exclude) → upper-triangle exclude pairs."""
    n = ex.shape[0]
    ii, jj = np.where(np.triu(ex < 0.5, k=1))
    return ii.astype(np.int64), jj.astype(np.int64)


def build_tip3p_water_topology(
    n_waters: int,
    *,
    atom_offset: int = 0,
) -> dict[str, np.ndarray]:
    """Build TIP3P parameters / bonded / excludes for ``n_waters`` molecules.

    Atom order per water: O, H1, H2. Global indices start at ``atom_offset``.
    """
    n_atoms = 3 * int(n_waters)
    charges = np.zeros(n_atoms, dtype=np.float64)
    sigma = np.zeros(n_atoms, dtype=np.float64)
    epsilon = np.zeros(n_atoms, dtype=np.float64)
    masses = np.zeros(n_atoms, dtype=np.float64)
    zs = np.zeros(n_atoms, dtype=np.int64)
    ow_eps, hw_eps = tip3p_epsilon_ev()

    bonds_a: list[int] = []
    bonds_b: list[int] = []
    bonds_k: list[float] = []
    bonds_r: list[float] = []
    ang_a: list[int] = []
    ang_b: list[int] = []
    ang_c: list[int] = []
    ang_k: list[float] = []
    ang_t: list[float] = []
    excl_i: list[int] = []
    excl_j: list[int] = []

    oh_k = float(TIP3P_PARAMS["OH_k_kcal"]) * KCAL_PER_MOL_TO_EV
    oh_r = float(TIP3P_PARAMS["OH_r0"])
    hoh_k = float(TIP3P_PARAMS["HOH_k_kcal"]) * KCAL_PER_MOL_TO_EV
    hoh_t = math.radians(float(TIP3P_PARAMS["HOH_theta0_deg"]))

    for w in range(int(n_waters)):
        o = atom_offset + 3 * w
        h1 = o + 1
        h2 = o + 2
        lo = 3 * w
        charges[lo] = TIP3P_PARAMS["OW_charge"]
        charges[lo + 1] = TIP3P_PARAMS["HW_charge"]
        charges[lo + 2] = TIP3P_PARAMS["HW_charge"]
        sigma[lo] = TIP3P_PARAMS["OW_sigma"]
        sigma[lo + 1] = TIP3P_PARAMS["HW_sigma"]
        sigma[lo + 2] = TIP3P_PARAMS["HW_sigma"]
        epsilon[lo] = ow_eps
        epsilon[lo + 1] = hw_eps
        epsilon[lo + 2] = hw_eps
        masses[lo] = TIP3P_PARAMS["OW_mass"]
        masses[lo + 1] = TIP3P_PARAMS["HW_mass"]
        masses[lo + 2] = TIP3P_PARAMS["HW_mass"]
        zs[lo] = TIP3P_PARAMS["OW_Z"]
        zs[lo + 1] = TIP3P_PARAMS["HW_Z"]
        zs[lo + 2] = TIP3P_PARAMS["HW_Z"]

        for h in (h1, h2):
            bonds_a.append(o)
            bonds_b.append(h)
            bonds_k.append(oh_k)
            bonds_r.append(oh_r)
        ang_a.append(h1)
        ang_b.append(o)
        ang_c.append(h2)
        ang_k.append(hoh_k)
        ang_t.append(hoh_t)
        # intramolecular exclusions: O-H1, O-H2, H1-H2
        for a, b in ((o, h1), (o, h2), (h1, h2)):
            excl_i.append(min(a, b))
            excl_j.append(max(a, b))

    return {
        "charges": charges,
        "sigma": sigma,
        "epsilon": epsilon,
        "masses": masses,
        "atomic_numbers": zs,
        "Ba": np.asarray(bonds_a, dtype=np.int64),
        "Bb": np.asarray(bonds_b, dtype=np.int64),
        "Bk": np.asarray(bonds_k, dtype=np.float64),
        "Br": np.asarray(bonds_r, dtype=np.float64),
        "Aa": np.asarray(ang_a, dtype=np.int64),
        "Ab": np.asarray(ang_b, dtype=np.int64),
        "Ac": np.asarray(ang_c, dtype=np.int64),
        "Ak": np.asarray(ang_k, dtype=np.float64),
        "At": np.asarray(ang_t, dtype=np.float64),
        "exclude_i": np.asarray(excl_i, dtype=np.int64),
        "exclude_j": np.asarray(excl_j, dtype=np.int64),
    }


def identify_water_residues(
    resnames: list[str] | np.ndarray,
    *,
    water_names: tuple[str, ...] = ("WAT", "HOH", "TIP3", "SOL", "TP3"),
) -> np.ndarray:
    """Return boolean mask over residues that are water."""
    names = np.asarray(resnames)
    water = np.zeros(len(names), dtype=bool)
    upper = {n.upper() for n in water_names}
    for i, name in enumerate(names):
        water[i] = str(name).strip().upper() in upper
    return water


def build_topology_from_sponge_and_tip3p(
    sponge_prefix: str | Path,
    *,
    n_waters: int,
    ml_atom_indices: list[int] | np.ndarray | None = None,
    solute_is_ml: bool = True,
    cell: np.ndarray | torch.Tensor | None = None,
) -> MLMMTopology:
    """Merge SPONGE solute MM parameters with TIP3P solvent.

    Layout: ``[solute atoms | water atoms (O,H,H)×n_waters]``.

    If ``solute_is_ml`` (default), all solute atoms are the ML region and water
    is MM. Bonded terms involving any ML atom are dropped (mechanical embedding,
    no link atoms). Solute exclusions are kept for PME bookkeeping but ML–ML
    Coulomb is removed via the PME(ML) correction regardless.
    """
    prefix = str(sponge_prefix)
    raw = convert_sponge_input_to_dict(prefix)
    n_solute = int(len(raw["C"]))
    n_waters = int(n_waters)
    n_water_atoms = 3 * n_waters
    n_atoms = n_solute + n_water_atoms

    charges = np.zeros(n_atoms, dtype=np.float64)
    sigma = np.zeros(n_atoms, dtype=np.float64)
    epsilon = np.zeros(n_atoms, dtype=np.float64)
    masses = np.ones(n_atoms, dtype=np.float64)
    zs = np.zeros(n_atoms, dtype=np.int64)

    charges[:n_solute] = sponge_charges_to_elementary(raw["C"])
    sigma[:n_solute] = np.asarray(raw["S"], dtype=np.float64)
    epsilon[:n_solute] = np.asarray(raw["E"], dtype=np.float64) * KCAL_PER_MOL_TO_EV

    # Infer solute Z / mass from SPONGE mass file if present, else leave placeholders
    mass_path = Path(prefix + "_mass.txt")
    if mass_path.is_file():
        m = np.loadtxt(mass_path).reshape(-1)
        if int(m[0]) == n_solute and m.size == n_solute + 1:
            m = m[1:]
        masses[:n_solute] = m[:n_solute]
    z_path = Path(prefix + "_atomic_number.txt")
    if z_path.is_file():
        zs[:n_solute] = np.loadtxt(z_path, dtype=np.int64).reshape(-1)[:n_solute]
    else:
        # crude: O if sigma~3.0 and heavy eps, else C/H by mass
        for i in range(n_solute):
            if masses[i] > 15.5:
                zs[i] = 8
            elif masses[i] > 11.5:
                zs[i] = 6
            elif masses[i] > 13.5:
                zs[i] = 7
            else:
                zs[i] = 1

    water = build_tip3p_water_topology(n_waters, atom_offset=n_solute)
    charges[n_solute:] = water["charges"]
    sigma[n_solute:] = water["sigma"]
    epsilon[n_solute:] = water["epsilon"]
    masses[n_solute:] = water["masses"]
    zs[n_solute:] = water["atomic_numbers"]

    ml_mask = np.zeros(n_atoms, dtype=bool)
    if ml_atom_indices is not None:
        ml_mask[np.asarray(ml_atom_indices, dtype=np.int64)] = True
    elif solute_is_ml:
        ml_mask[:n_solute] = True

    # Solute bonded (SPONGE, kcal → eV), then keep MM-only
    bond_idx = np.stack([raw["Ba"], raw["Bb"]], axis=1).astype(np.int64)
    bond_k = np.asarray(raw["Bk"], dtype=np.float64) * KCAL_PER_MOL_TO_EV
    bond_r0 = np.asarray(raw["Br"], dtype=np.float64)
    bond_idx, bond_k, (bond_r0,) = _filter_bonded_mm_only(
        bond_idx, bond_k, [bond_r0], ml_mask
    )

    angle_idx = np.stack([raw["Aa"], raw["Ab"], raw["Ac"]], axis=1).astype(np.int64)
    angle_k = np.asarray(raw["Ak"], dtype=np.float64) * KCAL_PER_MOL_TO_EV
    angle_t = np.asarray(raw["At"], dtype=np.float64)
    angle_idx, angle_k, (angle_t,) = _filter_bonded_mm_only(
        angle_idx, angle_k, [angle_t], ml_mask
    )

    dih_idx = np.stack([raw["Da"], raw["Db"], raw["Dc"], raw["Dd"]], axis=1).astype(
        np.int64
    )
    dih_k = np.asarray(raw["Dk"], dtype=np.float64) * KCAL_PER_MOL_TO_EV
    dih_n = np.asarray(raw["Dn"], dtype=np.int64)
    dih_p = np.asarray(raw["Dp"], dtype=np.float64)
    if dih_idx.size:
        dih_idx, dih_k, extras = _filter_bonded_mm_only(
            dih_idx, dih_k, [dih_n.astype(np.float64), dih_p], ml_mask
        )
        dih_n = extras[0].astype(np.int64)
        dih_p = extras[1]
    else:
        dih_idx = np.zeros((0, 4), dtype=np.int64)
        dih_k = np.zeros(0)
        dih_n = np.zeros(0, dtype=np.int64)
        dih_p = np.zeros(0)

    # Append water bonded
    w_bond = np.stack([water["Ba"], water["Bb"]], axis=1)
    bond_idx = np.concatenate([bond_idx, w_bond], axis=0) if bond_idx.size else w_bond
    bond_k = np.concatenate([bond_k, water["Bk"]])
    bond_r0 = np.concatenate([bond_r0, water["Br"]])

    w_ang = np.stack([water["Aa"], water["Ab"], water["Ac"]], axis=1)
    angle_idx = np.concatenate([angle_idx, w_ang], axis=0) if angle_idx.size else w_ang
    angle_k = np.concatenate([angle_k, water["Ak"]])
    angle_t = np.concatenate([angle_t, water["At"]])

    # Exclusions: solute dense matrix + water intramolecular
    excl_i, excl_j = _exclude_pairs_from_matrix(np.asarray(raw["Ex"]))
    excl_i = np.concatenate([excl_i, water["exclude_i"]])
    excl_j = np.concatenate([excl_j, water["exclude_j"]])

    # NB14: keep only MM–MM
    nb_a = np.asarray(raw["Na"], dtype=np.int64)
    nb_b = np.asarray(raw["Nb"], dtype=np.int64)
    nb_lj = np.asarray(raw["Nlf"], dtype=np.float64)
    nb_q = np.asarray(raw["Nqf"], dtype=np.float64)
    if nb_a.size:
        mm = ~ml_mask
        keep = mm[nb_a] & mm[nb_b]
        nb_a, nb_b, nb_lj, nb_q = nb_a[keep], nb_b[keep], nb_lj[keep], nb_q[keep]

    cell_t = None if cell is None else torch.as_tensor(cell, dtype=torch.float64)

    return MLMMTopology(
        ml_mask=torch.as_tensor(ml_mask),
        charges=torch.as_tensor(charges),
        sigma=torch.as_tensor(sigma),
        epsilon=torch.as_tensor(epsilon),
        atomic_numbers=torch.as_tensor(zs),
        masses=torch.as_tensor(masses),
        bond_idx=torch.as_tensor(bond_idx.reshape(-1, 2), dtype=torch.long),
        bond_k=torch.as_tensor(bond_k),
        bond_r0=torch.as_tensor(bond_r0),
        angle_idx=torch.as_tensor(angle_idx.reshape(-1, 3), dtype=torch.long),
        angle_k=torch.as_tensor(angle_k),
        angle_theta0=torch.as_tensor(angle_t),
        dihedral_idx=torch.as_tensor(dih_idx.reshape(-1, 4), dtype=torch.long)
        if dih_idx.size
        else _empty_long(0, 4),
        dihedral_k=torch.as_tensor(dih_k),
        dihedral_n=torch.as_tensor(dih_n, dtype=torch.long),
        dihedral_phi0=torch.as_tensor(dih_p),
        exclude_i=torch.as_tensor(excl_i, dtype=torch.long),
        exclude_j=torch.as_tensor(excl_j, dtype=torch.long),
        nb14_i=torch.as_tensor(nb_a, dtype=torch.long),
        nb14_j=torch.as_tensor(nb_b, dtype=torch.long),
        nb14_lj_factor=torch.as_tensor(nb_lj),
        nb14_q_factor=torch.as_tensor(nb_q),
        cell=cell_t,
    )


def build_tip3p_only_topology(
    n_waters: int,
    *,
    ml_water_indices: list[int] | None = None,
    cell: np.ndarray | torch.Tensor | None = None,
) -> MLMMTopology:
    """Topology for a pure TIP3P box (optionally mark some waters as ML)."""
    water = build_tip3p_water_topology(n_waters, atom_offset=0)
    n_atoms = 3 * int(n_waters)
    ml_mask = np.zeros(n_atoms, dtype=bool)
    if ml_water_indices:
        for w in ml_water_indices:
            base = 3 * int(w)
            ml_mask[base : base + 3] = True
    cell_t = None if cell is None else torch.as_tensor(cell, dtype=torch.float64)
    return MLMMTopology(
        ml_mask=torch.as_tensor(ml_mask),
        charges=torch.as_tensor(water["charges"]),
        sigma=torch.as_tensor(water["sigma"]),
        epsilon=torch.as_tensor(water["epsilon"]),
        atomic_numbers=torch.as_tensor(water["atomic_numbers"]),
        masses=torch.as_tensor(water["masses"]),
        bond_idx=torch.stack(
            [
                torch.as_tensor(water["Ba"], dtype=torch.long),
                torch.as_tensor(water["Bb"], dtype=torch.long),
            ],
            dim=1,
        ),
        bond_k=torch.as_tensor(water["Bk"]),
        bond_r0=torch.as_tensor(water["Br"]),
        angle_idx=torch.stack(
            [
                torch.as_tensor(water["Aa"], dtype=torch.long),
                torch.as_tensor(water["Ab"], dtype=torch.long),
                torch.as_tensor(water["Ac"], dtype=torch.long),
            ],
            dim=1,
        ),
        angle_k=torch.as_tensor(water["Ak"]),
        angle_theta0=torch.as_tensor(water["At"]),
        exclude_i=torch.as_tensor(water["exclude_i"], dtype=torch.long),
        exclude_j=torch.as_tensor(water["exclude_j"], dtype=torch.long),
        cell=cell_t,
    )


# ---------------------------------------------------------------------------
# Geometry helpers (PBC)
# ---------------------------------------------------------------------------


def _delta_mic(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Minimum-image displacement ``r_i - r_j``.

    Fast path for orthorhombic / cubic cells (diagonal); general cell otherwise.
    """
    cell = cell.to(dtype=pos_i.dtype, device=pos_i.device)
    if cell.ndim == 2 and bool(torch.allclose(cell, torch.diag(torch.diag(cell)), atol=1e-8)):
        box = torch.diag(cell)
        d = pos_i - pos_j
        return d - box * torch.round(d / box.clamp_min(1e-12))
    inv = torch.linalg.inv(cell)
    d = pos_i - pos_j
    frac = d @ inv
    frac = frac - torch.round(frac)
    return frac @ cell


def _box_lengths(cell: torch.Tensor) -> torch.Tensor:
    """Return (3,) orthorhombic box lengths from ``[3,3]`` or ``[B,3,3]`` cell."""
    if cell.ndim == 3:
        cell = cell[0]
    return torch.diag(cell.to(dtype=torch.get_default_dtype())).to(dtype=cell.dtype)


def _expand_pair_indices(
    local_i: torch.Tensor,
    local_j: torch.Tensor,
    n_systems: int,
    n_per: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_i.numel() == 0:
        return local_i, local_j
    offsets = torch.arange(n_systems, device=local_i.device, dtype=torch.long) * n_per
    gi = (offsets[:, None] + local_i[None, :]).reshape(-1)
    gj = (offsets[:, None] + local_j[None, :]).reshape(-1)
    return gi, gj


def _build_exclude_matrix(
    n_atoms: int,
    exclude_i: torch.Tensor,
    exclude_j: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    mat = torch.zeros(n_atoms, n_atoms, dtype=torch.bool, device=device)
    if exclude_i.numel() == 0:
        return mat
    mat[exclude_i, exclude_j] = True
    mat[exclude_j, exclude_i] = True
    return mat


# ---------------------------------------------------------------------------
# MM energy terms
# ---------------------------------------------------------------------------


def _mm_bonded_energy_forces(
    positions: torch.Tensor,
    system_idx: torch.Tensor,
    n_systems: int,
    n_per: int,
    topo: MLMMTopology,
    cell: torch.Tensor | None,
    pbc: bool,
    *,
    compute_forces: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """MM bonded energy [n_systems]; optional forces [n_atoms, 3].

    Only terms whose atoms are all in the MM region contribute.
    """
    device = positions.device
    dtype = positions.dtype
    energy = torch.zeros(n_systems, device=device, dtype=dtype)
    forces = (
        torch.zeros_like(positions) if compute_forces else None
    )
    ml = topo.ml_mask

    def _pos_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if pbc and cell is not None:
            c_ref = cell[0] if cell.ndim == 3 else cell
            return _delta_mic(positions[a], positions[b], c_ref)
        return positions[a] - positions[b]

    def _mm_keep(idx: torch.Tensor) -> torch.Tensor:
        """True where every atom of the term is MM."""
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        return ~ml[idx].any(dim=-1)

    # Bonds
    if topo.bond_k.numel() > 0:
        keep = _mm_keep(topo.bond_idx)
        if keep.any():
            bidx = topo.bond_idx[keep]
            bi, bj = _expand_pair_indices(bidx[:, 0], bidx[:, 1], n_systems, n_per)
            k = topo.bond_k[keep].repeat(n_systems)
            r0 = topo.bond_r0[keep].repeat(n_systems)
            dr = _pos_pair(bi, bj)
            r = torch.linalg.norm(dr, dim=-1).clamp_min(1e-12)
            x = r - r0
            e = k * x * x
            energy.scatter_add_(0, system_idx[bi], e)
            if forces is not None:
                fmag = 2.0 * k * x / r
                f = fmag[:, None] * dr
                forces.index_add_(0, bi, -f)
                forces.index_add_(0, bj, f)

    # Angles
    if topo.angle_k.numel() > 0:
        keep = _mm_keep(topo.angle_idx)
        if keep.any():
            aidx = topo.angle_idx[keep]
            ai, aj, ak = aidx[:, 0], aidx[:, 1], aidx[:, 2]
            offs = torch.arange(n_systems, device=device, dtype=torch.long)[:, None] * n_per
            Ai = (offs + ai[None, :]).reshape(-1)
            Aj = (offs + aj[None, :]).reshape(-1)
            Ak = (offs + ak[None, :]).reshape(-1)
            kk = topo.angle_k[keep].repeat(n_systems)
            t0 = topo.angle_theta0[keep].repeat(n_systems)
            v1 = _pos_pair(Ai, Aj)
            v2 = _pos_pair(Ak, Aj)
            n1 = torch.linalg.norm(v1, dim=-1).clamp_min(1e-12)
            n2 = torch.linalg.norm(v2, dim=-1).clamp_min(1e-12)
            cos_t = (v1 * v2).sum(-1) / (n1 * n2)
            cos_t = cos_t.clamp(-0.999999, 0.999999)
            theta = torch.arccos(cos_t)
            dtheta = theta - t0
            e = kk * dtheta * dtheta
            energy.scatter_add_(0, system_idx[Aj], e)
            if forces is not None:
                sin_t = torch.sqrt((1.0 - cos_t * cos_t).clamp_min(1e-12))
                dE_dtheta = 2.0 * kk * dtheta
                dE_dcos = dE_dtheta * (-1.0 / sin_t)
                inv_n1 = 1.0 / n1
                inv_n2 = 1.0 / n2
                dcos_dv1 = v2 * (inv_n1 * inv_n2)[:, None] - cos_t[:, None] * v1 * (
                    inv_n1 * inv_n1
                )[:, None]
                dcos_dv2 = v1 * (inv_n1 * inv_n2)[:, None] - cos_t[:, None] * v2 * (
                    inv_n2 * inv_n2
                )[:, None]
                f1 = dE_dcos[:, None] * dcos_dv1
                f2 = dE_dcos[:, None] * dcos_dv2
                forces.index_add_(0, Ai, -f1)
                forces.index_add_(0, Ak, -f2)
                forces.index_add_(0, Aj, f1 + f2)

    # Dihedrals (same convention as gaff_utils); autograd for forces
    if topo.dihedral_k.numel() > 0:
        keep = _mm_keep(topo.dihedral_idx)
        if keep.any():
            didx = topo.dihedral_idx[keep]
            da, db, dc, dd = didx[:, 0], didx[:, 1], didx[:, 2], didx[:, 3]
            offs = torch.arange(n_systems, device=device, dtype=torch.long)[:, None] * n_per
            Da = (offs + da[None, :]).reshape(-1)
            Db = (offs + db[None, :]).reshape(-1)
            Dc = (offs + dc[None, :]).reshape(-1)
            Dd = (offs + dd[None, :]).reshape(-1)
            kk = topo.dihedral_k[keep].repeat(n_systems)
            nn = topo.dihedral_n[keep].to(dtype=dtype).repeat(n_systems)
            phi0 = topo.dihedral_phi0[keep].repeat(n_systems)

            pos_req = positions.detach().requires_grad_(True)
            c_ref = None
            if pbc and cell is not None:
                c_ref = cell[0] if cell.ndim == 3 else cell
                v1 = _delta_mic(pos_req[Da], pos_req[Db], c_ref)
                v2 = _delta_mic(pos_req[Db], pos_req[Dc], c_ref)
                v3 = _delta_mic(pos_req[Dc], pos_req[Dd], c_ref)
            else:
                v1 = pos_req[Da] - pos_req[Db]
                v2 = pos_req[Db] - pos_req[Dc]
                v3 = pos_req[Dc] - pos_req[Dd]
            eps = 1e-12
            v1n = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + eps)
            v2n = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + eps)
            v3n = v3 / (torch.linalg.norm(v3, dim=-1, keepdim=True) + eps)
            a_norm = torch.linalg.cross(v2n, v1n, dim=-1)
            b_norm = torch.linalg.cross(v3n, v2n, dim=-1)
            cross_ab = torch.linalg.cross(a_norm, b_norm, dim=-1)
            sin_phi = (cross_ab * v2n).sum(-1)
            cos_phi = (a_norm * b_norm).sum(-1)
            active = kk > 1e-9
            sin_phi = torch.where(
                active, sin_phi.clamp(-0.999999, 0.999999), torch.ones_like(sin_phi)
            )
            cos_phi = torch.where(
                active, cos_phi.clamp(-0.999999, 0.999999), torch.zeros_like(cos_phi)
            )
            phi = torch.atan2(-sin_phi, cos_phi)
            e = kk * (1.0 + torch.cos(nn * phi - phi0))
            e_sys = torch.zeros(n_systems, device=device, dtype=dtype)
            e_sys.scatter_add_(0, system_idx[Db], e)
            energy = energy + e_sys
            if forces is not None:
                grad = torch.autograd.grad(e.sum(), pos_req, retain_graph=False)[0]
                forces = forces - grad

    return energy, forces


def _lj_energy_forces(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    system_idx: torch.Tensor,
    n_systems: int,
    n_per: int,
    topo: MLMMTopology,
    cutoff: float,
    *,
    compute_forces: bool,
    exclude_mat: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    lj_atom: torch.Tensor | None = None,
    sqrt_eps: torch.Tensor | None = None,
    ortho_box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Heterogeneous LJ for MM–MM and ML–MM pairs (no ML–ML).

    Optimizations vs naive full-NL pass:
    * keep only pairs where **both** atoms have σ,ε > 0 (drops TIP3P H)
    * half neighbor list (i < j) — no ½ double-counting
    * r² formulation (no ``sqrt`` on the hot path)
    """
    device = positions.device
    dtype = positions.dtype
    ml = topo.ml_mask
    sigma = topo.sigma
    if lj_atom is None:
        lj_atom = (topo.epsilon > 0) & (topo.sigma > 0)
    if sqrt_eps is None:
        sqrt_eps = torch.sqrt(topo.epsilon.clamp_min(0.0))

    if edge_index is None or unit_shifts is None:
        edge_index, _mapping, unit_shifts = torchsim_nl(
            positions,
            cell if cell.ndim == 3 else cell.unsqueeze(0),
            pbc,
            cutoff,
            system_idx,
        )

    i = edge_index[0]
    j = edge_index[1]
    # Local (per-system) indices for topology masks — cheaper than repeating
    # per-atom arrays across all replicas.
    li = torch.remainder(i, n_per)
    lj = torch.remainder(j, n_per)

    # Half list + both atoms LJ-active + not ML–ML
    keep = (i < j) & lj_atom[li] & lj_atom[lj] & ~(ml[li] & ml[lj])
    if exclude_mat is not None:
        keep = keep & ~exclude_mat[li, lj]
    elif topo.exclude_i.numel() > 0:
        a = torch.minimum(li, lj)
        b = torch.maximum(li, lj)
        ex_a = torch.minimum(topo.exclude_i, topo.exclude_j)
        ex_b = torch.maximum(topo.exclude_i, topo.exclude_j)
        excluded = torch.isin(a * n_per + b, ex_a * n_per + ex_b)
        keep = keep & ~excluded

    i = i[keep]
    j = j[keep]
    li = li[keep]
    lj = lj[keep]
    shifts = unit_shifts[keep]
    if i.numel() == 0:
        e0 = torch.zeros(n_systems, device=device, dtype=dtype)
        f0 = torch.zeros_like(positions) if compute_forces else None
        return e0, f0

    cell_row = cell if cell.ndim == 3 else cell.unsqueeze(0)
    sys_i = system_idx[i]
    if ortho_box is not None:
        shift_vec = shifts.to(dtype) * ortho_box.unsqueeze(0)
        disp = positions[i] - positions[j] - shift_vec
    else:
        c0 = cell_row[0]
        if bool(torch.allclose(c0, torch.diag(torch.diag(c0)), atol=1e-8)):
            box = torch.diag(c0).to(dtype=dtype)
            shift_vec = shifts.to(dtype) * box.unsqueeze(0)
            disp = positions[i] - positions[j] - shift_vec
        else:
            shift_vec = torch.einsum("ec,ecd->ed", shifts.to(dtype), cell_row[sys_i])
            disp = positions[i] - positions[j] - shift_vec

    cutoff_sq = float(cutoff) * float(cutoff)
    r2 = (disp * disp).sum(dim=-1).clamp_min(1e-24)
    active = r2 < cutoff_sq

    sig = 0.5 * (sigma[li] + sigma[lj])
    eps_ij = sqrt_eps[li] * sqrt_eps[lj]
    inv_r2 = torch.where(active, 1.0 / r2, torch.zeros_like(r2))
    sig2 = sig * sig
    x2 = sig2 * inv_r2
    x6 = x2 * x2 * x2
    x12 = x6 * x6
    pair_e = torch.where(active, 4.0 * eps_ij * (x12 - x6), torch.zeros_like(r2))

    energy = torch.zeros(n_systems, device=device, dtype=dtype)
    energy.scatter_add_(0, sys_i, pair_e)

    forces = None
    if compute_forces:
        # F_i = 24 ε (2 x12 - x6) / r² * disp
        fscale = torch.where(
            active, 24.0 * eps_ij * (2.0 * x12 - x6) * inv_r2, torch.zeros_like(r2)
        )
        fvec = fscale[:, None] * disp
        forces = torch.zeros_like(positions)
        forces.index_add_(0, i, fvec)
        forces.index_add_(0, j, -fvec)

    if topo.nb14_i.numel() > 0 and topo.nb14_lj_factor.numel() > 0:
        gi, gj = _expand_pair_indices(topo.nb14_i, topo.nb14_j, n_systems, n_per)
        fac = topo.nb14_lj_factor.repeat(n_systems)
        c_ref = cell_row[0] if cell_row.shape[0] >= 1 else cell
        d = _delta_mic(
            positions[gi], positions[gj], c_ref if c_ref.ndim == 2 else c_ref[0]
        )
        r2_14 = (d * d).sum(dim=-1).clamp_min(1e-24)
        inv_r2_14 = 1.0 / r2_14
        li14 = torch.remainder(gi, n_per)
        lj14 = torch.remainder(gj, n_per)
        sig = 0.5 * (sigma[li14] + sigma[lj14])
        eps_ij = sqrt_eps[li14] * sqrt_eps[lj14]
        x2 = (sig * sig) * inv_r2_14
        x6 = x2 * x2 * x2
        x12 = x6 * x6
        pe = 4.0 * eps_ij * (x12 - x6) * fac
        energy.scatter_add_(0, system_idx[gi], pe)
        if compute_forces and forces is not None:
            fscale = 24.0 * eps_ij * (2.0 * x12 - x6) * inv_r2_14 * fac
            fvec = fscale[:, None] * d
            forces.index_add_(0, gi, fvec)
            forces.index_add_(0, gj, -fvec)

    return energy, forces


def _exclusion_coulomb_correction(
    positions: torch.Tensor,
    system_idx: torch.Tensor,
    n_systems: int,
    n_per: int,
    topo: MLMMTopology,
    cell: torch.Tensor,
    pbc: bool,
    *,
    compute_forces: bool,
    excl_i_mm: torch.Tensor | None = None,
    excl_j_mm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Subtract bare Coulomb for excluded pairs that touch the MM region."""
    device = positions.device
    dtype = positions.dtype
    if excl_i_mm is None or excl_j_mm is None:
        if topo.exclude_i.numel() == 0:
            z = torch.zeros(n_systems, device=device, dtype=dtype)
            return z, (torch.zeros_like(positions) if compute_forces else None)
        ml = topo.ml_mask
        keep = ~(ml[topo.exclude_i] & ml[topo.exclude_j])
        if not bool(keep.any()):
            z = torch.zeros(n_systems, device=device, dtype=dtype)
            return z, (torch.zeros_like(positions) if compute_forces else None)
        excl_i_mm = topo.exclude_i[keep]
        excl_j_mm = topo.exclude_j[keep]
    if excl_i_mm.numel() == 0:
        z = torch.zeros(n_systems, device=device, dtype=dtype)
        return z, (torch.zeros_like(positions) if compute_forces else None)

    gi, gj = _expand_pair_indices(excl_i_mm, excl_j_mm, n_systems, n_per)
    q = topo.charges.repeat(n_systems) if n_systems > 1 else topo.charges
    c_ref = cell[0] if cell.ndim == 3 else cell
    if pbc:
        d = _delta_mic(positions[gi], positions[gj], c_ref)
    else:
        d = positions[gi] - positions[gj]
    r = torch.linalg.norm(d, dim=-1).clamp_min(1e-12)
    pair = KE_EV * q[gi] * q[gj] / r
    energy = torch.zeros(n_systems, device=device, dtype=dtype)
    energy.scatter_add_(0, system_idx[gi], -pair)

    forces = None
    if compute_forces:
        rhat = d / r[:, None]
        f_i = -(KE_EV * q[gi] * q[gj] / (r * r))[:, None] * rhat
        forces = torch.zeros_like(positions)
        forces.index_add_(0, gi, f_i)
        forces.index_add_(0, gj, -f_i)
    return energy, forces


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MechanicalEmbeddingMLMM(ModelInterface):
    """Mechanical embedding ML/MM model implementing :class:`ModelInterface`.

    MACE may already be multi-GPU (``ShardedMaceModel``). Independently, set
    ``mm_devices`` to shard classical MM terms (NL / PME|DSF|torchpme / LJ /
    bonded) across GPUs by replica — otherwise those terms run only on
    ``device``. ``electrostatics='torchpme'`` is Warp-free: replica shards run
    on CUDA streams from the main thread (same as nvalchemiops PME).
    """

    def __init__(
        self,
        topology: MLMMTopology,
        mace_model: Any | None = None,
        *,
        cutoff: float = 9.0,
        electrostatics: str = "pme",
        pme_accuracy: float = 1e-4,
        pme_mesh_spacing: float | None = 1.5,
        pme_smearing: float | None = None,
        torchpme_method: str = "p3m",
        nl_backend: str = "pytorch",
        dsf_alpha: float = 0.2,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = False,
        enable_mace: bool = True,
        enable_lj: bool = True,
        enable_bonded: bool = True,
        enable_electrostatics: bool = True,
        enable_pme: bool | None = None,
        enable_ml_coulomb_correction: bool = True,
        mm_devices: Sequence[torch.device | str | int] | None = None,
        _spawn_mm_workers: bool = True,
    ) -> None:
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = bool(compute_forces)
        self._compute_stress = bool(compute_stress)
        self._memory_scales_with = "n_atoms_x_density"

        mode = str(electrostatics).strip().lower()
        if mode not in ("pme", "dsf", "torchpme"):
            raise ValueError(
                f"electrostatics must be 'pme', 'dsf', or 'torchpme', got {electrostatics!r}"
            )
        self.electrostatics = mode
        self.cutoff = float(cutoff)
        self.pme_accuracy = float(pme_accuracy)
        self.pme_mesh_spacing = pme_mesh_spacing
        self.pme_smearing = pme_smearing
        self.torchpme_method = str(torchpme_method).strip().lower()
        self.nl_backend = str(nl_backend).strip().lower()
        self.dsf_alpha = float(dsf_alpha)
        self.enable_mace = bool(enable_mace)
        self.enable_lj = bool(enable_lj)
        self.enable_bonded = bool(enable_bonded)
        # Backward compat: enable_pme overrides enable_electrostatics when given
        if enable_pme is not None:
            enable_electrostatics = bool(enable_pme)
        self.enable_electrostatics = bool(enable_electrostatics)
        self.enable_pme = self.enable_electrostatics  # alias
        self.enable_ml_coulomb_correction = bool(enable_ml_coulomb_correction)

        self.topology = topology.to(self._device, self._dtype)
        self.mace_model = mace_model

        # Hot-path caches
        n = self.topology.n_atoms
        self._exclude_mat = _build_exclude_matrix(
            n,
            self.topology.exclude_i,
            self.topology.exclude_j,
            device=self._device,
        )
        ml = self.topology.ml_mask
        keep_excl = ~(ml[self.topology.exclude_i] & ml[self.topology.exclude_j])
        self._excl_i_mm = self.topology.exclude_i[keep_excl]
        self._excl_j_mm = self.topology.exclude_j[keep_excl]
        # DSF masks only MM-touching exclusions (same pairs PME corrects).
        self._exclude_mat_dsf = _build_exclude_matrix(
            n,
            self._excl_i_mm,
            self._excl_j_mm,
            device=self._device,
        )
        self._ml_idx = self.topology.ml_indices
        self._lj_atom = (self.topology.epsilon > 0) & (self.topology.sigma > 0)
        self._sqrt_eps = torch.sqrt(self.topology.epsilon.clamp_min(0.0))
        # Cache ortho box diagonal when cell is cubic/orthorhombic
        self._ortho_box: torch.Tensor | None = None
        c = self.topology.cell
        if c is not None and c.ndim == 2:
            c = c.to(device=self._device, dtype=self._dtype)
            if bool(torch.allclose(c, torch.diag(torch.diag(c)), atol=1e-8)):
                self._ortho_box = torch.diag(c)
        self._charge_cache: dict[int, torch.Tensor] = {}
        self._ml_charge_cache: dict[int, torch.Tensor] = {}

        self._elec = None
        self._elec_ml = None
        self._torchpme_calc = None
        if self.enable_electrostatics or self.enable_ml_coulomb_correction:
            if self.electrostatics == "torchpme":
                from .torch_pme_model import build_torchpme_calculator

                self._torchpme_calc = build_torchpme_calculator(
                    cutoff=self.cutoff,
                    mesh_spacing=self.pme_mesh_spacing,
                    smearing=self.pme_smearing,
                    method=self.torchpme_method,
                    device=self._device,
                    dtype=self._dtype,
                    full_neighbor_list=False,
                )
            elif self.electrostatics == "pme":
                from torch_sim.models.electrostatics import PMEModel

                pme_kw = dict(
                    cutoff=self.cutoff,
                    accuracy=self.pme_accuracy,
                    device=self._device,
                    dtype=self._dtype,
                    compute_forces=self._compute_forces,
                    compute_stress=False,
                )
                if self.pme_mesh_spacing is not None:
                    pme_kw["mesh_spacing"] = float(self.pme_mesh_spacing)
                self._elec = PMEModel(**pme_kw)
                self._elec_ml = PMEModel(**pme_kw)
            else:
                from torch_sim.models.electrostatics import DSFCoulombModel

                dsf_kw = dict(
                    cutoff=self.cutoff,
                    alpha=self.dsf_alpha,
                    device=self._device,
                    dtype=self._dtype,
                    compute_forces=self._compute_forces,
                    compute_stress=False,
                )
                self._elec = DSFCoulombModel(**dsf_kw)
                self._elec_ml = DSFCoulombModel(**dsf_kw)

        # Legacy aliases used by older call sites / tests
        self._pme = self._elec if self.electrostatics == "pme" else None
        self._pme_ml = self._elec_ml if self.electrostatics == "pme" else None

        if self.electrostatics == "torchpme":
            from .torch_pme_model import warmup_torchpme_device

            warmup_torchpme_device(self._device, self._dtype)

        # Multi-GPU MM workers (NL / PME / LJ / bonded); MACE stays on mace_model.
        self._mm_workers: list[MechanicalEmbeddingMLMM] = []
        self._mm_shard_cache: dict[str, Any] | None = None
        if _spawn_mm_workers and mm_devices is not None:
            from .sharded_mace import parse_cuda_devices

            devs = parse_cuda_devices(mm_devices)
            if len(devs) > 1:
                common = dict(
                    cutoff=self.cutoff,
                    electrostatics=self.electrostatics,
                    pme_accuracy=self.pme_accuracy,
                    pme_mesh_spacing=self.pme_mesh_spacing,
                    pme_smearing=self.pme_smearing,
                    torchpme_method=self.torchpme_method,
                    nl_backend=self.nl_backend,
                    dsf_alpha=self.dsf_alpha,
                    dtype=self._dtype,
                    compute_forces=self._compute_forces,
                    compute_stress=False,
                    enable_mace=False,
                    enable_lj=self.enable_lj,
                    enable_bonded=self.enable_bonded,
                    enable_electrostatics=self.enable_electrostatics,
                    enable_ml_coulomb_correction=self.enable_ml_coulomb_correction,
                    mm_devices=None,
                    _spawn_mm_workers=False,
                )
                for i, d in enumerate(devs):
                    w = MechanicalEmbeddingMLMM(
                        topology, None, device=d, **common
                    )
                    self.add_module(f"mm_worker_{i}", w)
                    self._mm_workers.append(w)
                # Register custom ops on the main thread before any parallel PME.
                if self.electrostatics == "pme":
                    _ensure_electrostatics_ops()
                if self.electrostatics == "torchpme":
                    from .torch_pme_model import warmup_torchpme_device

                    warmup_torchpme_device(self._device, self._dtype)
                    for w in self._mm_workers:
                        warmup_torchpme_device(w.device, self._dtype)

    # --- ModelInterface required ---
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @property
    def memory_scales_with(self):
        return self._memory_scales_with

    def _charges_full(self, n_sys: int) -> torch.Tensor:
        q = self._charge_cache.get(n_sys)
        if q is None or q.device != self._device:
            q = self.topology.charges.repeat(n_sys)
            self._charge_cache[n_sys] = q
        return q

    def _charges_ml(self, n_sys: int) -> torch.Tensor:
        q = self._ml_charge_cache.get(n_sys)
        if q is None or q.device != self._device:
            q = self.topology.charges[self._ml_idx].repeat(n_sys)
            self._ml_charge_cache[n_sys] = q
        return q

    def _mask_excluded_edges(
        self,
        edge_index: torch.Tensor,
        unit_shifts: torch.Tensor,
        mapping: torch.Tensor | None,
        n_per: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Drop MM-touching Amber exclusions from a neighbor list (for DSF)."""
        excl = self._exclude_mat_dsf
        if excl is None or not bool(excl.any()):
            return edge_index, unit_shifts, mapping
        i = edge_index[0]
        j = edge_index[1]
        keep = ~excl[i % n_per, j % n_per]
        ei = edge_index[:, keep]
        sh = unit_shifts[keep]
        mp = mapping[keep] if mapping is not None else None
        return ei, sh, mp

    def _build_neighbor_list(self, state, *, n_systems: int | None = None):
        """Shared NL for electrostatics + LJ (Warp or torchpme/vesin)."""
        if self.electrostatics == "torchpme":
            from .torch_pme_model import warp_free_neighbor_list

            return warp_free_neighbor_list(
                state.positions,
                state.row_vector_cell,
                state.pbc,
                self.cutoff,
                state.system_idx,
                half=True,
                backend=self.nl_backend,
                n_systems=n_systems,
            )
        return torchsim_nl(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            self.cutoff,
            state.system_idx,
        )

    def forward(self, state, **_kwargs: object) -> dict[str, torch.Tensor]:
        n_per = self.topology.n_atoms
        n_atoms = int(state.positions.shape[0])
        n_systems = n_atoms // n_per
        if n_atoms != n_systems * n_per:
            raise ValueError(
                f"state n_atoms={n_atoms} incompatible with topology N={n_per} "
                f"and n_systems={n_systems}"
            )

        device = state.positions.device
        dtype = state.positions.dtype
        energy = torch.zeros(n_systems, device=device, dtype=dtype)
        forces = (
            torch.zeros(n_atoms, 3, device=device, dtype=dtype)
            if self._compute_forces
            else None
        )

        # --- MACE on ML subset (may itself be multi-GPU) ---
        if self.enable_mace and self.mace_model is not None and self._ml_idx.numel() > 0:
            e_m, f_m = self._mace_subset(state)
            energy = energy + e_m
            if forces is not None and f_m is not None:
                forces = forces + f_m

        # --- Classical MM: optionally shard replicas across GPUs ---
        if self._mm_workers:
            e_mm, f_mm = self._mm_terms_sharded(state)
            energy = energy + e_mm
            if forces is not None and f_mm is not None:
                forces = forces + f_mm
            out: dict[str, torch.Tensor] = {"energy": energy.detach()}
            if forces is not None:
                out["forces"] = forces.detach()
            if self._compute_stress:
                out["stress"] = torch.zeros(n_systems, 3, 3, device=device, dtype=dtype)
            return out

        # Shared neighbor list for electrostatics + LJ
        if torch.is_tensor(state.pbc) and state.pbc.numel() > 0:
            # Avoid GPU sync (.item): solvated ML/MM always uses PBC when a cell exists.
            pbc_any = True
        else:
            pbc_any = bool(state.pbc) if not torch.is_tensor(state.pbc) else False
        need_nl = (self.enable_electrostatics or self.enable_lj) and pbc_any
        edge_index = mapping = unit_shifts = None
        if need_nl:
            edge_index, mapping, unit_shifts = self._build_neighbor_list(
                state, n_systems=n_systems
            )

        # --- Electrostatics (PME, DSF, or torchpme) ---
        elec_ready = self._elec is not None or self._torchpme_calc is not None
        if self.enable_electrostatics and elec_ready:
            e_p, f_p = self._electrostatics_all(
                state,
                edge_index=edge_index,
                mapping=mapping,
                unit_shifts=unit_shifts,
            )
            energy = energy + e_p
            if forces is not None and f_p is not None:
                forces = forces + f_p

            # PME / torchpme: reciprocal includes excluded pairs → subtract bare Coulomb.
            # DSF is real-space only: MM-touching exclusions are masked in the NL.
            if self.electrostatics in ("pme", "torchpme"):
                e_x, f_x = _exclusion_coulomb_correction(
                    state.positions,
                    state.system_idx,
                    n_systems,
                    n_per,
                    self.topology,
                    state.row_vector_cell,
                    pbc_any,
                    compute_forces=self._compute_forces,
                    excl_i_mm=self._excl_i_mm,
                    excl_j_mm=self._excl_j_mm,
                )
                energy = energy + e_x
                if forces is not None and f_x is not None:
                    forces = forces + f_x

        # --- ML–ML Coulomb correction ---
        if (
            self.enable_ml_coulomb_correction
            and self._ml_idx.numel() > 0
            and (self._elec_ml is not None or self._torchpme_calc is not None)
        ):
            e_c, f_c = self._ml_ml_coulomb_correction(state)
            energy = energy - e_c
            if forces is not None and f_c is not None:
                forces = forces - f_c

        # --- LJ (reuse NL) ---
        if self.enable_lj:
            e_lj, f_lj = _lj_energy_forces(
                state.positions,
                state.row_vector_cell,
                state.pbc,
                state.system_idx,
                n_systems,
                n_per,
                self.topology,
                self.cutoff,
                compute_forces=self._compute_forces,
                exclude_mat=self._exclude_mat,
                edge_index=edge_index,
                unit_shifts=unit_shifts,
                lj_atom=self._lj_atom,
                sqrt_eps=self._sqrt_eps,
                ortho_box=self._ortho_box,
            )
            energy = energy + e_lj
            if forces is not None and f_lj is not None:
                forces = forces + f_lj

        # --- MM bonded ---
        if self.enable_bonded:
            e_b, f_b = _mm_bonded_energy_forces(
                state.positions,
                state.system_idx,
                n_systems,
                n_per,
                self.topology,
                state.row_vector_cell,
                pbc_any,
                compute_forces=self._compute_forces,
            )
            energy = energy + e_b
            if forces is not None and f_b is not None:
                forces = forces + f_b

        out = {"energy": energy.detach()}
        if forces is not None:
            out["forces"] = forces.detach()
        if self._compute_stress:
            out["stress"] = torch.zeros(n_systems, 3, 3, device=device, dtype=dtype)
        return out

    def _shard_state(self, state, lo: int, hi: int, device: torch.device):
        """Contiguous replica shard on ``device`` (equal atoms per system)."""
        import torch_sim as ts

        n_systems = int(state.n_systems)
        n_atoms = int(state.n_atoms)
        n_per = n_atoms // n_systems
        b = hi - lo
        pos = state.positions.view(n_systems, n_per, 3)[lo:hi].reshape(-1, 3)
        z = state.atomic_numbers.view(n_systems, n_per)[lo:hi].reshape(-1)
        mass = state.masses.view(n_systems, n_per)[lo:hi].reshape(-1)
        cell_src = state.row_vector_cell
        if cell_src.shape[0] == n_systems:
            cell = cell_src[lo:hi]
        else:
            cell = cell_src[0:1].expand(b, 3, 3).contiguous()
        pbc = state.pbc
        if torch.is_tensor(pbc):
            pbc = pbc.to(device=device, non_blocking=True)

        pos = pos.to(device=device, dtype=self._dtype, non_blocking=True)
        mass = mass.to(device=device, dtype=self._dtype, non_blocking=True)
        cell = cell.to(device=device, dtype=self._dtype, non_blocking=True).contiguous()
        z = z.to(device=device, dtype=torch.long, non_blocking=True)
        system_idx = torch.arange(b, device=device, dtype=torch.long).repeat_interleave(
            n_per
        )
        return ts.SimState(
            positions=pos,
            masses=mass,
            cell=cell,
            pbc=pbc,
            atomic_numbers=z,
            system_idx=system_idx,
        )

    def _ensure_mm_shard_cache(self, state) -> dict[str, Any]:
        """Persistent per-GPU SimState, CUDA streams, and gather buffers."""
        from .sharded_mace import shard_system_ranges

        n_systems = int(state.n_systems)
        cache = self._mm_shard_cache
        if cache is not None and int(cache["n_systems"]) == n_systems:
            return cache
        n_per = self.topology.n_atoms
        ranges = shard_system_ranges(n_systems, len(self._mm_workers))
        workers = self._mm_workers[: len(ranges)]
        shards = []
        streams: list[torch.cuda.Stream | None] = []
        events: list[torch.cuda.Event | None] = []
        for (lo, hi), worker in zip(ranges, workers):
            shards.append(self._shard_state(state, lo, hi, worker.device))
            if worker.device.type == "cuda":
                streams.append(torch.cuda.Stream(device=worker.device))
                events.append(torch.cuda.Event())
            else:
                streams.append(None)
                events.append(None)
        cache = {
            "n_systems": n_systems,
            "n_per": n_per,
            "ranges": ranges,
            "workers": workers,
            "shards": shards,
            "streams": streams,
            "events": events,
            "energy": torch.empty(n_systems, device=self._device, dtype=self._dtype),
            "forces": (
                torch.empty(
                    n_systems * n_per, 3, device=self._device, dtype=self._dtype
                )
                if self._compute_forces
                else None
            ),
        }
        self._mm_shard_cache = cache
        return cache

    def _mm_terms_sharded(
        self, state
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run classical MM on replica shards across ``self._mm_workers``.

        Shards, CUDA streams, and gather buffers are cached. After the first
        call only positions (and optionally cell) are copied; masses / atomic
        numbers stay resident. Compute launches from the main thread.
        """
        if self.electrostatics == "pme":
            _ensure_electrostatics_ops()

        _dynamo = getattr(torch, "_dynamo", None)
        _prev_disable = None
        if _dynamo is not None:
            _prev_disable = bool(_dynamo.config.disable)
            _dynamo.config.disable = True

        try:
            cache = self._ensure_mm_shard_cache(state)
            n_per = int(cache["n_per"])
            n_systems = int(cache["n_systems"])
            pos_view = state.positions.view(n_systems, n_per, 3)
            cell_src = state.row_vector_cell
            src_stream = (
                torch.cuda.current_stream(state.positions.device)
                if state.positions.is_cuda
                else None
            )
            pending: list[dict[str, torch.Tensor]] = []
            energy_buf = cache["energy"]
            force_buf = cache["forces"]

            for (lo, hi), worker, shard, stream, event in zip(
                cache["ranges"],
                cache["workers"],
                cache["shards"],
                cache["streams"],
                cache["events"],
                strict=True,
            ):
                src_pos = pos_view[lo:hi].reshape(-1, 3)
                if stream is not None:
                    with torch.cuda.device(worker.device):
                        with torch.cuda.stream(stream):
                            if src_stream is not None:
                                stream.wait_stream(src_stream)
                            shard.positions.copy_(src_pos, non_blocking=True)
                            out = worker(shard)
                            energy_buf[lo:hi].copy_(out["energy"], non_blocking=True)
                            if force_buf is not None:
                                force_buf[lo * n_per : hi * n_per].copy_(
                                    out["forces"], non_blocking=True
                                )
                            event.record(stream)
                            pending.append(out)
                else:
                    shard.positions.copy_(src_pos)
                    out = worker(shard)
                    energy_buf[lo:hi].copy_(out["energy"])
                    if force_buf is not None:
                        force_buf[lo * n_per : hi * n_per].copy_(out["forces"])
                    pending.append(out)

            if self._device.type == "cuda":
                dst = torch.cuda.current_stream(self._device)
                for event in cache["events"]:
                    if event is not None:
                        dst.wait_event(event)
            energy = energy_buf.clone()
            forces = None if force_buf is None else force_buf.clone()
            del pending
            return energy, forces
        finally:
            if _dynamo is not None and _prev_disable is not None:
                _dynamo.config.disable = _prev_disable

    def _gather_mm_shard_outs(
        self, shard_outs: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        energies = [
            o["energy"].to(device=self._device, dtype=self._dtype, non_blocking=True)
            for o in shard_outs
        ]
        energy = torch.cat(energies, dim=0)
        forces = None
        if self._compute_forces:
            forces = torch.cat(
                [
                    o["forces"].to(
                        device=self._device, dtype=self._dtype, non_blocking=True
                    )
                    for o in shard_outs
                ],
                dim=0,
            )
        return energy, forces

    def profile_forward(
        self,
        state,
        *,
        warmup: int = 2,
        repeats: int = 10,
    ) -> dict[str, float]:
        """Time each ML/MM term with CUDA sync. Returns mean ms per call."""
        import time

        n_per = self.topology.n_atoms
        n_systems = int(state.n_systems)
        pbc_any = bool(state.pbc.any().item()) if state.pbc.numel() else False
        use_cuda = state.positions.is_cuda

        def _sync() -> None:
            if use_cuda:
                torch.cuda.synchronize(state.positions.device)

        def _time(fn) -> float:
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            return (time.perf_counter() - t0) * 1e3

        # Warmup full forward (compile / allocate)
        for _ in range(max(0, warmup)):
            _ = self.forward(state)
        _sync()

        # Shared NL once for LJ/PME reuse timing of each piece
        edge_index = mapping = unit_shifts = None
        need_nl = (self.enable_electrostatics or self.enable_lj) and pbc_any

        def _nl():
            nonlocal edge_index, mapping, unit_shifts
            edge_index, mapping, unit_shifts = self._build_neighbor_list(state)

        buckets: dict[str, list[float]] = {
            "mace": [],
            "mm_sharded": [],
            "neighbor_list": [],
            "pme_or_dsf": [],
            "pme_exclusion": [],
            "ml_coulomb_corr": [],
            "lj": [],
            "bonded": [],
            "forward_total": [],
        }

        for _ in range(repeats):
            buckets["forward_total"].append(_time(lambda: self.forward(state)))

            if self.enable_mace and self.mace_model is not None and self._ml_idx.numel() > 0:
                buckets["mace"].append(_time(lambda: self._mace_subset(state)))
            else:
                buckets["mace"].append(0.0)

            if self._mm_workers:
                buckets["mm_sharded"].append(_time(lambda: self._mm_terms_sharded(state)))
                for k in (
                    "neighbor_list",
                    "pme_or_dsf",
                    "pme_exclusion",
                    "ml_coulomb_corr",
                    "lj",
                    "bonded",
                ):
                    buckets[k].append(0.0)
                continue

            buckets["mm_sharded"].append(0.0)

            if need_nl:
                buckets["neighbor_list"].append(_time(_nl))
            else:
                edge_index = mapping = unit_shifts = None
                buckets["neighbor_list"].append(0.0)

            if self.enable_electrostatics and (
                self._elec is not None or self._torchpme_calc is not None
            ):
                buckets["pme_or_dsf"].append(
                    _time(
                        lambda: self._electrostatics_all(
                            state,
                            edge_index=edge_index,
                            mapping=mapping,
                            unit_shifts=unit_shifts,
                        )
                    )
                )
                if self.electrostatics in ("pme", "torchpme"):
                    buckets["pme_exclusion"].append(
                        _time(
                            lambda: _exclusion_coulomb_correction(
                                state.positions,
                                state.system_idx,
                                n_systems,
                                n_per,
                                self.topology,
                                state.row_vector_cell,
                                pbc_any,
                                compute_forces=self._compute_forces,
                                excl_i_mm=self._excl_i_mm,
                                excl_j_mm=self._excl_j_mm,
                            )
                        )
                    )
                else:
                    buckets["pme_exclusion"].append(0.0)
            else:
                buckets["pme_or_dsf"].append(0.0)
                buckets["pme_exclusion"].append(0.0)

            if (
                self.enable_ml_coulomb_correction
                and self._ml_idx.numel() > 0
                and (self._elec_ml is not None or self._torchpme_calc is not None)
            ):
                buckets["ml_coulomb_corr"].append(
                    _time(lambda: self._ml_ml_coulomb_correction(state))
                )
            else:
                buckets["ml_coulomb_corr"].append(0.0)

            if self.enable_lj:
                buckets["lj"].append(
                    _time(
                        lambda: _lj_energy_forces(
                            state.positions,
                            state.row_vector_cell,
                            state.pbc,
                            state.system_idx,
                            n_systems,
                            n_per,
                            self.topology,
                            self.cutoff,
                            compute_forces=self._compute_forces,
                            exclude_mat=self._exclude_mat,
                            edge_index=edge_index,
                            unit_shifts=unit_shifts,
                            lj_atom=self._lj_atom,
                            sqrt_eps=self._sqrt_eps,
                            ortho_box=self._ortho_box,
                        )
                    )
                )
            else:
                buckets["lj"].append(0.0)

            if self.enable_bonded:
                buckets["bonded"].append(
                    _time(
                        lambda: _mm_bonded_energy_forces(
                            state.positions,
                            state.system_idx,
                            n_systems,
                            n_per,
                            self.topology,
                            state.row_vector_cell,
                            pbc_any,
                            compute_forces=self._compute_forces,
                        )
                    )
                )
            else:
                buckets["bonded"].append(0.0)

        return {k: float(sum(v) / max(len(v), 1)) for k, v in buckets.items()}

    def _electrostatics_all(
        self,
        state,
        *,
        edge_index=None,
        mapping=None,
        unit_shifts=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Full-system electrostatics with optional shared neighbor list."""
        if self.electrostatics == "dsf":
            return self._dsf_all(
                state,
                edge_index=edge_index,
                mapping=mapping,
                unit_shifts=unit_shifts,
            )
        if self.electrostatics == "torchpme":
            return self._torchpme_all(
                state,
                edge_index=edge_index,
                mapping=mapping,
                unit_shifts=unit_shifts,
            )
        return self._pme_all(
                state,
                edge_index=edge_index,
                mapping=mapping,
                unit_shifts=unit_shifts,
            )
        return self._pme_all(
            state,
            edge_index=edge_index,
            mapping=mapping,
            unit_shifts=unit_shifts,
        )

    def _dsf_all(
        self,
        state,
        *,
        edge_index=None,
        mapping=None,
        unit_shifts=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from torch_sim import transforms
        from nvalchemiops.torch.interactions.electrostatics import dsf_coulomb

        n_sys = int(state.n_systems)
        n_per = self.topology.n_atoms
        charges = self._charges_full(n_sys)
        if edge_index is None:
            st = self._attach_charges(state, charges)
            out = self._elec(st)
            return out["energy"], out.get("forces")

        # Mask Amber exclusions (DSF has no reciprocal contamination)
        edge_index, unit_shifts, mapping = self._mask_excluded_edges(
            edge_index, unit_shifts, mapping, n_per
        )
        edge_csr, neighbor_ptr, shifts = transforms.build_csr_neighbor_list(
            edge_index,
            mapping,
            unit_shifts,
            state.positions.shape[0],
        )
        cell = state.row_vector_cell.contiguous()
        out = dsf_coulomb(
            positions=state.positions,
            charges=charges,
            cutoff=self.cutoff,
            alpha=self.dsf_alpha,
            neighbor_list=edge_csr,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=shifts,
            cell=cell,
            batch_idx=state.system_idx.to(torch.int32),
            compute_forces=self._compute_forces,
            compute_virial=False,
            num_systems=state.n_systems,
        )
        if not isinstance(out, tuple):
            out = (out,)
        # DSF kernel returns per-system energy already
        energy = (out[0] * KE_EV).to(self._dtype)
        forces = None
        if self._compute_forces:
            forces = (out[1] * KE_EV).to(self._dtype)
        return energy.detach(), (None if forces is None else forces.detach())

    def _pme_all(
        self,
        state,
        *,
        edge_index=None,
        mapping=None,
        unit_shifts=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Full-system PME; reuses a precomputed neighbor list when provided."""
        from torch_sim import transforms
        from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald

        n_sys = int(state.n_systems)
        charges = self._charges_full(n_sys)
        if edge_index is None:
            st = self._attach_charges(state, charges)
            out = self._elec(st)
            return out["energy"], out.get("forces")

        edge_csr, neighbor_ptr, shifts = transforms.build_csr_neighbor_list(
            edge_index,
            mapping,
            unit_shifts,
            state.positions.shape[0],
        )
        cell = state.row_vector_cell.contiguous()
        batch_idx = state.system_idx.to(torch.int32) if state.n_systems > 1 else None
        pme_kwargs: dict = dict(
            positions=state.positions,
            charges=charges,
            cell=cell,
            neighbor_list=edge_csr,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=shifts,
            batch_idx=batch_idx,
            compute_forces=self._compute_forces,
            compute_virial=False,
            accuracy=self.pme_accuracy,
            spline_order=getattr(self._elec, "spline_order", 4),
        )
        if self.pme_mesh_spacing is not None:
            pme_kwargs["mesh_spacing"] = self.pme_mesh_spacing
        _ensure_electrostatics_ops()
        out = particle_mesh_ewald(**pme_kwargs)
        if not isinstance(out, tuple):
            out = (out,)
        per_atom_energy = out[0] * KE_EV
        energy = torch.zeros(
            state.n_systems, dtype=torch.float64, device=state.positions.device
        )
        energy.scatter_add_(0, state.system_idx.long(), per_atom_energy)
        energy = energy.to(self._dtype)
        forces = None
        if self._compute_forces:
            forces = (out[1] * KE_EV).to(self._dtype)
        return energy.detach(), (None if forces is None else forces.detach())

    def _torchpme_all(
        self,
        state,
        *,
        edge_index=None,
        mapping=None,
        unit_shifts=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from .torch_pme_model import torchpme_energy_forces

        if self._torchpme_calc is None:
            raise RuntimeError("torchpme calculator is not initialized")
        n_sys = int(state.positions.shape[0]) // int(self.topology.n_atoms)
        charges = self._charges_full(n_sys)
        return torchpme_energy_forces(
            state.positions,
            charges,
            state.row_vector_cell,
            state.system_idx,
            self._torchpme_calc,
            cutoff=self.cutoff,
            compute_forces=self._compute_forces,
            edge_index=edge_index,
            unit_shifts=unit_shifts,
            mapping=mapping,
            ortho_box=self._ortho_box,
            pbc=state.pbc,
            nl_backend=self.nl_backend,
            full_neighbor_list=False,
            n_systems=n_sys,
        )

    def _attach_charges(self, state, charges: torch.Tensor):
        """Return a shallow-copied state with ``partial_charges`` set."""
        import torch_sim as ts

        new_extras = dict(state.atom_extras)
        new_extras["partial_charges"] = charges
        return ts.SimState(
            positions=state.positions,
            masses=state.masses,
            cell=state.cell if hasattr(state, "cell") else state.row_vector_cell,
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            _atom_extras=new_extras,
            _system_extras=dict(state.system_extras),
        )

    def _ml_ml_coulomb_correction(
        self, state
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Electrostatics on the ML subset only (to be subtracted)."""
        import torch_sim as ts

        n_per = self.topology.n_atoms
        n_sys = int(state.n_systems)
        ml_idx = self._ml_idx
        n_ml = int(ml_idx.numel())
        if n_ml == 0:
            z = torch.zeros(
                n_sys, device=state.positions.device, dtype=state.positions.dtype
            )
            return z, None

        pos = state.positions.view(n_sys, n_per, 3)[:, ml_idx, :].reshape(-1, 3)
        masses = state.masses.view(n_sys, n_per)[:, ml_idx].reshape(-1)
        zs = self.topology.atomic_numbers[ml_idx].repeat(n_sys)
        q = self._charges_ml(n_sys)
        system_idx = torch.arange(n_sys, device=pos.device).repeat_interleave(n_ml)
        cell = state.row_vector_cell
        if cell.shape[0] == 1 and n_sys > 1:
            cell = cell.expand(n_sys, 3, 3).contiguous()

        if self.electrostatics == "torchpme":
            from .torch_pme_model import torchpme_energy_forces

            e, f_ml = torchpme_energy_forces(
                pos,
                q,
                cell,
                system_idx,
                self._torchpme_calc,
                cutoff=self.cutoff,
                compute_forces=self._compute_forces,
                ortho_box=self._ortho_box,
                pbc=state.pbc,
                nl_backend=self.nl_backend,
                full_neighbor_list=False,
            )
            if f_ml is None or not self._compute_forces:
                return e, None
            forces = torch.zeros_like(state.positions)
            forces.view(n_sys, n_per, 3)[:, ml_idx, :] = f_ml.view(n_sys, n_ml, 3)
            return e, forces

        ml_state = ts.SimState(
            positions=pos,
            masses=masses,
            cell=cell,
            pbc=state.pbc if state.pbc.ndim == 1 else state.pbc[0],
            atomic_numbers=zs,
            system_idx=system_idx,
            partial_charges=q,
        )
        out = self._elec_ml(ml_state)
        e = out["energy"]
        f_ml = out.get("forces")
        if f_ml is None or not self._compute_forces:
            return e, None
        forces = torch.zeros_like(state.positions)
        forces.view(n_sys, n_per, 3)[:, ml_idx, :] = f_ml.view(n_sys, n_ml, 3)
        return e, forces

    def _mace_subset(self, state) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch_sim as ts

        n_per = self.topology.n_atoms
        n_sys = int(state.n_systems)
        ml_idx = self._ml_idx
        n_ml = int(ml_idx.numel())
        pos = state.positions.view(n_sys, n_per, 3)[:, ml_idx, :].reshape(-1, 3)
        masses = state.masses.view(n_sys, n_per)[:, ml_idx].reshape(-1)
        zs = self.topology.atomic_numbers[ml_idx].repeat(n_sys)
        system_idx = torch.arange(n_sys, device=pos.device).repeat_interleave(n_ml)
        cell = state.row_vector_cell
        if cell.shape[0] == 1 and n_sys > 1:
            cell = cell.expand(n_sys, 3, 3).contiguous()

        extras = {}
        if state.has_extras("total_charge"):
            extras["total_charge"] = state.total_charge
        if state.has_extras("total_spin"):
            extras["total_spin"] = state.total_spin

        ml_state = ts.SimState(
            positions=pos,
            masses=masses,
            cell=cell,
            pbc=False,
            atomic_numbers=zs,
            system_idx=system_idx,
            **extras,
        )
        out = self.mace_model(ml_state)
        e = out["energy"]
        f_ml = out.get("forces")
        if f_ml is None or not self._compute_forces:
            return e, None
        forces = torch.zeros_like(state.positions)
        forces.view(n_sys, n_per, 3)[:, ml_idx, :] = f_ml.view(n_sys, n_ml, 3)
        return e, forces


def make_water_dimer_state(
    *,
    box: float = 20.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
):
    """Two TIP3P waters in a cubic box (for tests)."""
    import torch_sim as ts

    device = device or torch.device("cpu")
    # geometry: water A at origin, water B translated
    oh = float(TIP3P_PARAMS["OH_r0"])
    ang = math.radians(float(TIP3P_PARAMS["HOH_theta0_deg"]))
    hx = oh * math.sin(ang / 2.0)
    hz = oh * math.cos(ang / 2.0)

    def one_water(origin: np.ndarray) -> np.ndarray:
        return np.array(
            [
                origin,
                origin + np.array([hx, 0.0, hz]),
                origin + np.array([-hx, 0.0, hz]),
            ],
            dtype=np.float64,
        )

    coords = np.concatenate(
        [one_water(np.array([5.0, 5.0, 5.0])), one_water(np.array([8.0, 5.0, 5.0]))],
        axis=0,
    )
    topo = build_tip3p_only_topology(2, cell=np.eye(3) * box)
    topo = topo.to(device, dtype)
    pos = torch.as_tensor(coords, dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device) * box
    state = ts.SimState(
        positions=pos,
        masses=topo.masses,
        cell=cell.unsqueeze(0),
        pbc=True,
        atomic_numbers=topo.atomic_numbers,
        partial_charges=topo.charges,
    )
    return state, topo
