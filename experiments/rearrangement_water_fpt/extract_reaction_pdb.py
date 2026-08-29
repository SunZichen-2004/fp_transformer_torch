#!/usr/bin/env python3
"""Pick substrate / TS / product from the last sFPT stage and write PDBs + a schematic.

Last stage of 87173, full box (solute + 334 TIP3P). CVs:
  d1 = O7–C8 (breaking), d2 = C1–C10 (forming).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "rearrangement_water_fpt"
FPT = EXP / "results" / "claisen_water_fpt_mh-0_g8_87173"
BOX_PDB = (
    ROOT
    / "experiments"
    / "rearrangement_water_metadynamics"
    / "mlmm_box"
    / "box.pdb"
)
BOND_FILE = (
    ROOT
    / "experiments"
    / "rearrangement_water_metadynamics"
    / "mlmm_box"
    / "solute_gaff"
    / "solute_bond.txt"
)
OUT = EXP / "pdb_for_reaction"
N_SOLUTE = 20
LBOX = 22.0
# 0-based CV atoms
I_O7, I_C8 = 6, 7
I_C1, I_C10 = 0, 9


def load_solute_bonds(path: Path) -> list[tuple[int, int]]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    bonds = []
    for line in lines[1 : 1 + n]:
        a, b, *_ = line.split()
        bonds.append((int(a), int(b)))
    return bonds


def wrap_to_box_center(xyz: np.ndarray, *, n_solute: int = N_SOLUTE, L: float = LBOX) -> np.ndarray:
    """Make the solute whole, wrap waters as molecules, put solute COM at box centre.

    Same convention as the modelling ``box.pdb`` (solute COM = (L/2, L/2, L/2)).
    """
    xyz = np.asarray(xyz, dtype=np.float64).copy()

    def mic(delta: np.ndarray) -> np.ndarray:
        return delta - L * np.round(delta / L)

    # 1) unsplit solute onto atom 0
    xyz[:n_solute] = xyz[0] + mic(xyz[:n_solute] - xyz[0])
    com = xyz[:n_solute].mean(axis=0)

    # 2) each water (O,H,H) shares the oxygen image relative to solute COM
    for i in range(n_solute, xyz.shape[0], 3):
        shift = mic(xyz[i] - com) - (xyz[i] - com)
        xyz[i : i + 3] += shift

    # 3) translate solute COM to box centre, like box.pdb
    xyz += np.array([0.5 * L, 0.5 * L, 0.5 * L]) - xyz[:n_solute].mean(axis=0)

    # 4) fold waters into [0, L) as whole molecules (solute already interior)
    for i in range(n_solute, xyz.shape[0], 3):
        xyz[i : i + 3] -= L * np.floor(xyz[i] / L)
    return xyz


def load_pdb_atom_lines(path: Path) -> list[str]:
    return [
        line.rstrip("\n")
        for line in path.read_text().splitlines()
        if line.startswith("ATOM") or line.startswith("HETATM")
    ]


def write_pdb(path: Path, xyz: np.ndarray, atom_lines: list[str], *, remark: str) -> None:
    """Keep modelling-PDB columns; only replace Cartesian coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(atom_lines) != xyz.shape[0]:
        raise ValueError(f"template atoms {len(atom_lines)} != coords {xyz.shape[0]}")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"REMARK   {remark}\n")
        fh.write(
            "CRYST1  {0:7.3f}  {0:7.3f}  {0:7.3f}  90.00  90.00  90.00 P 1           1\n".format(
                LBOX
            )
        )
        for line, (x, y, z) in zip(atom_lines, xyz, strict=True):
            fh.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}\n")
        fh.write("END\n")
    print(f"[pdb] {path}", flush=True)


def pick_replicas(cv: np.ndarray) -> dict[str, int]:
    d1, d2 = cv[:, 0], cv[:, 1]

    def _best(mask: np.ndarray, score: np.ndarray) -> int:
        cand = np.where(mask)[0]
        if cand.size == 0:
            return int(np.argmin(score))
        return int(cand[np.argmin(score[cand])])

    # substrate: typical ether C–O, allyl still far (not the d2 wall)
    i_sub = _best(
        (d1 > 1.35) & (d1 < 1.50) & (d2 > 3.2) & (d2 < 5.0),
        (d1 - 1.43) ** 2 + (d2 - 3.6) ** 2,
    )
    # product: typical C–C, ether oxygen detached
    i_pro = _best(
        (d2 > 1.40) & (d2 < 1.55) & (d1 > 3.2) & (d1 < 5.0),
        (d2 - 1.50) ** 2 + (d1 - 3.6) ** 2,
    )
    # TS: concerted, both partial ~2.4 Å
    i_ts = _best(
        (d1 > 2.05) & (d1 < 2.70) & (d2 > 2.05) & (d2 < 2.70),
        (d1 - 2.40) ** 2 + (d2 - 2.40) ** 2,
    )
    return {"substrate": i_sub, "ts": i_ts, "product": i_pro}


def align_to_ref(xyz: np.ndarray, ref: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Kabsch-align ``xyz[idx]`` onto ``ref[idx]``, apply to all atoms."""
    a = xyz[idx] - xyz[idx].mean(axis=0)
    b = ref[idx] - ref[idx].mean(axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    out = (xyz - xyz[idx].mean(axis=0)) @ r + ref[idx].mean(axis=0)
    return out


def draw_schematic(
    frames: dict[str, np.ndarray],
    cv: np.ndarray,
    picks: dict[str, int],
    bonds: list[tuple[int, int]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    color = {"C": "#222222", "O": "#C0392B", "H": "#F5F5F5"}
    r_sol = {"C": 0.38, "O": 0.40, "H": 0.22}
    water_o, water_h = "#6BAED6", "#DEEBF7"

    # align TS / product phenyl (C1–C6) onto substrate
    phenyl = np.arange(6)
    ref = frames["substrate"]
    aligned = {
        "substrate": ref,
        "ts": align_to_ref(frames["ts"], ref, phenyl),
        "product": align_to_ref(frames["product"], ref, phenyl),
    }

    labels = {
        "substrate": "Substrate",
        "ts": "TS",
        "product": "Product",
    }
    subtitles = {
        "substrate": r"allyl phenyl ether  ($d_1$ bonded)",
        "ts": r"concerted Claisen TS",
        "product": r"$o$-allyl phenol  ($d_2$ bonded)",
    }
    order = ["substrate", "ts", "product"]

    fig = plt.figure(figsize=(13.6, 5.15))
    fig.suptitle(
        "Claisen rearrangement in explicit water  (334 TIP3P, 22 Å box)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for k, key in enumerate(order):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        xyz = aligned[key]
        ig = picks[key]
        d1, d2 = float(cv[ig, 0]), float(cv[ig, 1])

        # waters first (behind)
        wo = xyz[N_SOLUTE::3]
        wh1 = xyz[N_SOLUTE + 1 :: 3]
        wh2 = xyz[N_SOLUTE + 2 :: 3]
        ax.scatter(
            wo[:, 0], wo[:, 1], wo[:, 2],
            c=water_o, s=28, alpha=0.38, linewidths=0, depthshade=False, zorder=1,
        )
        ax.scatter(
            np.r_[wh1[:, 0], wh2[:, 0]],
            np.r_[wh1[:, 1], wh2[:, 1]],
            np.r_[wh1[:, 2], wh2[:, 2]],
            c=water_h, s=8, alpha=0.20, linewidths=0, depthshade=False, zorder=1,
        )
        wseg = []
        for o, h1, h2 in zip(wo, wh1, wh2):
            wseg.append([o, h1])
            wseg.append([o, h2])
        lc_w = Line3DCollection(wseg, colors=(0.42, 0.68, 0.84, 0.18), linewidths=0.45)
        ax.add_collection3d(lc_w)

        # solute bonds
        seg = [[xyz[i], xyz[j]] for i, j in bonds]
        lc = Line3DCollection(seg, colors="#111111", linewidths=2.2, zorder=3)
        ax.add_collection3d(lc)

        # reacting distances
        ax.plot(
            [xyz[I_O7, 0], xyz[I_C8, 0]],
            [xyz[I_O7, 1], xyz[I_C8, 1]],
            [xyz[I_O7, 2], xyz[I_C8, 2]],
            color="#E67E22",
            lw=2.6 if d1 < 1.8 else 1.8,
            ls="-" if d1 < 1.8 else "--",
            zorder=4,
        )
        ax.plot(
            [xyz[I_C1, 0], xyz[I_C10, 0]],
            [xyz[I_C1, 1], xyz[I_C10, 1]],
            [xyz[I_C1, 2], xyz[I_C10, 2]],
            color="#1A7A3A",
            lw=2.6 if d2 < 1.8 else 1.8,
            ls="-" if d2 < 1.8 else "--",
            zorder=4,
        )

        sol = xyz[:N_SOLUTE]
        # element from template order: CCCCCCOCCCHHHHHHHHHH
        elems = list("CCCCCCOCCCHHHHHHHHHH")
        for p, e in zip(sol, elems):
            ax.scatter(
                [p[0]], [p[1]], [p[2]],
                c=color[e], s=280 * (r_sol[e] / 0.38) ** 2,
                edgecolors="k", linewidths=0.35,
                depthshade=False, zorder=5,
            )

        ax.set_title(f"{labels[key]}\n{subtitles[key]}", pad=8)
        ax.text2D(
            0.50, 0.02,
            rf"$d_1$={d1:.2f} Å   $d_2$={d2:.2f} Å",
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11,
        )
        # same view: look onto the reacting plane
        ax.view_init(elev=18, azim=-55)
        ax.set_box_aspect((1, 1, 1))
        c0 = sol.mean(axis=0)
        span = 6.2
        ax.set_xlim(c0[0] - span, c0[0] + span)
        ax.set_ylim(c0[1] - span, c0[1] + span)
        ax.set_zlim(c0[2] - span, c0[2] + span)
        ax.set_axis_off()
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        if k < 2:
            fig.text(
                0.335 + 0.31 * k, 0.52, r"$\longrightarrow$",
                ha="center", va="center", fontsize=22, color="0.25",
            )

    fig.text(
        0.02, 0.08,
        "Orange dashed: breaking O7–C8  ·  Green dashed: forming C1–C10  ·  "
        "Light-blue: explicit TIP3P water",
        fontsize=11, color="0.25",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.08, wspace=0.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"[fig] {out_png}", flush=True)


def write_pml(out: Path, picks: dict[str, int], cv: np.ndarray) -> None:
    lines = [
        "# PyMOL: load the three PDBs and show explicit-water Claisen",
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, off",
        "set orthoscopic, on",
        "set stick_radius, 0.14",
        "set sphere_scale, 0.22",
        "set cartoon_fancy_helices, 0",
    ]
    for i, key in enumerate(("substrate", "ts", "product"), start=1):
        ig = picks[key]
        lines += [
            f"load {key}.pdb, {key}",
            f"hide everything, {key}",
            f"show sticks, {key} and resn LIG",
            f"show spheres, {key} and resn LIG and elem C+O",
            f"show lines, {key} and resn WAT",
            f"color gray20, {key} and resn LIG and elem C",
            f"color tv_red, {key} and resn LIG and elem O",
            f"color white, {key} and resn LIG and elem H",
            f"color palecyan, {key} and resn WAT",
            f"set stick_transparency, 0.0, {key}",
            f"set line_width, 1.0, {key}",
            f"translate [{(i-2)*28:.1f}, 0, 0], {key}",
            f"# replica {ig}  d1={cv[ig,0]:.3f}  d2={cv[ig,1]:.3f}",
        ]
    lines += [
        "zoom vis, 4",
        "set ray_shadows, 0",
        "png schematic_pymol.png, dpi=400, ray=1, width=2400, height=900",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"[pml] {out}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv = np.load(FPT / "stage0245_lambda0p025000_cv.npy")
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    xyz_all = np.load(FPT / "stage0245_lambda0p025000_coords_full.npy")
    print(f"[load] cv {cv.shape}  xyz {xyz_all.shape}", flush=True)

    picks = pick_replicas(cv)
    atom_lines = load_pdb_atom_lines(BOX_PDB)
    bonds = load_solute_bonds(BOND_FILE)
    frames = {}
    meta_lines = ["# replica  role  d1(O7-C8)  d2(C1-C10)  from stage0245"]
    for key, ig in picks.items():
        xyz = wrap_to_box_center(xyz_all[ig])
        frames[key] = xyz
        d1, d2 = float(cv[ig, 0]), float(cv[ig, 1])
        remark = (
            f"{key}  replica={ig}  d1(O7-C8)={d1:.4f}  d2(C1-C10)={d2:.4f}  "
            f"stage0245  N={xyz.shape[0]}  n_wat=334  L=22.0"
        )
        write_pdb(OUT / f"{key}.pdb", xyz, atom_lines, remark=remark)
        meta_lines.append(f"{ig:5d}  {key:10s}  {d1:8.4f}  {d2:8.4f}")
        print(f"[pick] {key:10s} replica={ig:4d}  d1={d1:.3f}  d2={d2:.3f}", flush=True)

    (OUT / "selection.txt").write_text("\n".join(meta_lines) + "\n")
    write_pml(OUT / "render.pml", picks, cv)
    draw_schematic(frames, cv, picks, bonds, OUT / "claisen_explicit_water_schematic.png")


if __name__ == "__main__":
    main()
