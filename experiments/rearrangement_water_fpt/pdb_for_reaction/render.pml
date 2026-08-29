# PyMOL: load the three PDBs and show explicit-water Claisen
reinitialize
bg_color white
set ray_opaque_background, off
set orthoscopic, on
set stick_radius, 0.14
set sphere_scale, 0.22
set cartoon_fancy_helices, 0
load substrate.pdb, substrate
hide everything, substrate
show sticks, substrate and resn LIG
show spheres, substrate and resn LIG and elem C+O
show lines, substrate and resn WAT
color gray20, substrate and resn LIG and elem C
color tv_red, substrate and resn LIG and elem O
color white, substrate and resn LIG and elem H
color palecyan, substrate and resn WAT
set stick_transparency, 0.0, substrate
set line_width, 1.0, substrate
translate [-28.0, 0, 0], substrate
# replica 289  d1=1.443  d2=3.567
load ts.pdb, ts
hide everything, ts
show sticks, ts and resn LIG
show spheres, ts and resn LIG and elem C+O
show lines, ts and resn WAT
color gray20, ts and resn LIG and elem C
color tv_red, ts and resn LIG and elem O
color white, ts and resn LIG and elem H
color palecyan, ts and resn WAT
set stick_transparency, 0.0, ts
set line_width, 1.0, ts
translate [0.0, 0, 0], ts
# replica 172  d1=2.285  d2=2.364
load product.pdb, product
hide everything, product
show sticks, product and resn LIG
show spheres, product and resn LIG and elem C+O
show lines, product and resn WAT
color gray20, product and resn LIG and elem C
color tv_red, product and resn LIG and elem O
color white, product and resn LIG and elem H
color palecyan, product and resn WAT
set stick_transparency, 0.0, product
set line_width, 1.0, product
translate [28.0, 0, 0], product
# replica 138  d1=3.541  d2=1.519
zoom vis, 4
set ray_shadows, 0
png schematic_pymol.png, dpi=400, ray=1, width=2400, height=900
