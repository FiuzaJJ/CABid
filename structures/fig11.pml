# Figure 11: TM-align false positives in SCOPe40 (query cyan, target orange)
#   A  d1nkzb_ (f.3.1.1)  vs d1q90m_ (f.23.25.1)   TM-score 0.803
#   B  d1y02a2 (g.50.1.1) vs d2gvia2 (g.39.1.18)   TM-score 0.590
#
# Run from this folder:  pymol -cq fig11.pml

load d1nkzb_.pdb
load d1q90m_.pdb
load d1y02a2.pdb
load d2gvia2.pdb

hide everything
show cartoon
bg_color white
set ray_opaque_background, 1
set cartoon_fancy_helices, 1
set antialias, 2
set ambient, 0.15
set specular, 0.2

color cyan, d1nkzb_ or d1y02a2
color orange, d1q90m_ or d2gvia2

super d1q90m_, d1nkzb_
super d2gvia2, d1y02a2

disable all
enable d1nkzb_
enable d1q90m_
orient d1nkzb_ or d1q90m_
zoom d1nkzb_ or d1q90m_, 3
png ../figures/fig11a_tmalign_fp_membrane.png, width=1400, height=1000, dpi=300, ray=1

disable all
enable d1y02a2
enable d2gvia2
orient d1y02a2 or d2gvia2
zoom d1y02a2 or d2gvia2, 3
png ../figures/fig11b_tmalign_fp_small.png, width=1400, height=1000, dpi=300, ray=1
