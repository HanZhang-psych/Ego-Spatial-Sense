"""Stilwell (2023) salience battery: is a more salient singleton
suppressed more?

The model is fitted only on canonical (green/red) displays; here its
weights are applied, unchanged, to Stilwell's displays rendered with
their TRUE colors (blue/pink/teal/red pairings). Observed high/low
salience singleton rates: 7.0 / 11.3 percent; the graded per-pair
structure is the target pattern. Zero history traces (they balance
across conditions). Needs the raw Stilwell2023.txt (not in repo).

  python stilwell_salience.py [path/to/Stilwell2023.txt]
"""
import sys
import numpy as np, pandas as pd, torch
from front_end import render, item_positions, shape_for, IMG
from build_contexts import (opponency_contrast, shape_match_map,
                            template_axis, wedge_profiles, NBINS)
from model import load_final

def build(targCol, singCol, targLoc, singLoc):
    pos, _ = item_positions(6)
    items = [dict(x=pos[j][0], y=pos[j][1],
                  color=(singCol if (j + 1) == singLoc and singCol != "none"
                         else targCol),
                  shape=shape_for(j + 1, targLoc)) for j in range(6)]
    img = render(items)
    maps = opponency_contrast(img)
    fmap = shape_match_map(render(items, scale=2))
    u = template_axis(targCol)
    P, _ = wedge_profiles(maps, u, (0.0, 0.0), 6)
    sc = fmap.shape[0] // IMG
    to_px = lambda v: int(np.clip((v + 0.75) / 1.5 * IMG * sc, 0, IMG*sc - 1))
    F = np.array([fmap[to_px(pos[j][1]), to_px(pos[j][0])] for j in range(6)])
    return P.astype(np.float32), F.astype(np.float32)

# normalization constants from the CANONICAL batch (as in build_contexts)
print("canonical normalization pass...", flush=True)
cP, cF = [], []
for targLoc in range(1, 7):
    for singLoc in range(1, 7):
        if singLoc == targLoc: continue
        P, F = build("green", "red", targLoc, singLoc)
        cP.append(P); cF.append(F)
    P, F = build("green", "none", targLoc, 0)
    cP.append(P); cF.append(F)
cP, cF = np.stack(cP), np.stack(cF)
P_STD = np.abs(cP[..., :2]).std()
F_STD = cF.std()
print(f"P_std {P_STD:.4f}  F_std {F_STD:.4f}", flush=True)

m = load_final()
RADII = torch.linspace(0.09, 1.1, NBINS)

def probs(targCol, singCol, targLoc, singLoc):
    P, F = build(targCol, singCol, targLoc, singLoc)
    P = P.copy(); P[..., :2] /= P_STD; F = F / F_STD
    uT, uS = template_axis(targCol), template_axis(singCol)
    cphi = torch.tensor([uT[0]*uS[0] + uT[1]*uS[1]], dtype=torch.float32)
    sphi = torch.tensor([-uT[1]*uS[0] + uT[0]*uS[1]], dtype=torch.float32)
    z = torch.zeros(1, 6)
    F_ = m.field(torch.tensor(P)[None], torch.tensor(F)[None],
                 torch.full((1, 6), 0.5), z, z, RADII, cphi, sphi)
    return torch.softmax(F_, 1)[0].detach().numpy()

path = (sys.argv[1] if len(sys.argv) > 1 else
        "../search_data/Data Files/Stilwell2023.txt")
df = pd.read_csv(path, sep="\t", low_memory=False)
fs = df[(df.saccindex == 1) & (df.singType == "sing") & (df.currloc >= 1)]
lut = {}
out = []
for (tc, sc_, sal), g in fs.groupby(["targCol", "singCol", "singSal"]):
    ps = pt = 0.0; n = 0
    for (tl, sl), gg in g.groupby(["targLoc", "singLoc"]):
        key = (tc, sc_, int(tl), int(sl))
        if key not in lut:
            lut[key] = probs(tc, sc_, int(tl), int(sl))
        p = lut[key]
        ps += p[int(sl) - 1] * len(gg); pt += p[int(tl) - 1] * len(gg)
        n += len(gg)
    obs_s = (g.currloc == g.singLoc).mean() * 100
    out.append((sal, tc, sc_, n, obs_s, ps / n * 100, pt / n * 100))
    print(f"{sal:4s} targ={tc:5s} sing={sc_:5s} n={n:6d}  "
          f"obs sing {obs_s:5.1f}  model sing {ps/n*100:5.1f}  "
          f"model targ {pt/n*100:5.1f}", flush=True)

o = pd.DataFrame(out, columns=["sal","tc","sc","n","obs_s","mod_s","mod_t"])
for sal, g in o.groupby("sal"):
    w = g.n / g.n.sum()
    print(f"== {sal}: observed singleton {(g.obs_s*w).sum():.1f}%   "
          f"model singleton {(g.mod_s*w).sum():.1f}%   "
          f"(zero traces; plain-item baseline ~{(100-(g.mod_s*w).sum()-(g.mod_t*w).sum())/4:.1f})")
