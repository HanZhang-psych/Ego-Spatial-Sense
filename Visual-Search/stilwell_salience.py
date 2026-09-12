"""Stilwell (2023) salience battery: is a more salient singleton
suppressed more?

The model is fitted only on canonical (green/red) displays, and
Stilwell's LOW-salience trials are excluded from the pooled dataset
entirely (pool_data.py) - so the low-salience condition is fully
out-of-sample: neither its colors nor its choices ever touch the
fit. Weights are applied, unchanged, to Stilwell's displays rendered
with the pipeline's SCHEMATIC colors (a "red vs a less-red" family;
Han's call - the aim is the qualitative gradient, not Stilwell's
exact numbers. The paper's exact CIE colors were probed once and
INVERT the gradient under this crude opponency front end - on
record in RESULTS - so the claim is scoped to schematic colors).
Sensed evidence (point-sensing readout), zero history traces (they
balance across conditions). Needs Stilwell2023.txt (not in repo).

  python stilwell_salience.py [path/to/Stilwell2023.txt]
"""
import sys

import numpy as np
import pandas as pd
import torch

from build_contexts import display_senses
from model import load_final

m = load_final()
S = np.load("dataset/senses.npz")
NORM = S["NORM"]


def probs(targCol, singCol, targLoc, singLoc):
    A = display_senses(6, targLoc, singLoc, targCol, singCol) / NORM
    A = torch.tensor(A)
    with torch.no_grad():
        F = m.g_C * A[:, 0] + m.g_F * A[:, 1]
    return torch.softmax(F[None], 1)[0].numpy()


path = (sys.argv[1] if len(sys.argv) > 1 else
        "../search_data/Data Files/Stilwell2023.txt")
df = pd.read_csv(path, sep="\t", low_memory=False)
fs = df[(df.saccindex == 1) & (df.singType == "sing") & (df.currloc >= 1)]
lut = {}
out = []
for (tc, sc_, sal), g in fs.groupby(["targCol", "singCol", "singSal"]):
    ps = pt = 0.0
    n = 0
    for (tl, sl), gg in g.groupby(["targLoc", "singLoc"]):
        key = (tc, sc_, int(tl), int(sl))
        if key not in lut:
            lut[key] = probs(tc, sc_, int(tl), int(sl))
        p = lut[key]
        ps += p[int(sl) - 1] * len(gg)
        pt += p[int(tl) - 1] * len(gg)
        n += len(gg)
    obs_s = (g.currloc == g.singLoc).mean() * 100
    out.append((sal, tc, sc_, n, obs_s, ps / n * 100, pt / n * 100))
    print(f"{sal:4s} targ={tc:5s} sing={sc_:5s} n={n:6d}  "
          f"obs sing {obs_s:5.1f}  model sing {ps/n*100:5.1f}  "
          f"model targ {pt/n*100:5.1f}", flush=True)

o = pd.DataFrame(out, columns=["sal", "tc", "sc", "n", "obs_s", "mod_s",
                               "mod_t"])
for sal, g in o.groupby("sal"):
    w = g.n / g.n.sum()
    print(f"== {sal}: observed singleton {(g.obs_s*w).sum():.1f}%   "
          f"model singleton {(g.mod_s*w).sum():.1f}%   "
          f"(zero traces; plain-item baseline "
          f"~{(100-(g.mod_s*w).sum()-(g.mod_t*w).sum())/4:.1f})")
