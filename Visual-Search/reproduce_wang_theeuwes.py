"""Reproduce the Wang & Theeuwes (2018) distractor-location
probability-cueing effects from the final fitted model - no refitting,
no new parameters.

Design: setsize 6, green items / red singleton, first saccades from
center; singleton present on 70% of trials, at one high-probability
(HP) location on 65% of present trials; target uniform. 400 biased
trials per run, statistics over the last 300 (learned regime).

W&T's oculomotor signatures (Wang, Samara & Theeuwes 2019):
 1. Fewer first saccades to the singleton at the HP vs LP locations.
 2. Fewer first saccades to the target when it appears at the HP
    location (impaired target selection there).
 3. A spatial GRADIENT: suppression spills onto locations near the HP
    location. Our traces are slot-indexed with no spread, so the model
    commits to NO gradient - a named, falsifiable divergence.

Stimulus evidence comes from the precomputed v2.1 pixel contexts
(fixloc = 0, green/red); history via the fitted traces, field =
window * [stim + traces] (history-inside final form).

Usage: python reproduce_wang_theeuwes.py
"""

import json

import numpy as np
import pandas as pd
import torch

from fit_v21 import SearchEs2ModelV21
from reproduce_gaspelin import load_final

S = 6
HP = 0


def build_stim_lookup(model, r0):
    """stim[targLoc-1, singLoc(0=absent)] -> per-item windowed stimulus
    drive from the pixel contexts (fixloc=0, green/red, setsize 6)."""
    sacc = pd.read_csv("dataset/saccades_ctx.csv", low_memory=False)
    ctx = np.load("dataset/contexts_v21.npz")
    P = torch.tensor(ctx["P"])
    FORM = torch.tensor(ctx["FORM"])
    radii = torch.linspace(0.09, 1.1, P.shape[2])
    sub = sacc[(sacc.setsize == 6) & (sacc.fixloc == 0)
               & (sacc.targCol == "green")
               & (sacc.singCol.isin(["red", "none"]))]
    key = sub[["targLoc", "singLoc", "ctx"]].drop_duplicates()
    lut = {}
    with torch.no_grad():
        w = torch.sigmoid(model.k * (r0 - radii))
        for _, r in key.iterrows():
            ci = int(r.ctx)
            m = (model.g_T * P[ci, :, :, 0] + model.g_O * P[ci, :, :, 1]
                 + model.w_p * P[ci, :, :, 2])
            stim = (torch.relu(m) * w).sum(-1) + model.g_form * FORM[ci]
            lut[(int(r.targLoc), int(r.singLoc))] = stim
    return lut


def main():
    model, r0 = load_final()
    lut = build_stim_lookup(model, r0)
    print(f"{len(lut)} (targLoc, singLoc) stimulus contexts available")
    etaT = model.eta_T.item(); etaD = model.eta_D.item()
    bT = model.beta_T.item(); bD = model.beta_D.item()
    win_item = torch.sigmoid(model.k * (r0 - 0.5)).item()  # ring radius
    rng = np.random.default_rng(1)

    acc = dict(cap_hp=[0, 0], cap_lp=[0, 0], targ_hp=[0, 0], targ_lp=[0, 0])
    cap_by_dist = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for run in range(400):
        hT = np.zeros(S); hD = np.zeros(S)
        for t in range(400):
            present = rng.random() < 0.70
            if present:
                sing = (HP if rng.random() < 0.65 else
                        int(rng.choice([j for j in range(S) if j != HP]))) + 1
            else:
                sing = 0
            targ = int(rng.choice([j + 1 for j in range(S) if j + 1 != sing]))
            if (targ, sing) in lut and t >= 100:
                stim = lut[(targ, sing)]
                hist = win_item * (bT * torch.tensor(hT)
                                   + bD * torch.tensor(hD))
                F = stim + hist
                F[torch.arange(6) >= S] = -1e9
                p = torch.softmax(F, 0).numpy()
                if present:
                    k = "cap_hp" if sing - 1 == HP else "cap_lp"
                    acc[k][0] += p[sing - 1]; acc[k][1] += 1
                    if sing - 1 != HP:
                        dd = min(abs(sing - 1 - HP), S - abs(sing - 1 - HP))
                        cap_by_dist[dd][0] += p[sing - 1]
                        cap_by_dist[dd][1] += 1
                k = "targ_hp" if targ - 1 == HP else "targ_lp"
                acc[k][0] += p[targ - 1]; acc[k][1] += 1
            hT *= (1 - etaT); hT[targ - 1] += etaT
            eD = np.zeros(S)
            if sing > 0:
                eD[sing - 1] = 1.0
            hD = (1 - etaD) * hD + etaD * eD

    r = {k: 100 * v[0] / max(v[1], 1) for k, v in acc.items()}
    print("\n== W&T signature 1: capture by singleton location ==")
    print(f"first saccades to singleton at HP location: {r['cap_hp']:.2f}%")
    print(f"first saccades to singleton at LP location: {r['cap_lp']:.2f}%")
    print("\n== W&T signature 2: target selection at the suppressed location ==")
    print(f"first saccades to target at HP location:    {r['targ_hp']:.2f}%")
    print(f"first saccades to target elsewhere:         {r['targ_lp']:.2f}%")
    print("\n== W&T signature 3: spatial gradient around HP (LP capture by distance) ==")
    for d in (1, 2, 3):
        v = cap_by_dist[d]
        print(f"LP capture at ring distance {d} from HP: {100*v[0]/max(v[1],1):.2f}%")
    print("(model commits to a FLAT profile - slot traces have no spatial "
          "spread; W&T observed a gradient)")


if __name__ == "__main__":
    main()
