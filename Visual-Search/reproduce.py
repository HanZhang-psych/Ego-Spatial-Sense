"""Reproduction batteries for the final model (no refitting).

  python reproduce.py            # both batteries
  python reproduce.py --which gaspelin
  python reproduce.py --which wang

Gaspelin battery (held-out subjects, first saccades - the model's
scope): oculomotor suppression, intertrial location priming.
Wang & Theeuwes: high-probability distractor-location simulation
(capture reduction at the HP location, target cost there, gradient).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

import data
from build_contexts import NBINS, MAXR
from front_end import item_positions
from model import load_final, color_angles

RADII = torch.linspace(0.09, MAXR, NBINS)


def model_probs(m, sacc, tt, P, FORM):
    with torch.no_grad():
        cphi, sphi = color_angles(sacc)
        oT, oD = m.compute_traces(tt["eT"], tt["eD"], None)
        hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        F = m.field(P, FORM, tt["d"], hT, hD, RADII, cphi, sphi)
        F = F.masked_fill(~tt["valid"], -1e9)
        return torch.softmax(F, 1)


def gaspelin(m):
    sacc, ev = data.load_frames()
    sacc, tt = data.build_tensors(sacc, ev)
    ctx = np.load("dataset/contexts_v21.npz")
    cid = torch.tensor(sacc.ctx.values.astype(int))
    P, FORM = torch.tensor(ctx["P"])[cid], torch.tensor(ctx["FORM"])[cid]
    prob = model_probs(m, sacc, tt, P, FORM)
    _, test = data.subject_split(tt)
    held = test.numpy()

    N = len(sacc)
    rows = torch.arange(N)
    isT = torch.zeros(N, 6)
    isT[rows, torch.tensor(sacc.targLoc.values) - 1] = 1
    isS = torch.zeros(N, 6)
    sing = torch.tensor(sacc.singLoc.values)
    sp = sing > 0
    isS[rows[sp], sing[sp] - 1] = 1
    chose = torch.zeros(N, 6)
    chose[rows, tt["choice"]] = 1

    def rates(mask):
        mk = torch.tensor(mask)
        ot = (chose[mk] * isT[mk]).sum() / mk.sum()
        os_ = (chose[mk] * isS[mk]).sum() / mk.sum()
        mt = (prob[mk] * isT[mk]).sum() / mk.sum()
        ms = (prob[mk] * isS[mk]).sum() / mk.sum()
        nns = (tt["valid"][mk].sum(1).float() - 2).clamp(min=1).mean()
        return [100 * float(x) for x in
                (ot, os_, (1 - ot - os_) / nns, mt, ms, (1 - mt - ms) / nns)]

    first = (sacc.saccindex == 1).values
    print("== oculomotor suppression (first saccades, singleton present) ==")
    ot, os_, ons, mt, ms, mns = rates(held & first & sp.numpy())
    print("          target  singleton  NS/item   (paper: 42.0 / 7.9 / 14.2)")
    print(f"observed  {ot:5.1f}   {os_:6.1f}   {ons:6.1f}")
    print(f"model     {mt:5.1f}   {ms:6.1f}   {mns:6.1f}")

    ev2 = ev.copy()
    ev2["block"] = pd.to_numeric(ev2["block"], errors="coerce").fillna(0.0)
    ev2["trial"] = pd.to_numeric(ev2["trial"], errors="coerce")
    ev2["subj"] = ev2["subj"].astype(str)
    ev2 = ev2.sort_values(["study", "subj", "block", "trial"])
    ev2["prevT"] = ev2.groupby(["study", "subj"]).targLoc.shift(1)
    ev2["prevS"] = ev2.groupby(["study", "subj"]).singLoc.shift(1)
    key = ["study", "subj", "block", "trial"]
    sacc2 = sacc.merge(ev2[key + ["prevT", "prevS"]], on=key, how="left")
    prevT, prevS = sacc2.prevT.values, sacc2.prevS.values

    print("\n== intertrial location priming (first saccades) ==")
    tr = held & first & (prevT == sacc.targLoc.values)
    tc = held & first & (prevT != sacc.targLoc.values) & ~np.isnan(prevT)
    for name, mask in [("target-loc REPEAT", tr), ("target-loc change", tc)]:
        ot, _, _, mt, _, _ = rates(mask)
        print(f"{name}: obs {ot:5.1f}%  model {mt:5.1f}%  (paper: 73 vs 37)")
    sr = held & first & sp.numpy() & (prevS == sacc.singLoc.values) & (prevS > 0)
    sc = held & first & sp.numpy() & (prevS != sacc.singLoc.values)
    for name, mask in [("singleton-loc REPEAT", sr), ("singleton-loc change", sc)]:
        _, os_, _, _, ms, _ = rates(mask)
        print(f"{name}: obs {os_:5.1f}%  model {ms:5.1f}%  (paper: 5 vs 10)")


def wang(m, runs=400, trials=400, hp=0):
    sacc, _ = data.load_frames()
    ctx = np.load("dataset/contexts_v21.npz")
    P = torch.tensor(ctx["P"])
    FORM = torch.tensor(ctx["FORM"])
    sub = sacc[(sacc.setsize == 6) & (sacc.fixloc == 0)
               & (sacc.targCol == "green")
               & (sacc.singCol.isin(["red", "none"]))]
    keys = sub[["targLoc", "singLoc", "ctx"]].drop_duplicates()
    from build_contexts import template_axis
    uT, uS = template_axis("green"), template_axis("red")
    cphi = uT[0]*uS[0] + uT[1]*uS[1]
    sphi = -uT[1]*uS[0] + uT[0]*uS[1]
    lut = {}
    with torch.no_grad():
        win = torch.sigmoid(m.k * (m.r0 - RADII))
        wi = torch.sigmoid(m.k * (m.r0 - 0.5)).item()
        for _, r in keys.iterrows():
            ci = int(r.ctx)
            has_sing = int(r.singLoc) > 0
            dproj = ((cphi * P[ci, :, :, 0] + sphi * P[ci, :, :, 1])
                     if has_sing else 0.0)
            mix = m.a * P[ci, :, :, 0] - m.b * dproj
            lut[(int(r.targLoc), int(r.singLoc))] = \
                (torch.relu(mix) * win).sum(-1) + wi * m.g_form * FORM[ci]
    etaT, etaD = m.eta_T.item(), m.eta_D.item()
    bT, bD = m.beta_T.item(), m.beta_D.item()
    rng = np.random.default_rng(1)
    acc = dict(cap_hp=[0, 0], cap_lp=[0, 0], targ_hp=[0, 0], targ_lp=[0, 0])
    dist_acc = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for _ in range(runs):
        hT = np.zeros(6)
        hD = np.zeros(6)
        for t in range(trials):
            present = rng.random() < 0.70
            sing = ((hp if rng.random() < 0.65 else
                     int(rng.choice([j for j in range(6) if j != hp]))) + 1
                    if present else 0)
            targ = int(rng.choice([j + 1 for j in range(6) if j + 1 != sing]))
            if (targ, sing) in lut and t >= 100:
                F = lut[(targ, sing)] + wi * (bT * torch.tensor(hT)
                                              + bD * torch.tensor(hD))
                p = torch.softmax(F, 0).numpy()
                if present:
                    key = "cap_hp" if sing - 1 == hp else "cap_lp"
                    acc[key][0] += p[sing - 1]
                    acc[key][1] += 1
                    if sing - 1 != hp:
                        dd = min(abs(sing - 1 - hp), 6 - abs(sing - 1 - hp))
                        dist_acc[dd][0] += p[sing - 1]
                        dist_acc[dd][1] += 1
                key = "targ_hp" if targ - 1 == hp else "targ_lp"
                acc[key][0] += p[targ - 1]
                acc[key][1] += 1
            hT *= (1 - etaT)
            hT[targ - 1] += etaT
            eD = np.zeros(6)
            if sing > 0:
                eD[sing - 1] = 1.0
            hD = (1 - etaD) * hD + etaD * eD
    r = {k: 100 * v[0] / max(v[1], 1) for k, v in acc.items()}
    print("== Wang & Theeuwes HP-distractor-location simulation ==")
    print(f"capture at HP location: {r['cap_hp']:.2f}%   at LP: {r['cap_lp']:.2f}%")
    print(f"target found at HP: {r['targ_hp']:.2f}%   elsewhere: {r['targ_lp']:.2f}%")
    for dd in (1, 2, 3):
        v = dist_acc[dd]
        print(f"LP capture at ring distance {dd} from HP: "
              f"{100*v[0]/max(v[1],1):.2f}%")
    print("(slot traces predict a FLAT gradient; W&T observed spillover)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both",
                    choices=["both", "gaspelin", "wang"])
    args = ap.parse_args()
    m = load_final()
    if args.which in ("both", "gaspelin"):
        gaspelin(m)
        print()
    if args.which in ("both", "wang"):
        wang(m)


if __name__ == "__main__":
    main()
