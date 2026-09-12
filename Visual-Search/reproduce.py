"""Reproduction battery for the final model (point-sensing; no refitting).

  python reproduce.py

Held-out subjects, first saccades (the model's scope): oculomotor
suppression and intertrial location priming. A former W&T-style
HP-location battery was removed: the additional-singleton paradigm
(shape-singleton target, bottom-up salience capture) is outside what
this feature-search model represents, and transplanting only the
statistical manipulation onto our displays was judged misleading
(see RESULTS, "The Wang & Theeuwes battery is a transplant").
"""

import json

import numpy as np
import pandas as pd
import torch

import data
from model import load_final


def model_probs(m, sacc, tt, A, BH6, BH4):
    with torch.no_grad():
        oT, oD = m.compute_traces(tt["eT"], tt["eD"])
        hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        m6 = torch.tensor((sacc.setsize == 6).values)
        F = m.field(A, hT, hD, BH6, BH4, m6)
        F = F.masked_fill(~tt["valid"], -1e9)
        return torch.softmax(F, 1)


def gaspelin(m):
    sacc, ev = data.load_frames()
    sacc, tt = data.build_tensors(sacc, ev)
    S = np.load("dataset/senses.npz")
    cid = torch.tensor(sacc.ctx.values.astype(int))
    A = torch.tensor(S["A"])[cid]
    prob = model_probs(m, sacc, tt, A,
                       torch.tensor(S["BH6"]), torch.tensor(S["BH4"]))
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


def main():
    gaspelin(load_final())


if __name__ == "__main__":
    main()
