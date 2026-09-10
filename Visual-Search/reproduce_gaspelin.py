"""Reproduction battery: the final model vs the source paper's key
oculomotor results (Drennan & Gaspelin 2026), on held-out subjects.

1. Oculomotor suppression (first saccades, singleton present):
   % to target / singleton / per-item nonsingleton  (paper: 42.0 /
   7.9 / 14.2; suppression = singleton below baseline).
2. Suppression persists across saccades 1-5 (paper Fig. 6: positive
   suppression effect at every index, shrinking with index).
3. Intertrial location priming (paper Fig. 8): first saccades to the
   target on target-location repeat vs change trials (73.3 vs 36.6);
   to the singleton on singleton-location repeat vs change (4.6 vs
   10.1).

Model probabilities from the fitted v2.1 final form (history inside
the sigmoid window; results_history_order.json), no refitting.
Out of scope by design: manual RTs, latency quantiles, continuous
gaze, practice-block capture.

Usage: python reproduce_gaspelin.py
"""

import json

import numpy as np
import pandas as pd
import torch

import fit_pooled as fp
from fit_v21 import SearchEs2ModelV21


def load_final():
    r = json.load(open("results_history_order.json"))["history_inside_window"]
    p = r["params"]
    m = SearchEs2ModelV21()
    with torch.no_grad():
        m.g_T.copy_(torch.tensor(p["g_T"]))
        m.g_O.copy_(torch.tensor(p["g_O"]))
        m.w_p.copy_(torch.tensor(p["w_p"]))
        m.g_form.copy_(torch.tensor(p["g_form"]))
        m.raw_k.copy_(torch.tensor(np.log(np.expm1(p["k"]))))
        m.beta_T.copy_(torch.tensor(p["beta_T"]))
        m.beta_D.copy_(torch.tensor(p["beta_D"]))
        m.raw_eta_T.copy_(torch.logit(torch.tensor(p["eta_T"])))
        m.raw_eta_D.copy_(torch.logit(torch.tensor(p["eta_D"])))
        m.g_I.copy_(torch.tensor(p["g_I"]))
    return m, r["r0"]


def main():
    sacc = pd.read_csv("dataset/saccades_ctx.csv", low_memory=False)
    ev = pd.read_csv("dataset/events.csv", low_memory=False)
    ctx = np.load("dataset/contexts_v21.npz")
    sacc, tt = fp.build_tensors(sacc, ev)
    P = torch.tensor(ctx["P"])
    FORM = torch.tensor(ctx["FORM"])
    cid = torch.tensor(sacc.ctx.values.astype(int))
    radii = torch.linspace(0.09, 1.1, P.shape[2])
    N = len(sacc)
    dist = torch.ones(N, 6)
    for j in range(1, 7):
        c = f"d{j}"
        v = sacc[c].notna().values
        dist[v, j - 1] = torch.tensor(sacc.loc[v, c].values, dtype=torch.float32)

    m, r0 = load_final()
    with torch.no_grad():
        oT, oD = m.compute_traces(tt["eT"], tt["eD"], None)
        hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        w = torch.sigmoid(m.k * (r0 - radii))
        mm = m.g_T * P[..., 0] + m.g_O * P[..., 1] + m.w_p * P[..., 2]
        stim = ((torch.relu(mm) * w).sum(-1) + m.g_form * FORM)[cid]
        win_item = torch.sigmoid(m.k * (r0 - dist))
        hist = (m.beta_T * hT + m.beta_D * hD
                + m.g_I * tt["visited"].float()) * win_item
        F = (stim + hist).masked_fill(~tt["valid"], -1e9)
        prob = torch.softmax(F, 1)

    rng = np.random.default_rng(0)
    S = tt["eT"].shape[0]
    test_subj = torch.zeros(S, dtype=torch.bool)
    test_subj[rng.choice(S, int(round(S * 0.2)), replace=False)] = True
    held = test_subj[tt["si"]].numpy()

    rows = torch.arange(N)
    isT = torch.zeros(N, 6); isT[rows, torch.tensor(sacc.targLoc.values) - 1] = 1
    sing = torch.tensor(sacc.singLoc.values)
    isS = torch.zeros(N, 6)
    sp = sing > 0
    isS[rows[sp], sing[sp] - 1] = 1
    chose = torch.zeros(N, 6); chose[rows, tt["choice"]] = 1

    def rates(mask):
        mk = torch.tensor(mask)
        ot = (chose[mk] * isT[mk]).sum() / mk.sum()
        os_ = (chose[mk] * isS[mk]).sum() / mk.sum()
        mt = (prob[mk] * isT[mk]).sum() / mk.sum()
        ms = (prob[mk] * isS[mk]).sum() / mk.sum()
        nns = (tt["valid"][mk].sum(1).float() - 2).clamp(min=1)
        ons = ((1 - ot - os_) / nns.mean())
        mns = ((1 - mt - ms) / nns.mean())
        return [100 * x.item() for x in (ot, os_, ons, mt, ms, mns)]

    first = (sacc.saccindex == 1).values
    print("== 1. Oculomotor suppression (first saccades, singleton present, held-out) ==")
    ot, os_, ons, mt, ms, mns = rates(held & first & sp.numpy())
    print(f"          target  singleton  NS/item   (paper obs: 42.0 / 7.9 / 14.2)")
    print(f"observed  {ot:5.1f}   {os_:6.1f}   {ons:6.1f}")
    print(f"model     {mt:5.1f}   {ms:6.1f}   {mns:6.1f}")

    print("\n== 2. Suppression effect by saccade index (singleton present, held-out) ==")
    print("index   observed(sing-NS)   model(sing-NS)   n")
    for k in range(1, 6):
        mk = held & (sacc.saccindex == k).values & sp.numpy()
        if mk.sum() < 200:
            continue
        ot, os_, ons, mt, ms, mns = rates(mk)
        print(f"  {k}      {os_-ons:+7.1f}            {ms-mns:+7.1f}      {int(mk.sum())}")

    ev2 = ev.copy()
    ev2["block"] = pd.to_numeric(ev2["block"], errors="coerce").fillna(0.0)
    ev2["trial"] = pd.to_numeric(ev2["trial"], errors="coerce")
    ev2["subj"] = ev2["subj"].astype(str)
    ev2 = ev2.sort_values(["study", "subj", "block", "trial"])
    ev2["prevT"] = ev2.groupby(["study", "subj"]).targLoc.shift(1)
    ev2["prevS"] = ev2.groupby(["study", "subj"]).singLoc.shift(1)
    key = ["study", "subj", "block", "trial"]
    sacc2 = sacc.merge(ev2[key + ["prevT", "prevS"]], on=key, how="left")
    prevT = sacc2.prevT.values
    prevS = sacc2.prevS.values

    print("\n== 3. Intertrial location priming (first saccades, held-out) ==")
    tr = held & first & (prevT == sacc.targLoc.values)
    tc = held & first & (prevT != sacc.targLoc.values) & ~np.isnan(prevT)
    for name, mask in [("target-loc REPEAT", tr), ("target-loc change", tc)]:
        ot, _, _, mt, _, _ = rates(mask)
        print(f"{name}: obs {ot:5.1f}%  model {mt:5.1f}%   (paper: 73.3 vs 36.6)")
    sr = held & first & sp.numpy() & (prevS == sacc.singLoc.values) & (prevS > 0)
    sc = held & first & sp.numpy() & (prevS != sacc.singLoc.values)
    for name, mask in [("singleton-loc REPEAT", sr), ("singleton-loc change", sc)]:
        _, os_, _, _, ms, _ = rates(mask)
        print(f"{name}: obs {os_:5.1f}%  model {ms:5.1f}%   (paper: 4.6 vs 10.1)")


if __name__ == "__main__":
    main()
