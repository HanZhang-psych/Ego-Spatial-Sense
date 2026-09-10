"""Fit the trace spatial-spread kernel sigma_h (v2.1 final form + 1).

The presence-driven trace update is convolved over ring-neighbor
slots with a mass-preserving Gaussian kernel of fitted width sigma_h
(in slot units of ring distance): h <- (1-eta)h + eta*(e @ K(sigma)).
sigma -> 0 recovers the slot-exact final model (the nested null,
held-out 1.22833). Motivation: the Wang & Theeuwes spatial gradient,
which slot traces cannot express, and the agent-side spread-trace
result (same component, same reason).

Usage: python fit_sigma.py [--epochs 300]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fit_pooled as fp
from fit_v21 import SearchEs2ModelV21


class SigmaModel(SearchEs2ModelV21):
    def __init__(self):
        super().__init__()
        self.r0 = nn.Parameter(torch.tensor(0.5))
        self.raw_sigma = nn.Parameter(torch.tensor(-1.5))  # softplus, slots

    @property
    def sigma(self):
        return nn.functional.softplus(self.raw_sigma)

    def kernels(self, dmat):
        K = torch.exp(-dmat ** 2 / (2 * self.sigma ** 2 + 1e-8))
        return K / K.sum(-1, keepdim=True)

    def spread_traces(self, eT, eD, dmat):
        K = self.kernels(dmat)                       # [S, 6, 6]
        eTs = torch.einsum("stj,sjk->stk", eT, K)
        eDs = torch.einsum("stj,sjk->stk", eD, K)
        return self.compute_traces(eTs, eDs, None)


def ring_dmat(setsize):
    d = np.full((6, 6), 99.0)
    for i in range(6):
        d[i, i] = 0.0
    for i in range(setsize):
        for j in range(setsize):
            d[i, j] = min(abs(i - j), setsize - abs(i - j))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="results_fit_sigma.json")
    args = ap.parse_args()

    sacc = pd.read_csv("dataset/saccades_ctx.csv", low_memory=False)
    ev = pd.read_csv("dataset/events.csv", low_memory=False)
    ctx = np.load("dataset/contexts_v21.npz")
    sacc, tt = fp.build_tensors(sacc, ev)
    P = torch.tensor(ctx["P"]); FORM = torch.tensor(ctx["FORM"])
    cid = torch.tensor(sacc.ctx.values.astype(int))
    radii = torch.linspace(0.09, 1.1, P.shape[2])
    N = len(sacc)
    dist = torch.ones(N, 6)
    for j in range(1, 7):
        c = f"d{j}"
        v = sacc[c].notna().values
        dist[v, j - 1] = torch.tensor(sacc.loc[v, c].values, dtype=torch.float32)

    # per-subject setsize -> ring-distance matrix
    S = tt["eT"].shape[0]
    subj_set = sacc.groupby("si").setsize.first()
    dmat = torch.stack([torch.tensor(ring_dmat(int(subj_set.get(s, 6))),
                                     dtype=torch.float32) for s in range(S)])

    rng = np.random.default_rng(0)
    test_subj = torch.zeros(S, dtype=torch.bool)
    test_subj[rng.choice(S, int(round(S * 0.2)), replace=False)] = True
    test = test_subj[tt["si"]]
    train = ~test

    results = {}
    for name, free_sigma in [("sigma_zero_null", False), ("sigma_free", True)]:
        m = SigmaModel()
        if not free_sigma:
            with torch.no_grad():
                m.raw_sigma.fill_(-12.0)      # sigma ~ 0: slot-exact
            m.raw_sigma.requires_grad_(False)

        def nll(mask):
            oT, oD = m.spread_traces(tt["eT"], tt["eD"], dmat)
            hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
            w = torch.sigmoid(m.k * (m.r0 - radii))
            mm = m.g_T * P[..., 0] + m.g_O * P[..., 1] + m.w_p * P[..., 2]
            stim = ((torch.relu(mm) * w).sum(-1) + m.g_form * FORM)[cid]
            win_item = torch.sigmoid(m.k * (m.r0 - dist))
            hist = (m.beta_T * hT + m.beta_D * hD
                    + m.g_I * tt["visited"].float()) * win_item
            F = (stim + hist).masked_fill(~tt["valid"], -1e9)
            lp = torch.log_softmax(F, 1).gather(
                1, tt["choice"][:, None]).squeeze(1)
            return -lp[mask].mean()

        opt = torch.optim.Adam([p for p in m.parameters()
                                if p.requires_grad], lr=0.05)
        print(f"== {name} ==")
        for e in range(args.epochs):
            opt.zero_grad(); l = nll(train); l.backward(); opt.step()
            if e % 100 == 0 or e == args.epochs - 1:
                print(f"  epoch {e}: {l.item():.4f} sigma={m.sigma.item():.3f}"
                      f" r0={m.r0.item():.3f}", flush=True)
        results[name] = dict(params=m.named_values(),
                             sigma=m.sigma.item(), r0=m.r0.item(),
                             train_nll_per_saccade=nll(train).item(),
                             test_nll_per_saccade=nll(test).item())
    print(json.dumps(results, indent=2))
    json.dump(results, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
