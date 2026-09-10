"""v2.1 fit: single goal-modified salience map (goal-early).

Assembled stimulus evidence per item, from precomputed radial contrast
profiles (build_contexts_v21.py), with the attention window FITTED in
the sensor readout:

  s_i = sum_bins exp(-k*r_b) * relu(g_T*D_T[b] + g_O*D_O[b] + w_p*D_P[b])
  F_i = s_i + g_form*FORM_i + beta_T*hT_i + beta_D*hD_i + g_I*visited_i

The rectification implements the architectural commitment: feature-
level suppression is attenuation/relegation (toward zero), never
negative writing. NOTE: this script is the goal-early step of the
model-comparison ledger (history OUTSIDE the window, exponential
window); the FINAL model - sigmoid window, history inside - lives in
results_history_order.json and is loaded by reproduce_gaspelin.py.

10 weights: g_T, g_O, w_p, g_form, k, beta_T, beta_D, eta_T, eta_D, g_I.
Usage: python fit_v21.py [--epochs 300]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fit_pooled as fp
from model import NLOC, SearchEs2Model


class SearchEs2ModelV21(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_T = nn.Parameter(torch.tensor(1.0))
        self.g_O = nn.Parameter(torch.tensor(0.0))
        self.w_p = nn.Parameter(torch.tensor(0.5))
        self.g_form = nn.Parameter(torch.tensor(1.0))
        self.raw_k = nn.Parameter(torch.tensor(0.7))   # window decay (softplus)
        self.beta_T = nn.Parameter(torch.tensor(0.5))
        self.beta_D = nn.Parameter(torch.tensor(-0.1))
        self.raw_eta_T = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_D = nn.Parameter(torch.tensor(0.0))
        self.g_I = nn.Parameter(torch.tensor(0.0))

    @property
    def k(self):
        return nn.functional.softplus(self.raw_k)

    @property
    def eta_T(self):
        return torch.sigmoid(self.raw_eta_T)

    @property
    def eta_D(self):
        return torch.sigmoid(self.raw_eta_D)

    compute_traces = SearchEs2Model.compute_traces

    def stim_ctx(self, P, radii):
        m = (self.g_T * P[..., 0] + self.g_O * P[..., 1]
             + self.w_p * P[..., 2])
        w_near = torch.exp(-self.k * radii)   # FITTED attention window
        return (torch.relu(m) * w_near).sum(-1)

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            g_T=self.g_T.item(), g_O=self.g_O.item(), w_p=self.w_p.item(),
            g_form=self.g_form.item(), k=self.k.item(),
            beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
            eta_T=self.eta_T.item(), eta_D=self.eta_D.item(),
            g_I=self.g_I.item()).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", default="results_fit_v21.json")
    args = ap.parse_args()

    sacc = pd.read_csv("dataset/saccades_ctx.csv", low_memory=False)
    ev = pd.read_csv("dataset/events.csv", low_memory=False)
    ctx = np.load("dataset/contexts_v21.npz")
    sacc, tt = fp.build_tensors(sacc, ev)
    P = torch.tensor(ctx["P"])                     # [nctx, 6, NBINS, 3]
    FORM = torch.tensor(ctx["FORM"])
    cid = torch.tensor(sacc.ctx.values.astype(int))
    nbins = P.shape[2]
    radii = torch.linspace(0.09, 1.1, nbins)
    S = tt["eT"].shape[0]
    print(f"{len(sacc)} saccades, {S} subjects, {P.shape[0]} contexts")

    rng = np.random.default_rng(args.split_seed)
    test_subj = torch.zeros(S, dtype=torch.bool)
    test_subj[rng.choice(S, int(round(S * args.test_frac)), replace=False)] = True
    test = test_subj[tt["si"]]
    train = ~test

    def nll(m, mask, use_traces=True):
        if use_traces:
            oT, oD = m.compute_traces(tt["eT"], tt["eD"], None)
            hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        else:
            hT = hD = torch.zeros(len(sacc), NLOC)
        stim = m.stim_ctx(P, radii) + m.g_form * FORM   # per context
        F_ctx = stim                     # window now lives in the sensor
        F = F_ctx[cid] + m.beta_T * hT + m.beta_D * hD \
            + m.g_I * tt["visited"].float()
        F = F.masked_fill(~tt["valid"], -1e9)
        lp = torch.log_softmax(F, 1).gather(1, tt["choice"][:, None]).squeeze(1)
        return -lp[mask].mean()

    results = {}
    for name, use_traces, frozen in [
            ("v21_null_no_traces", False,
             ["beta_T", "beta_D", "raw_eta_T", "raw_eta_D", "g_I"]),
            ("v21_full", True, [])]:
        m = SearchEs2ModelV21()
        with torch.no_grad():
            for nm in frozen:
                getattr(m, nm).zero_()
        for nm in frozen:
            getattr(m, nm).requires_grad_(False)
        opt = torch.optim.Adam([p for p in m.parameters()
                                if p.requires_grad], lr=0.05)
        print(f"== {name} ==")
        for e in range(args.epochs):
            opt.zero_grad()
            loss = nll(m, train, use_traces)
            loss.backward()
            opt.step()
            if e % 100 == 0 or e == args.epochs - 1:
                print(f"  epoch {e}: {loss.item():.4f} {m.named_values()}",
                      flush=True)
        results[name] = dict(
            params=m.named_values(),
            train_nll_per_saccade=nll(m, train, use_traces).item(),
            test_nll_per_saccade=nll(m, test, use_traces).item())

    v2 = json.load(open("results_fit_v2.json"))
    results["reference_test_nll"] = dict(
        v2_full_goal_late=v2["v2_full"]["test_nll_per_saccade"],
        v1_flags_traces_ior=v2["v1_reference"]["traces_ior"])
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
