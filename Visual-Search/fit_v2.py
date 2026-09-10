"""v2 pooled fit: the model consumes pixel-derived evidence.

Field per item i (channel evidence E from build_contexts.py, sensor
range R feeding the envelope, traces and IoR exactly as v1):

  F_i = env(R_i) * (g_form*FORM_i + g_simT*SIM_T_i + g_simS*SIM_S_i
                    + w_sal*SAL_i)
        + beta_T*hT_i + beta_D*hD_i + g_I*visited_i

10 weights. The scientific upgrades over v1's role flags: g_simS
(rejection template on the distractor color) and w_sal (gain on raw
salience - singleton-detection mode if positive, salience suppression
if negative) are now DISTINCT weights, separable via cross-study color
variation and the Stilwell 2023 salience manipulation.

Usage: python fit_v2.py [--epochs 300]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fit_pooled as fp
from model import NLOC


class SearchEs2ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_k = nn.Parameter(torch.tensor(1.0))
        self.g_form = nn.Parameter(torch.tensor(1.0))
        self.g_simT = nn.Parameter(torch.tensor(0.5))
        self.g_simS = nn.Parameter(torch.tensor(0.0))
        self.w_sal = nn.Parameter(torch.tensor(0.0))
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

    compute_traces = __import__("model").SearchEs2Model.compute_traces

    def field(self, EV, R, hT, hD, visited):
        stim = (self.g_simT * EV[..., 0] + self.g_simS * EV[..., 1]
                + self.w_sal * EV[..., 2] + self.g_form * EV[..., 3])
        env = torch.sigmoid(self.k * (1.0 - R))
        return (env * stim + self.beta_T * hT + self.beta_D * hD
                + self.g_I * visited.float())

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            k=self.k.item(), g_form=self.g_form.item(),
            g_simT=self.g_simT.item(), g_simS=self.g_simS.item(),
            w_sal=self.w_sal.item(), beta_T=self.beta_T.item(),
            beta_D=self.beta_D.item(), eta_T=self.eta_T.item(),
            eta_D=self.eta_D.item(), g_I=self.g_I.item()).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", default="results_fit_v2.json")
    args = ap.parse_args()

    sacc = pd.read_csv("dataset/saccades_ctx.csv", low_memory=False)
    ev = pd.read_csv("dataset/events.csv", low_memory=False)
    ctx = np.load("dataset/contexts.npz")
    sacc, tt = fp.build_tensors(sacc, ev)
    EV = torch.tensor(ctx["E"][sacc.ctx.values.astype(int)])
    R = torch.tensor(ctx["R"][sacc.ctx.values.astype(int)])
    S = tt["eT"].shape[0]
    print(f"{len(sacc)} saccades, {S} subjects")

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
        F = m.field(EV, R, hT, hD, tt["visited"])
        F = F.masked_fill(~tt["valid"], -1e9)
        lp = torch.log_softmax(F, 1).gather(1, tt["choice"][:, None]).squeeze(1)
        return -lp[mask].mean()

    results = {}
    for name, use_traces, frozen in [
            ("v2_null_no_traces", False,
             ["beta_T", "beta_D", "raw_eta_T", "raw_eta_D", "g_I"]),
            ("v2_full", True, []),
            ("v2_color_rejection_only", True, ["w_sal"]),
            ("v2_salience_only", True, ["g_simS"])]:
        m = SearchEs2ModelV2()
        with torch.no_grad():
            for n in frozen:
                getattr(m, n).zero_()
        for n in frozen:
            getattr(m, n).requires_grad_(False)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                               lr=0.05)
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

    v1 = json.load(open("results_fit.json"))["models"]
    results["v1_reference"] = {k: v1[k]["test_nll_per_saccade"]
                               for k in ("null_no_traces", "traces", "traces_ior")}
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
