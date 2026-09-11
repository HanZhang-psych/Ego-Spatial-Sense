"""Fit the final search model (and its headline ablations).

  python fit.py                    # the final model; writes weights_final.json
  python fit.py --variant no_traces   # memory layer off (traces + IoR)
  python fit.py --variant no_ior      # IoR off
  python fit.py --variant sigma       # trace spatial spread sigma_h free

All variants: pooled MLE on saccades 1-5, subject-level 80/20 split
(held-out people), 300 epochs Adam. Results appended to
results_final.json; RESULTS.md is the running record.
"""

import argparse
import json
import os

import numpy as np
import torch

import data
from front_end import shape_for  # noqa: F401  (kept: pipeline import check)
from model import SearchModel, color_angles

RADII = torch.linspace(0.09, 1.1, 24)


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
    ap.add_argument("--variant", default="final",
                    choices=["final", "no_traces", "no_ior", "sigma",
                             "no_shape", "rectify_first", "gated_shape"])
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    sacc, ev = data.load_frames()
    sacc, tt = data.build_tensors(sacc, ev)
    ctx = np.load("dataset/contexts_v21.npz")
    cid = torch.tensor(sacc.ctx.values.astype(int))
    P = torch.tensor(ctx["P"])[cid]
    FORM = torch.tensor(ctx["FORM"])[cid]
    train, test = data.subject_split(tt)
    cphi, sphi = color_angles(sacc)
    print(f"{len(sacc)} saccades; train/test split by subject")

    dmat = None
    if args.variant == "sigma":
        subj_set = sacc.groupby("si").setsize.first()
        S = tt["eT"].shape[0]
        dmat = torch.stack([torch.tensor(ring_dmat(int(subj_set.get(s, 6))),
                                         dtype=torch.float32)
                            for s in range(S)])

    m = SearchModel()
    frozen = {"no_traces": ["beta_T", "beta_D", "raw_eta_T", "raw_eta_D", "g_I"],
              "no_ior": ["g_I"],
              "no_shape": ["g_form"]}.get(args.variant, [])
    if args.variant != "sigma":
        frozen = frozen + ["raw_sigma"]
    with torch.no_grad():
        for n in frozen:
            if n != "raw_sigma":
                getattr(m, n).zero_()
    for n in frozen:
        getattr(m, n).requires_grad_(False)
    if args.variant == "sigma":
        with torch.no_grad():
            m.raw_sigma.fill_(-1.5)

    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=0.05)

    def nll(mask):
        oT, oD = m.compute_traces(tt["eT"], tt["eD"], dmat)
        hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        F = m.field(P, FORM, tt["d"], tt["visited"], hT, hD, RADII,
                    cphi, sphi,
                    rect="before" if args.variant == "rectify_first"
                    else "after",
                    shape_gated=(args.variant == "gated_shape"))
        F = F.masked_fill(~tt["valid"], -1e9)
        lp = torch.log_softmax(F, 1).gather(1, tt["choice"][:, None]).squeeze(1)
        return -lp[mask].mean()

    for e in range(args.epochs):
        opt.zero_grad()
        loss = nll(train)
        loss.backward()
        opt.step()
        if e % 50 == 0 or e == args.epochs - 1:
            print(f"epoch {e}: {loss.item():.4f} {m.named_values()}", flush=True)

    out = dict(params=m.named_values(),
               train_nll_per_saccade=nll(train).item(),
               test_nll_per_saccade=nll(test).item())
    res = (json.load(open("results_final.json"))
           if os.path.exists("results_final.json") else {})
    res[args.variant] = out
    json.dump(res, open("results_final.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    if args.variant == "final":
        json.dump(m.named_values(), open("weights_final.json", "w"), indent=1)
        print("saved weights_final.json")


if __name__ == "__main__":
    main()
