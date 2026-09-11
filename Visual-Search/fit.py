"""Fit the final search model.

  python fit.py        # writes weights_final.json + results_final.json

Pooled MLE on first saccades, subject-level 80/20 split (held-out
people), 300 epochs Adam. RESULTS.md is the running record (the
alternative models priced there were fitted with earlier revisions
of this script; the final model is the only one kept in code).
"""

import argparse
import json

import numpy as np
import torch

import data
from front_end import shape_for  # noqa: F401  (kept: pipeline import check)
from model import SearchModel, color_angles

RADII = torch.linspace(0.09, 1.1, 24)


def main():
    ap = argparse.ArgumentParser()
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

    m = SearchModel()
    m.raw_sigma.requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=0.05)

    def nll(mask):
        oT, oD = m.compute_traces(tt["eT"], tt["eD"], None)
        hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        F = m.field(P, FORM, tt["d"], hT, hD, RADII, cphi, sphi)
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
    json.dump(out, open("results_final.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    json.dump(m.named_values(), open("weights_final.json", "w"), indent=1)
    print("saved weights_final.json")


if __name__ == "__main__":
    main()
