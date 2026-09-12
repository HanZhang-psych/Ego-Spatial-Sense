"""Fit the final search model (point-sensing readout).

  python fit.py        # writes weights_final.json + results_final.json

Pooled MLE on first saccades, subject-level 80/20 split (held-out
people), 600 epochs Adam, on the sensed evidence from
build_contexts.py (dataset/senses.npz). RESULTS.md is the running
record; superseded constructions live in the git history.
"""

import argparse
import json

import numpy as np
import torch

import data
from model import SearchModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()

    sacc, ev = data.load_frames()
    sacc, tt = data.build_tensors(sacc, ev)
    S = np.load("dataset/senses.npz")
    cid = torch.tensor(sacc.ctx.values.astype(int))
    A = torch.tensor(S["A"])[cid]
    BH6, BH4 = torch.tensor(S["BH6"]), torch.tensor(S["BH4"])
    m6 = torch.tensor((sacc.setsize == 6).values)
    train, test = data.subject_split(tt)
    print(f"{len(sacc)} saccades; train/test split by subject")

    m = SearchModel()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=0.05)

    def nll(mask):
        oT, oD = m.compute_traces(tt["eT"], tt["eD"])
        hT, hD = oT[tt["si"], tt["ti"]], oD[tt["si"], tt["ti"]]
        F = m.field(A, hT, hD, BH6, BH4, m6)
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
