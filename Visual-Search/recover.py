"""Parameter recovery for SearchEs2Model.

Simulates a synthetic participant pool from the fitted weights, then the
synthetic data are refit from scratch with fit_pooled.py; recovery =
the refit landing on the generating values.

Design realism: the REAL trial sequences (events.csv: each subject's
actual target/singleton location order, practice included) drive the
traces, so history statistics are authentic. Items sit on an
iso-eccentric ring (center-to-item = 0.5 in diameter units, matching
the normalized distances of the real data). Within each non-practice
trial the model generates a scanpath exactly as specified: start at
center, sample from the softmax over the choice set (fixated item
excluded, IoR on visited items), stop at the target or after 5
saccades. Traces are presence-driven, so simulated choices never feed
back into them - practice trials need no simulated saccades.

Usage:
  python recover.py --params results_fit.json --out dataset/synthetic_saccades.csv
  python fit_pooled.py --saccades dataset/synthetic_saccades.csv --out results_recovery.json
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch

from model import SearchEs2Model


def ring_coords(setsize):
    return [(0.5 * math.cos(2 * math.pi * j / setsize),
             0.5 * math.sin(2 * math.pi * j / setsize))
            for j in range(setsize)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="results_fit.json")
    ap.add_argument("--variant", default="traces_ior")
    ap.add_argument("--events", default="dataset/events.csv")
    ap.add_argument("--out", default="dataset/synthetic_saccades.csv")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    gen = json.load(open(args.params))["models"][args.variant]["params"]
    print("generating params:", gen)
    model = SearchEs2Model()
    with torch.no_grad():
        model.raw_k.copy_(torch.tensor(np.log(np.expm1(gen["k"]))))
        model.g_T.copy_(torch.tensor(gen["g_T"]))
        model.g_S.copy_(torch.tensor(gen["g_S"]))
        model.beta_T.copy_(torch.tensor(gen["beta_T"]))
        model.beta_D.copy_(torch.tensor(gen["beta_D"]))
        model.raw_eta_T.copy_(torch.logit(torch.tensor(gen["eta_T"])))
        model.raw_eta_D.copy_(torch.logit(torch.tensor(gen["eta_D"])))
        model.g_I.copy_(torch.tensor(gen["g_I"]))

    ev = pd.read_csv(args.events)
    ev["block"] = pd.to_numeric(ev["block"], errors="coerce").fillna(0.0)
    ev["trial"] = pd.to_numeric(ev["trial"], errors="coerce")
    ev = ev.sort_values(["study", "subj", "block", "trial"])
    rng = np.random.default_rng(args.seed)
    etaT, etaD = gen["eta_T"], gen["eta_D"]

    rows = []
    with torch.no_grad():
        for (study, subj), sub in ev.groupby(["study", "subj"], sort=False):
            hT = np.zeros(6)
            hD = np.zeros(6)
            for r in sub.itertuples():
                s = int(r.setsize)
                if r.practice == 0:
                    coords = ring_coords(s)
                    targ, sing = int(r.targLoc), int(r.singLoc)
                    fx, fy, fixloc = 0.0, 0.0, 0
                    visited = np.zeros(6, bool)
                    for k in range(1, 6):
                        d = torch.ones(1, 6)
                        valid = torch.zeros(1, 6, dtype=torch.bool)
                        for j in range(s):
                            d[0, j] = math.hypot(coords[j][0] - fx,
                                                 coords[j][1] - fy)
                            valid[0, j] = (j + 1) != fixloc
                        isT = torch.zeros(1, 6)
                        isT[0, targ - 1] = 1.0
                        isS = torch.zeros(1, 6)
                        if sing > 0:
                            isS[0, sing - 1] = 1.0
                        F = model.field(d, isT, isS,
                                        torch.tensor(hT, dtype=torch.float32)[None],
                                        torch.tensor(hD, dtype=torch.float32)[None],
                                        torch.tensor(visited)[None])
                        F = F.masked_fill(~valid, -1e9)
                        p = torch.softmax(F, 1)[0].numpy()
                        choice = int(rng.choice(6, p=p / p.sum())) + 1
                        rows.append(dict(
                            study=study, subj=subj, block=r.block,
                            trial=r.trial, saccindex=k, setsize=s,
                            choice=choice, fixloc=fixloc, targLoc=targ,
                            singLoc=sing,
                            **{f"d{j+1}": d[0, j].item() if j < s else np.nan
                               for j in range(6)}))
                        if choice == targ:
                            break
                        if fixloc > 0:
                            visited[fixloc - 1] = True
                        fixloc = choice
                        fx, fy = coords[choice - 1]
                hT = (1 - etaT) * hT
                hT[int(r.targLoc) - 1] += etaT
                eD = np.zeros(6)
                if int(r.singLoc) > 0:
                    eD[int(r.singLoc) - 1] = 1.0
                hD = (1 - etaD) * hD + etaD * eD
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} synthetic saccades "
          f"({(out.saccindex == 1).sum()} first) -> {args.out}")
    print(out.saccindex.value_counts().sort_index())


if __name__ == "__main__":
    main()
