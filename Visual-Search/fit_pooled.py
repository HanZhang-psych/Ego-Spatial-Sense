"""One pooled MLE fit of SearchEs2Model to saccades 1-5 (all studies).

Reads dataset/saccades.csv + dataset/events.csv (from pool_data.py).
Traces are conditioned on ALL trials in order (practice included);
likelihood is scored on the pooled saccade table only. Prints the seven
fitted weights, the null-model comparison, and three diagnostics:
observed vs model first-saccade rates (target / singleton / nonsingleton
baseline), singleton-location-repeat priming, and the no-IoR refixation
check (observed vs predicted rate of returning to already-visited items).

Usage: python fit_pooled.py [--epochs 400] [--out results_fit.json]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from model import NLOC, SearchEs2Model


def build_tensors(sacc, ev):
    # chunked CSV parsing can leave key columns with mixed int/str values,
    # which silently breaks the merge - force numeric
    for df in (sacc, ev):
        df["block"] = pd.to_numeric(df["block"], errors="coerce").fillna(0.0)
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
        df["subj"] = df["subj"].astype(str)
    ev = ev.sort_values(["study", "subj", "block", "trial"]).reset_index(drop=True)
    ev["skey"] = ev.study.astype(str) + "|" + ev.subj.astype(str)
    subjects = {s: i for i, s in enumerate(ev.skey.unique())}
    ev["si"] = ev.skey.map(subjects)
    ev["ti"] = ev.groupby("si").cumcount()
    S, T = len(subjects), int(ev.ti.max()) + 1
    eT = torch.zeros(S, T, NLOC)
    eD = torch.zeros(S, T, NLOC)
    eT[ev.si.values, ev.ti.values, ev.targLoc.values - 1] = 1.0
    has_sing = ev.singLoc.values > 0
    eD[ev.si.values[has_sing], ev.ti.values[has_sing],
       ev.singLoc.values[has_sing] - 1] = 1.0

    key = ["study", "subj", "block", "trial"]
    n_before = len(sacc)
    sacc = sacc.merge(ev[key + ["si", "ti"]], on=key, how="inner")
    if len(sacc) != n_before:
        raise RuntimeError(f"merge lost {n_before - len(sacc)} saccades")

    N = len(sacc)
    d = torch.ones(N, NLOC)
    valid = torch.zeros(N, NLOC, dtype=torch.bool)
    for j in range(1, NLOC + 1):
        col = f"d{j}"
        if col in sacc.columns:
            v = sacc[col].notna().values & (sacc.setsize.values >= j)
            d[v, j - 1] = torch.tensor(sacc.loc[v, col].values, dtype=torch.float32)
            valid[:, j - 1] = torch.tensor(v)
    rows = torch.arange(N)
    fix = torch.tensor(sacc.fixloc.values)
    fixated = fix > 0
    valid[rows[fixated], fix[fixated] - 1] = False
    isT = torch.zeros(N, NLOC)
    isT[rows, torch.tensor(sacc.targLoc.values) - 1] = 1.0
    isS = torch.zeros(N, NLOC)
    sing = torch.tensor(sacc.singLoc.values)
    sp = sing > 0
    isS[rows[sp], sing[sp] - 1] = 1.0
    choice = torch.tensor(sacc.choice.values) - 1

    # visited-items mask for the IoR diagnostic (previous landings this trial)
    visited = torch.zeros(N, NLOC, dtype=torch.bool)
    order = sacc.sort_values(key + ["saccindex"]).index
    prev_key, seen = None, set()
    for idx in order:
        r = sacc.loc[idx]
        tk = (r.study, r.subj, r.block, r.trial)
        if tk != prev_key:
            prev_key, seen = tk, set()
        for j in seen:
            if j != r.fixloc:
                visited[idx, j - 1] = True
        seen.add(int(r.choice))
        if r.fixloc > 0:
            seen.add(int(r.fixloc))
    return sacc, dict(eT=eT, eD=eD, si=torch.tensor(sacc.si.values),
                      ti=torch.tensor(sacc.ti.values), d=d, valid=valid,
                      isT=isT, isS=isS, choice=choice, visited=visited)


def nll(model, tt, use_traces=True):
    if use_traces:
        outT, outD = model.compute_traces(tt["eT"], tt["eD"], None)
        hT = outT[tt["si"], tt["ti"]]
        hD = outD[tt["si"], tt["ti"]]
    else:
        hT = hD = torch.zeros_like(tt["d"])
    lp = model.log_prob(tt["d"], tt["isT"], tt["isS"], hT, hD,
                        tt["valid"], tt["choice"])
    return -lp.mean(), lp


def probs(model, tt):
    outT, outD = model.compute_traces(tt["eT"], tt["eD"], None)
    hT = outT[tt["si"], tt["ti"]]
    hD = outD[tt["si"], tt["ti"]]
    F = model.field(tt["d"], tt["isT"], tt["isS"], hT, hD)
    F = F.masked_fill(~tt["valid"], -1e9)
    return torch.softmax(F, dim=1)


def fit(model, tt, epochs, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss, _ = nll(model, tt)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep}: nll/saccade {loss.item():.4f} "
                  f"{model.named_values()}", flush=True)
    return model


def diagnostics(model, sacc, tt):
    p = probs(model, tt).detach()
    first = torch.tensor((sacc.saccindex == 1).values)
    sp = torch.tensor((sacc.singLoc > 0).values) & first
    obs_t = tt["isT"][sp].gather(1, tt["choice"][sp, None]).mean().item()
    obs_s = tt["isS"][sp].gather(1, tt["choice"][sp, None]).mean().item()
    mod_t = (p[sp] * tt["isT"][sp]).sum(1).mean().item()
    mod_s = (p[sp] * tt["isS"][sp]).sum(1).mean().item()
    n_ns = (tt["valid"][sp].float().sum(1) - 2).clamp(min=1)
    obs_ns = (1 - obs_t - obs_s) / n_ns.mean().item()
    mod_ns = ((1 - (p[sp] * (tt["isT"][sp] + tt["isS"][sp])).sum(1)) / n_ns).mean().item()
    out = {"first_saccade_%": dict(
        target=dict(obs=100 * obs_t, model=100 * mod_t),
        singleton=dict(obs=100 * obs_s, model=100 * mod_s),
        nonsingleton_per_item=dict(obs=100 * obs_ns, model=100 * mod_ns))}

    later = ~first
    if later.any():
        vis = tt["visited"][later]
        obs_refix = tt["visited"][later].gather(
            1, tt["choice"][later, None]).float().mean().item()
        mod_refix = (p[later] * vis.float()).sum(1).mean().item()
        out["refixation_rate_%_no_IoR_model"] = dict(
            obs=100 * obs_refix, model=100 * mod_refix)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saccades", default="dataset/saccades.csv")
    ap.add_argument("--events", default="dataset/events.csv")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default="results_fit.json")
    args = ap.parse_args()

    sacc = pd.read_csv(args.saccades)
    ev = pd.read_csv(args.events)
    sacc, tt = build_tensors(sacc, ev)
    print(f"{len(sacc)} saccades, {tt['eT'].shape[0]} subjects, "
          f"max {tt['eT'].shape[1]} trials")

    model = SearchEs2Model()
    fit(model, tt, args.epochs)
    full_nll, _ = nll(model, tt)

    null = SearchEs2Model()
    with torch.no_grad():
        null.beta_T.zero_()
        null.beta_D.zero_()
    for p in [null.beta_T, null.beta_D, null.raw_eta_T, null.raw_eta_D]:
        p.requires_grad_(False)
    fit(null, tt, args.epochs)
    null_nll, _ = nll(null, tt, use_traces=False)

    res = dict(params=model.named_values(),
               nll_per_saccade=full_nll.item(),
               null_no_traces=dict(params=null.named_values(),
                                   nll_per_saccade=null_nll.item()),
               delta_total_nll=(null_nll.item() - full_nll.item()) * len(sacc),
               n_saccades=len(sacc),
               diagnostics=diagnostics(model, sacc, tt))
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
