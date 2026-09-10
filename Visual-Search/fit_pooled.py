"""Pooled MLE fits of SearchEs2Model to saccades 1-5 (all studies),
with a subject-level train/test split and a nested model comparison:
no-trace null -> traces -> traces + IoR (single visited-item penalty).

Reads dataset/saccades.csv + dataset/events.csv (from pool_data.py).
Traces are conditioned on ALL trials in order (practice included);
likelihood is scored on the pooled saccade table only. Models are fit
on a subject-level train split and evaluated on held-out subjects
(traces are runtime state, so they compute within held-out subjects;
what generalizes or not is the pooled weights). Reports fitted weights,
train/test NLL per model, and diagnostics on the held-out subjects
(first-saccade rates; refixation rate vs the IoR prediction).

Usage: python fit_pooled.py [--epochs 300] [--test_frac 0.2] [--split_seed 0]
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


def nll(model, tt, use_traces=True, use_ior=True, mask=None):
    if use_traces:
        outT, outD = model.compute_traces(tt["eT"], tt["eD"], None)
        hT = outT[tt["si"], tt["ti"]]
        hD = outD[tt["si"], tt["ti"]]
    else:
        hT = hD = torch.zeros_like(tt["d"])
    lp = model.log_prob(tt["d"], tt["isT"], tt["isS"], hT, hD,
                        tt["valid"], tt["choice"],
                        tt["visited"] if use_ior else None)
    if mask is not None:
        lp = lp[mask]
    return -lp.mean(), lp


def probs(model, tt, use_ior=True):
    outT, outD = model.compute_traces(tt["eT"], tt["eD"], None)
    hT = outT[tt["si"], tt["ti"]]
    hD = outD[tt["si"], tt["ti"]]
    F = model.field(tt["d"], tt["isT"], tt["isS"], hT, hD,
                    tt["visited"] if use_ior else None)
    F = F.masked_fill(~tt["valid"], -1e9)
    return torch.softmax(F, dim=1)


def fit(model, tt, epochs, lr=0.05, use_traces=True, use_ior=True, mask=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss, _ = nll(model, tt, use_traces, use_ior, mask)
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"  epoch {ep}: nll/saccade {loss.item():.4f} "
                  f"{model.named_values()}", flush=True)
    return model


def subset(tt, mask):
    out = dict(tt)
    for k in ("si", "ti", "d", "valid", "isT", "isS", "choice", "visited"):
        out[k] = tt[k][mask]
    return out


def diagnostics(model, sacc, tt, use_ior=True):
    p = probs(model, tt, use_ior).detach()
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
        out["refixation_rate_%"] = dict(
            obs=100 * obs_refix, model=100 * mod_refix)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saccades", default="dataset/saccades.csv")
    ap.add_argument("--events", default="dataset/events.csv")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", default="results_fit.json")
    args = ap.parse_args()

    sacc = pd.read_csv(args.saccades)
    ev = pd.read_csv(args.events)
    sacc, tt = build_tensors(sacc, ev)
    S = tt["eT"].shape[0]
    print(f"{len(sacc)} saccades, {S} subjects, max {tt['eT'].shape[1]} trials")

    rng = np.random.default_rng(args.split_seed)
    test_subj = torch.zeros(S, dtype=torch.bool)
    test_subj[rng.choice(S, int(round(S * args.test_frac)), replace=False)] = True
    test = test_subj[tt["si"]]
    train = ~test
    print(f"split: {int(train.sum())} train / {int(test.sum())} test saccades "
          f"({int((~test_subj).sum())}/{int(test_subj.sum())} subjects)")

    def freeze(m, names):
        with torch.no_grad():
            for n in names:
                getattr(m, n).zero_()
        for n in names:
            getattr(m, n).requires_grad_(False)

    variants = {}
    specs = [
        ("null_no_traces", dict(use_traces=False, use_ior=False,
         frozen=["beta_T", "beta_D", "raw_eta_T", "raw_eta_D", "g_I"])),
        ("traces", dict(use_traces=True, use_ior=False, frozen=["g_I"])),
        ("traces_ior", dict(use_traces=True, use_ior=True, frozen=[])),
    ]
    for name, spec in specs:
        print(f"== {name} ==")
        m = SearchEs2Model()
        freeze(m, spec["frozen"])
        fit(m, tt, args.epochs, use_traces=spec["use_traces"],
            use_ior=spec["use_ior"], mask=train)
        tr, _ = nll(m, tt, spec["use_traces"], spec["use_ior"], train)
        te, _ = nll(m, tt, spec["use_traces"], spec["use_ior"], test)
        variants[name] = dict(params=m.named_values(),
                              n_free=7 + 1 - len(spec["frozen"]),
                              train_nll_per_saccade=tr.item(),
                              test_nll_per_saccade=te.item())
        variants[name]["model_obj"] = m
        variants[name]["use_ior"] = spec["use_ior"]

    base = variants["traces"]["test_nll_per_saccade"]
    res = dict(n_saccades=len(sacc),
               split=dict(test_frac=args.test_frac, seed=args.split_seed,
                          n_train=int(train.sum()), n_test=int(test.sum())),
               models={k: {kk: vv for kk, vv in v.items()
                           if kk not in ("model_obj", "use_ior")}
                       for k, v in variants.items()},
               delta_test_nll_total=dict(
                   traces_vs_null=(variants["null_no_traces"]["test_nll_per_saccade"]
                                   - base) * int(test.sum()),
                   ior_vs_traces=(base
                                  - variants["traces_ior"]["test_nll_per_saccade"])
                   * int(test.sum())))
    for name in ("traces", "traces_ior"):
        v = variants[name]
        res["models"][name]["diagnostics_test"] = diagnostics(
            v["model_obj"], sacc[test.numpy()], subset(tt, test), v["use_ior"])
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
