import json
"""Six-panel results figure (figures/results.png). Run from Visual-Search/.

Requires matplotlib. Panels: A held-out model comparison; B suppression
effect; C/D target/singleton location-priming decay curves (observed vs
model); E parameter recovery; F HP-distractor-location predictions.
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fit_pooled import build_tensors, probs
from model import SearchEs2Model

res = json.load(open("results_fit.json"))
rec = json.load(open("results_recovery.json"))

sacc = pd.read_csv("dataset/saccades.csv")
ev = pd.read_csv("dataset/events.csv")
sacc, tt = build_tensors(sacc, ev)

gen = res["models"]["traces_ior"]["params"]
m = SearchEs2Model()
with torch.no_grad():
    m.raw_k.copy_(torch.tensor(np.log(np.expm1(gen["k"]))))
    m.g_T.copy_(torch.tensor(gen["g_T"]))
    m.g_S.copy_(torch.tensor(gen["g_S"]))
    m.beta_T.copy_(torch.tensor(gen["beta_T"]))
    m.beta_D.copy_(torch.tensor(gen["beta_D"]))
    m.raw_eta_T.copy_(torch.logit(torch.tensor(gen["eta_T"])))
    m.raw_eta_D.copy_(torch.logit(torch.tensor(gen["eta_D"])))
    m.g_I.copy_(torch.tensor(gen["g_I"]))
p_all = probs(m, tt, use_ior=True).detach().numpy()

# ---- lag (priming) analysis on first saccades ----
first = (sacc.saccindex == 1).values
ev2 = ev.copy()
ev2["block"] = pd.to_numeric(ev2["block"], errors="coerce").fillna(0.0)
ev2["trial"] = pd.to_numeric(ev2["trial"], errors="coerce")
ev2["subj"] = ev2["subj"].astype(str)
ev2 = ev2.sort_values(["study", "subj", "block", "trial"]).reset_index(drop=True)
ev2["skey"] = ev2.study.astype(str) + "|" + ev2.subj.astype(str)
ev2["si"] = ev2.skey.map({s: i for i, s in enumerate(ev2.skey.unique())})
ev2["ti"] = ev2.groupby("si").cumcount()
targ_by = {}
sing_by = {}
for s, sub in ev2.groupby("si"):
    targ_by[s] = sub.targLoc.values
    sing_by[s] = sub.singLoc.values

MAXLAG = 8
obsT = np.zeros(MAXLAG + 1); nT = np.zeros(MAXLAG + 1); modT = np.zeros(MAXLAG + 1)
obsD = np.zeros(MAXLAG + 1); nD = np.zeros(MAXLAG + 1); modD = np.zeros(MAXLAG + 1)
si_arr = tt["si"].numpy(); ti_arr = tt["ti"].numpy()
choice = tt["choice"].numpy(); valid = tt["valid"].numpy()
targLoc = sacc.targLoc.values; singLoc = sacc.singLoc.values
setsize = sacc.setsize.values
idxs = np.where(first)[0]
for i in idxs:
    s, t = si_arr[i], ti_arr[i]
    tv, sv = targ_by[s], sing_by[s]
    for j in range(1, setsize[i] + 1):
        if j == targLoc[i] or j == singLoc[i] or not valid[i, j - 1]:
            continue
        # lag since j was the target
        lagT = 0
        for L in range(1, min(MAXLAG, t) + 1):
            if tv[t - L] == j:
                lagT = L
                break
        if lagT:
            obsT[lagT] += (choice[i] == j - 1)
            modT[lagT] += p_all[i, j - 1]
            nT[lagT] += 1
        lagD = 0
        for L in range(1, min(MAXLAG, t) + 1):
            if sv[t - L] == j:
                lagD = L
                break
        if lagD:
            obsD[lagD] += (choice[i] == j - 1)
            modD[lagD] += p_all[i, j - 1]
            nD[lagD] += 1

base_obs = np.mean([(choice[i] == j - 1)
                    for i in idxs for j in range(1, setsize[i] + 1)
                    if j != targLoc[i] and j != singLoc[i] and valid[i, j - 1]])
lags = np.arange(1, MAXLAG + 1)

# ---- chance NLL ----
setn = np.where(sacc.fixloc.values > 0, setsize - 1, setsize)
chance = np.log(setn).mean()

# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
fig.suptitle("Priority-field model fitted to 217,595 human saccades "
             "(Drennan & Gaspelin 2026 pooled data)", fontsize=13)

ax = axes[0, 0]
names = ["chance", "gains+env\n(3 wts)", "+traces\n(7 wts)", "+IoR\n(8 wts)"]
vals = [chance,
        res["models"]["null_no_traces"]["test_nll_per_saccade"],
        res["models"]["traces"]["test_nll_per_saccade"],
        res["models"]["traces_ior"]["test_nll_per_saccade"]]
cols = ["#bbbbbb", "#8888cc", "#5555bb", "#2233aa"]
ax.bar(names, vals, color=cols)
ax.set_ylim(1.15, 1.8)
ax.set_ylabel("held-out NLL per saccade (lower = better)")
ax.set_title("A  Model comparison, held-out subjects")
for x, v in enumerate(vals):
    ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

ax = axes[0, 1]
d = res["models"]["traces_ior"]["diagnostics_test"]["first_saccade_%"]
labels = ["target", "singleton", "nonsingleton\n(per item)"]
obs = [d["target"]["obs"], d["singleton"]["obs"], d["nonsingleton_per_item"]["obs"]]
mod = [d["target"]["model"], d["singleton"]["model"], d["nonsingleton_per_item"]["model"]]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, obs, w, label="observed", color="#444444")
ax.bar(x + w/2, mod, w, label="model", color="#2233aa")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("% of first saccades")
ax.axhline(obs[2], color="#999999", ls=":", lw=1)
ax.set_title("B  Oculomotor suppression (held-out)")
ax.legend()

ax = axes[0, 2]
ax.plot(lags, 100*obsT[1:]/np.maximum(nT[1:], 1), "o-", color="#444444", label="observed")
ax.plot(lags, 100*modT[1:]/np.maximum(nT[1:], 1), "s--", color="#118844", label="model")
ax.axhline(100*base_obs, color="#999999", ls=":", lw=1, label="baseline")
ax.set_xlabel("trials since this location held the TARGET")
ax.set_ylabel("% first saccades to that location")
ax.set_title("C  Target-location priming decay")
ax.legend()

ax = axes[1, 0]
ax.plot(lags, 100*obsD[1:]/np.maximum(nD[1:], 1), "o-", color="#444444", label="observed")
ax.plot(lags, 100*modD[1:]/np.maximum(nD[1:], 1), "s--", color="#aa2222", label="model")
ax.axhline(100*base_obs, color="#999999", ls=":", lw=1, label="baseline")
ax.set_xlabel("trials since this location held the SINGLETON")
ax.set_ylabel("% first saccades to that location")
ax.set_title("D  Singleton-location suppression decay")
ax.legend()

ax = axes[1, 1]
g = res["models"]["traces_ior"]["params"]
r = rec["models"]["traces_ior"]["params"]
keys = ["g_T", "g_S", "beta_T", "beta_D", "eta_T", "eta_D", "g_I", "k"]
gv = [g[k] for k in keys]; rv = [r[k] for k in keys]
ax.scatter(gv, rv, color="#2233aa", zorder=3)
lim = [min(gv+rv) - 0.4, max(gv+rv) + 0.4]
ax.plot(lim, lim, color="#999999", lw=1)
for k_, x_, y_ in zip(keys, gv, rv):
    ax.annotate(k_, (x_, y_), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("generating value"); ax.set_ylabel("recovered value")
ax.set_title("E  Parameter recovery (synthetic pool)")

ax = axes[1, 2]
labels = ["singleton\ncapture,\nHP loc", "singleton\ncapture,\nLP loc",
          "target\nfound,\nat HP loc", "target\nfound,\nelsewhere"]
vals = [3.16, 4.60, 43.14, 50.66]
cols = ["#aa2222", "#dd8888", "#118844", "#88cc99"]
ax.bar(labels, vals, color=cols)
for x_, v in enumerate(vals):
    ax.text(x_, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("% of first saccades (predicted)")
ax.set_title("F  Prediction: high-probability distractor location")

plt.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("figures", exist_ok=True)
out = "figures/results.png"
plt.savefig(out, dpi=130)
print("saved", out)
print("lag counts T:", nT[1:].astype(int).tolist())
print("lag counts D:", nD[1:].astype(int).tolist())
