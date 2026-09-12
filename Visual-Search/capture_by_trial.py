# observed vs model oculomotor capture effect across the session
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data, reproduce
from model import load_final

m = load_final()
sacc, ev = data.load_frames()
sacc, tt = data.build_tensors(sacc, ev)
S = np.load("dataset/senses.npz")
cid = torch.tensor(sacc.ctx.values.astype(int))
A = torch.tensor(S["A"])[cid]
prob = reproduce.model_probs(m, sacc, tt, A,
                             torch.tensor(S["BH6"]), torch.tensor(S["BH4"]),
                             torch.tensor(S["D6"]), torch.tensor(S["D4"])).numpy()
_, test = data.subject_split(tt)
held = test.numpy()

N = len(sacc)
sing = sacc.singLoc.values
targ = sacc.targLoc.values
choice = tt["choice"].numpy() + 1
valid = tt["valid"].numpy()
ti = tt["ti"].numpy()          # within-subject trial counter

sp = (sing > 0) & held
nplain = valid[sp].sum(1) - 2
obs_s = (choice[sp] == sing[sp]).astype(float)
obs_t = (choice[sp] == targ[sp]).astype(float)
obs_base = (1 - obs_s - obs_t) / np.maximum(nplain, 1)
mod_s = prob[sp, sing[sp] - 1]
mod_t = prob[sp, targ[sp] - 1]
mod_base = (1 - mod_s - mod_t) / np.maximum(nplain, 1)
tt_i = ti[sp]

df = pd.DataFrame(dict(t=tt_i, oc=(obs_s - obs_base) * 100,
                       mc=(mod_s - mod_base) * 100))
df = df[df.t < 120]
g = df.groupby("t").agg(["mean", "sem", "size"])
# running mean (window 15, like the paper's smoothing)
win = 15
oc = g[("oc", "mean")].rolling(win, center=True, min_periods=5).mean()
ocse = g[("oc", "sem")].rolling(win, center=True, min_periods=5).mean() / np.sqrt(win)
mc = g[("mc", "mean")].rolling(win, center=True, min_periods=5).mean()

plt.figure(figsize=(8, 4.5))
plt.axhline(0, color="k", ls=":", lw=1)
plt.fill_between(oc.index, oc - 1.96*ocse*np.sqrt(win), oc + 1.96*ocse*np.sqrt(win),
                 color="#bbbbbb", alpha=0.6, label="observed (95% band)")
plt.plot(oc.index, oc, color="#222222", lw=1.5, label="observed capture effect")
plt.plot(mc.index, mc, color="#2233aa", lw=2, label="model")
plt.xlabel("trial (within subject)")
plt.ylabel("oculomotor capture effect (%)")
plt.title("singleton fixations minus plain-item baseline, held-out subjects\n"
          f"(running mean, window {win})")
plt.legend()
plt.tight_layout()
plt.savefig("figures/capture_by_trial.png", dpi=130)
print("first 10 trials: obs", df[df.t < 10].oc.mean().round(2),
      " model", df[df.t < 10].mc.mean().round(2))
print("trials 60-119:  obs", df[df.t >= 60].oc.mean().round(2),
      " model", df[df.t >= 60].mc.mean().round(2))
print("saved figures/capture_by_trial.png")
