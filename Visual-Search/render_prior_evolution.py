"""Render the pre-onset spatial prior evolving over trials.

The prior in closed form (final model, history-inside ordering):
  F_pre(q) = window(q) * [beta_T*h_T(q) + beta_D*h_D(q)]
with the window centered on the pre-trial fixation (display center)
and the traces painted at the six item slots. Simulated design:
targets appear at one slot (green outline, bottom) on 70% of trials,
singletons at the opposite slot (red outline, top) on 70% of
singleton-present trials (70% present), for 60 biased trials; then 30
unbiased trials (both uniform) to show decay.
Weights are the fitted history-inside values (weights_final.json).

Output: figures/prior_evolution.png
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = json.load(open("weights_final.json"))
r0 = p["r0"]
BT, BD, ETA_T, ETA_D = p["beta_T"], p["beta_D"], p["eta_T"], p["eta_D"]
K = p["k"]

S = 6
ANG = [2 * np.pi * j / S - np.pi / 2 for j in range(S)]
POS = [(0.5 * np.cos(a), 0.5 * np.sin(a)) for a in ANG]
T_LOC, S_LOC = 0, 3          # top slot for targets, bottom for singletons

G = 161
ax_ = np.linspace(-0.75, 0.75, G)
XX, YY = np.meshgrid(ax_, ax_)
RR = np.hypot(XX, YY)
WINDOW = 1 / (1 + np.exp(-K * (r0 - RR)))
SIG = 0.09


def prior_map(hT, hD):
    m = np.zeros_like(XX)
    for j in range(S):
        bump = np.exp(-((XX - POS[j][0]) ** 2 + (YY - POS[j][1]) ** 2)
                      / (2 * SIG ** 2))
        m += (BT * hT[j] + BD * hD[j]) * bump
    return WINDOW * m


def main():
    rng = np.random.default_rng(3)
    hT = np.zeros(S)
    hD = np.zeros(S)
    snaps = {}
    series_T, series_S = [], []
    snap_at = [0, 10, 30, 60, 90]
    for t in range(91):
        if t in snap_at:
            snaps[t] = prior_map(hT, hD)
        w_slot = 1 / (1 + np.exp(-K * (r0 - 0.5)))   # window at ring radius
        series_T.append(w_slot * BT * hT[T_LOC])
        series_S.append(w_slot * BD * hD[S_LOC])
        biased = t < 60
        targ = T_LOC if (biased and rng.random() < 0.7) else int(rng.choice(S))
        present = rng.random() < 0.7
        if present:
            sing = S_LOC if (biased and rng.random() < 0.7) \
                else int(rng.choice([j for j in range(S) if j != targ]))
        else:
            sing = -1
        while targ == sing:
            targ = int(rng.choice(S))
        hT = (1 - ETA_T) * hT
        hT[targ] += ETA_T
        eD = np.zeros(S)
        if sing >= 0:
            eD[sing] = 1.0
        hD = (1 - ETA_D) * hD + ETA_D * eD

    vmax = max(snaps[k].max() for k in snaps)
    vmin = min(snaps[k].min() for k in snaps)
    # two-slope norm: enhancement and suppression each use their full
    # half of the colormap (a symmetric scale hides the shallow
    # suppression under the ~7x larger target bump)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(vmin, -1e-6),
                        vmax=max(vmax, 1e-6))
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[2.2, 1])
    titles = ["trial 0 (no history)", "trial 10 (biased)", "trial 30 (biased)",
              "trial 60 (end of bias)", "trial 90 (30 unbiased)"]
    for i, t in enumerate(snap_at):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(snaps[t], origin="lower",
                       extent=[-.75, .75, -.75, .75],
                       cmap="RdBu_r", norm=norm)
        for j, (x, y) in enumerate(POS):
            ec = ("#118844" if j == T_LOC else
                  "#aa2222" if j == S_LOC else "#888888")
            ax.add_patch(plt.Circle((x, y), 0.07, fill=False, ec=ec, lw=1.4))
        ax.plot(0, 0, "+", color="k", ms=8)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axc = fig.add_subplot(gs[1, :])
    axc.plot(series_T, color="#118844",
             label="prior at target-biased slot (window-weighted)")
    axc.plot(series_S, color="#aa2222",
             label="prior at singleton-biased slot (window-weighted)")
    axc.axhline(0, color="#999999", lw=0.8)
    axc.axvline(60, color="#555555", ls=":", lw=1)
    axc.text(61, axc.get_ylim()[1] * 0.75, "bias removed", fontsize=9)
    axc.set_xlabel("trial")
    axc.set_ylabel("prior value at slot")
    axc.legend(fontsize=9, loc="upper left")
    plt.tight_layout(rect=[0.015, 0, 0.955, 0.95])
    cax = fig.add_axes([0.965, 0.42, 0.011, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("prior (red = enhance, blue = suppress;\n"
                 "halves scaled separately)", fontsize=8)
    fig.suptitle("Pre-onset spatial prior F_pre = window x history, evolving "
                 "over a biased sequence (fitted weights)", fontsize=12)
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/prior_evolution.png", dpi=130)
    print("saved figures/prior_evolution.png")


if __name__ == "__main__":
    main()
