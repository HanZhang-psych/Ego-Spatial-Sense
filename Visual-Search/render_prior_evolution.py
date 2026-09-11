"""Render the pre-onset spatial prior evolving over trials.

The prior in closed form (final model, history-inside ordering):
  F_pre(q) = window(q) * [beta_T*h_T(q) + beta_D*h_D(q)]
with the window centered on the pre-trial fixation (display center)
and the traces painted at the six item slots. Simulated design
(Wang & Theeuwes): the singleton appears at ONE high-probability
slot (red outline, top) on 65% of singleton-present trials (70%
present); the TARGET is always uniform - W&T bias only the
distractor location. 60 biased trials, then 30 unbiased to show
decay.
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
S_LOC = 3                    # the high-probability singleton slot (top)

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


def simulate(rng):
    """One W&T sequence; yields (t, hT, hD, targ, sing) per trial."""
    hT = np.zeros(S)
    hD = np.zeros(S)
    for t in range(91):
        biased = t < 60
        present = rng.random() < 0.7
        if present:
            sing = S_LOC if (biased and rng.random() < 0.65) \
                else int(rng.choice(S))
        else:
            sing = -1
        targ = int(rng.choice([j for j in range(S) if j != sing]))
        yield t, hT.copy(), hD.copy(), targ, sing
        hT = (1 - ETA_T) * hT
        hT[targ] += ETA_T
        eD = np.zeros(S)
        if sing >= 0:
            eD[sing] = 1.0
        hD = (1 - ETA_D) * hD + ETA_D * eD


def main():
    snap_at = [0, 10, 30, 60, 90]
    # maps: EXPECTED traces, averaged over many sequences (the target
    # trace is spatially uniform in expectation - W&T targets are
    # unbiased - so the systematic structure is the HP-slot dip)
    RUNS = 400
    accT = {t: np.zeros(S) for t in snap_at}
    accD = {t: np.zeros(S) for t in snap_at}
    rng = np.random.default_rng(0)
    for _ in range(RUNS):
        for t, hT, hD, targ, sing in simulate(rng):
            if t in snap_at:
                accT[t] += hT / RUNS
                accD[t] += hD / RUNS
    snaps = {t: prior_map(accT[t], accD[t]) for t in snap_at}

    # series + event rugs: ONE example sequence
    rng = np.random.default_rng(3)
    series_T, series_S = [], []
    ev_T, ev_S = [], []
    w_slot = 1 / (1 + np.exp(-K * (r0 - 0.5)))       # window at ring radius
    for t, hT, hD, targ, sing in simulate(rng):
        prior = w_slot * (BT * hT + BD * hD)
        series_S.append(prior[S_LOC])
        series_T.append(np.delete(prior, S_LOC).mean())
        ev_T.append(targ)
        ev_S.append(sing)

    # symmetric scale: in EXPECTATION the HP slot's suppression and
    # its (deficient) target-history prior nearly cancel, so the
    # honest picture is a neutral hole in a red ring - an asymmetric
    # norm would stretch the ~0.02 residual negative into full blue
    from matplotlib.colors import Normalize
    vmax = max(abs(snaps[k]).max() for k in snaps)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[2.2, 1])
    titles = ["trial 0 (no history)", "trial 10 (biased)", "trial 30 (biased)",
              "trial 60 (end of bias)", "trial 90 (30 unbiased)"]
    # (maps: average over sequences; curve below: one example sequence)
    for i, t in enumerate(snap_at):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(snaps[t], origin="lower",
                       extent=[-.75, .75, -.75, .75],
                       cmap="RdBu_r", norm=norm)
        for j, (x, y) in enumerate(POS):
            ec = "#aa2222" if j == S_LOC else "#888888"
            ax.add_patch(plt.Circle((x, y), 0.07, fill=False, ec=ec, lw=1.4))
        ax.plot(0, 0, "+", color="k", ms=8)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axc = fig.add_subplot(gs[1, :])
    axc.plot(series_S, color="#aa2222",
             label="prior at the HP singleton slot")
    axc.plot(series_T, color="#888888",
             label="prior averaged over the other five slots")
    # event rugs: where the target / singleton ACTUALLY appeared each
    # trial (filled = at the biased slot, faint tick = elsewhere,
    # nothing = singleton absent)
    tt_ = np.arange(len(ev_T))
    tb = np.array([e == S_LOC for e in ev_T])     # target landed at HP slot
    sb = np.array([e == S_LOC for e in ev_S])
    sp_ = np.array([e >= 0 for e in ev_S])
    yT, yS = -0.62, -0.80
    axc.scatter(tt_[sb], np.full(sb.sum(), yS), marker="|", s=48,
                color="#aa2222", lw=1.6,
                label="trial's singleton at the HP slot")
    axc.scatter(tt_[sp_ & ~sb], np.full((sp_ & ~sb).sum(), yS),
                marker="|", s=20, color="#ddbbbb", lw=1.0,
                label="singleton elsewhere (absent: no tick)")
    axc.scatter(tt_[tb], np.full(tb.sum(), yT), marker="|", s=48,
                color="#118844", lw=1.6,
                label="trial's (uniform) target at the HP slot")
    axc.set_ylim(-0.95, None)
    axc.axhline(0, color="#999999", lw=0.8)
    axc.axvline(60, color="#555555", ls=":", lw=1)
    axc.text(61, axc.get_ylim()[1] * 0.80, "bias removed", fontsize=9)
    axc.set_xlabel("trial")
    axc.set_ylabel("prior value at slot")
    axc.legend(fontsize=9, loc="upper left")
    plt.tight_layout(rect=[0.015, 0, 0.955, 0.95])
    cax = fig.add_axes([0.965, 0.42, 0.011, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("expected prior (symmetric scale)", fontsize=8)
    fig.suptitle("Pre-onset spatial prior F_pre = window x history, W&T "
                 "HP-distractor design (maps: expected over sequences; "
                 "curves: one example sequence)", fontsize=12)
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/prior_evolution.png", dpi=130)
    print("saved figures/prior_evolution.png")


if __name__ == "__main__":
    main()
