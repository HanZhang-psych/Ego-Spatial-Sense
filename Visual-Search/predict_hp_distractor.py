"""Model predictions for a high-probability distractor-location design
(Wang & Theeuwes 2018-style), generated from the FITTED weights - no new
parameters, no refitting.

Design: setsize 6 ring; singleton present on 70% of trials; when
present it appears at one high-probability (HP) location on 65% of
present trials (each low-probability location 7%); target uniform over
the remaining locations. Phase 1: 400 biased trials. Phase 2: 200
unbiased trials (singleton uniform). First saccades from center.

For each simulated trial the exact softmax choice probabilities are
computed (no sampling); traces evolve by the fitted eta's over
Monte-Carlo sequences. Reported: capture by the singleton at HP vs LP
locations, target-selection cost when the target falls at the HP
location (the source-blind spillover signature), build-up and
extinction time courses.

Usage: python predict_hp_distractor.py [--params results_fit.json]
"""

import argparse
import json

import numpy as np
import torch

from model import SearchEs2Model

S = 6
HP = 0

def load_model(path, variant="traces_ior"):
    gen = json.load(open(path))["models"][variant]["params"]
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
    return m, gen


def first_saccade_probs(model, targ, sing, hT, hD):
    d = torch.full((1, S), 0.5)
    isT = torch.zeros(1, S); isT[0, targ] = 1.0
    isS = torch.zeros(1, S)
    if sing >= 0:
        isS[0, sing] = 1.0
    with torch.no_grad():
        F = model.field(d, isT, isS,
                        torch.tensor(hT, dtype=torch.float32)[None],
                        torch.tensor(hD, dtype=torch.float32)[None])
        return torch.softmax(F, 1)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="results_fit.json")
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--biased", type=int, default=400)
    ap.add_argument("--unbiased", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, gen = load_model(args.params)
    etaT, etaD = gen["eta_T"], gen["eta_D"]
    rng = np.random.default_rng(args.seed)
    T = args.biased + args.unbiased

    acc = {k: np.zeros(T) for k in
           ["p_sing_hp", "n_sing_hp", "p_sing_lp", "n_sing_lp",
            "p_targ_hp", "n_targ_hp", "p_targ_lp", "n_targ_lp"]}
    for _ in range(args.runs):
        hT = np.zeros(S); hD = np.zeros(S)
        for t in range(T):
            present = rng.random() < 0.70
            if present:
                if t < args.biased:
                    sing = HP if rng.random() < 0.65 else \
                        int(rng.choice([j for j in range(S) if j != HP]))
                else:
                    sing = int(rng.choice(S))
            else:
                sing = -1
            targ = int(rng.choice([j for j in range(S) if j != sing]))
            p = first_saccade_probs(model, targ, sing, hT, hD)
            if present:
                key = "hp" if sing == HP else "lp"
                acc[f"p_sing_{key}"][t] += p[sing]
                acc[f"n_sing_{key}"][t] += 1
            key = "hp" if targ == HP else "lp"
            acc[f"p_targ_{key}"][t] += p[targ]
            acc[f"n_targ_{key}"][t] += 1
            hT *= (1 - etaT); hT[targ] += etaT
            eD = np.zeros(S)
            if sing >= 0:
                eD[sing] = 1.0
            hD = (1 - etaD) * hD + etaD * eD

    def rate(pre, lo, hi):
        p = acc[f"p_{pre}"][lo:hi].sum() / max(acc[f"n_{pre}"][lo:hi].sum(), 1)
        return 100 * p

    b = args.biased
    print(f"fitted weights: beta_D={gen['beta_D']:.3f} eta_D={gen['eta_D']:.3f} "
          f"beta_T={gen['beta_T']:.3f} eta_T={gen['eta_T']:.3f}\n")
    print("== biased phase, asymptote (last 200 biased trials) ==")
    print(f"first saccades to singleton at HP location: {rate('sing_hp', b-200, b):.2f}%")
    print(f"first saccades to singleton at LP location: {rate('sing_lp', b-200, b):.2f}%")
    print(f"first saccades to target when target at HP: {rate('targ_hp', b-200, b):.2f}%")
    print(f"first saccades to target when target at LP: {rate('targ_lp', b-200, b):.2f}%")
    print("\n== build-up (biased phase, bins of 25 trials, singleton-at-HP capture %) ==")
    for lo in range(0, min(b, 200), 25):
        print(f"trials {lo+1:3d}-{lo+25:3d}: HP {rate('sing_hp', lo, lo+25):.2f}%  "
              f"LP {rate('sing_lp', lo, lo+25):.2f}%")
    print("\n== extinction (unbiased phase, bins of 25, former-HP capture %) ==")
    for lo in range(b, min(b+150, T), 25):
        print(f"trials +{lo-b+1:3d}-{lo-b+25:3d}: former-HP {rate('sing_hp', lo, lo+25):.2f}%  "
              f"other {rate('sing_lp', lo, lo+25):.2f}%")


if __name__ == "__main__":
    main()
