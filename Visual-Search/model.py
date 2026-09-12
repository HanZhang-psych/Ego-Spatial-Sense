"""The final search model - first fixations only, one construction.

The model builds a PRE-WINDOW priority map over the display -

  map(x) = g_C * D_T(x)     (SIGNED template-axis color contrast:
                             one gain lifts goal-colored locations
                             and depresses opposite-colored ones)
         + g_F * shape match at x
         + beta_T * h_T field(x) + beta_D * h_D field(x)

(the history fields are each item's history value smoothed by a
fixed kernel with PEAK HEIGHT 1, PAINT_SIG in build_contexts.py - a
stated model assumption) - and SENSES it at each item's center:
F_i = M(x_i), the point-sensing readout (adopted 2026-09-11; see
RESULTS). Softmax over the six sensed values predicts the first
saccade. The model carries NO attention window (removed
2026-09-12): in first-fixation scope, with all items equidistant
from central fixation, a window multiplies every sensed priority
by one shared scalar the gains absorb - fits with and without it
are exactly identical (RESULTS).

Implementation: the sensed channel values are precomputed
(build_contexts.py: A[ctx, item, channel] = signed D_T and shape
match at the item centers), the history kernel is the sensed matrix
BH (~identity), and

  F_i = g_C*A[i,0] + g_F*A[i,1]
        + ((beta_T*h_T + beta_D*h_D) @ BH.T)[i]

SIX free parameters - the final form of record:

  g_C     goal-color gain (signed template-axis contrast; the
          identified NET goal modulation - two-color displays
          cannot separate enhancement from suppression, the
          g_T/g_D ridge in RESULTS)
  g_F     goal-shape gain (template match)
  beta_T  pull toward past target locations
  beta_D  push from past distractor locations
  eta_T   target-memory speed (leaky accumulator)
  eta_D   distractor-memory speed

Every parameter is identified and sign-interpretable; fitted
values (600 epochs, subject split seed 0): g_C +0.226, g_F +0.498,
beta_T +2.13, beta_D -0.49, eta_T 0.60, eta_D 0.16; held-out NLL
1.39749 (script split) / 1.37415 (notebook split).

Weights live in weights_final.json (written by fit.py).
"""

import json

import torch
import torch.nn as nn

NLOC = 6


class SearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_C = nn.Parameter(torch.tensor(0.5))   # goal-color gain (signed D_T)
        self.g_F = nn.Parameter(torch.tensor(1.0))   # template-shape gain
        self.beta_T = nn.Parameter(torch.tensor(0.5))
        self.beta_D = nn.Parameter(torch.tensor(-0.1))
        self.raw_eta_T = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_D = nn.Parameter(torch.tensor(0.0))
        self.raw_sigma = nn.Parameter(torch.tensor(-12.0))  # trace spread (off)

    @property
    def eta_T(self):
        return torch.sigmoid(self.raw_eta_T)

    @property
    def eta_D(self):
        return torch.sigmoid(self.raw_eta_D)

    @property
    def sigma(self):
        return nn.functional.softplus(self.raw_sigma)

    def compute_traces(self, eT, eD, dmat=None):
        if dmat is not None:
            K = torch.exp(-dmat ** 2 / (2 * self.sigma ** 2 + 1e-8))
            K = K / K.sum(-1, keepdim=True)
            eT = torch.einsum("stj,sjk->stk", eT, K)
            eD = torch.einsum("stj,sjk->stk", eD, K)
        S, T, L = eT.shape
        hT = torch.zeros(S, L)
        hD = torch.zeros(S, L)
        outT = torch.empty(S, T, L)
        outD = torch.empty(S, T, L)
        for t in range(T):
            outT[:, t] = hT
            outD[:, t] = hD
            hT = (1 - self.eta_T) * hT + self.eta_T * eT[:, t]
            hD = (1 - self.eta_D) * hD + self.eta_D * eD[:, t]
        return outT, outD

    def field(self, A, hT, hD, BH6, BH4, m6):
        """Point-sensing readout: F_i = M sensed at item i's center.

        A [N,6,2]: per-saccade sensed channels (signed template-axis
        contrast D_T, shape match) in dataset units. BH6/BH4:
        history kernel sensed between item centers; m6: set-size-6
        mask. No attention window (see the module docstring)."""
        vals = self.beta_T * hT + self.beta_D * hD
        sal = self.g_C * A[..., 0] + self.g_F * A[..., 1]
        F = torch.zeros(A.shape[0], 6)
        for msk, BH in ((m6, BH6), (~m6, BH4)):
            F[msk] = sal[msk] + vals[msk] @ BH.T
        return F

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            g_C=self.g_C.item(), g_F=self.g_F.item(),
            beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
            eta_T=self.eta_T.item(), eta_D=self.eta_D.item(),
            sigma=self.sigma.item()).items()}

    def load_values(self, w):
        import numpy as np
        with torch.no_grad():
            for n in ("g_C", "g_F", "beta_T", "beta_D"):
                getattr(self, n).copy_(torch.tensor(float(w[n])))
            self.raw_eta_T.copy_(torch.logit(torch.tensor(float(w["eta_T"]))))
            self.raw_eta_D.copy_(torch.logit(torch.tensor(float(w["eta_D"]))))
        return self


def load_final(path="weights_final.json"):
    return SearchModel().load_values(json.load(open(path)))
