"""The final search model - first fixations only, one construction.

The model builds a PRE-WINDOW priority map over the display -

  map(x) = alpha_P * P(x)   (goal-independent sensory field:
                             bottom-up color salience)
         + g_C * C_T(x)     (target-color evidence)
         + g_F * S_T(x)     (target-shape evidence)
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

Field units are GREYSCALE: background-transparent color salience
lies in [0, 1], while target-color and target-shape evidence lie in [-1, 1].
With the kernel's peak-1 convention every
weight then reads identically - the priority delivered by a
full-strength unit of its field - so alpha_P, g_C, g_F, beta_T,
and beta_D compare directly.

Implementation: the sensed field values are precomputed
(build_contexts.py: A[ctx, item, field] = P, C_T, S_T at the item
centers, greyscale units), the history kernel is the sensed matrix
BH (~identity), and

  F_i = alpha_P*A[i,0] + g_C*A[i,1] + g_F*A[i,2]
        + ((beta_T*h_T + beta_D*h_D) @ BH.T)[i]

SEVEN free parameters - the final form of record:

  alpha_P bottom-up sensory color-salience gain
  g_C     goal-color gain (signed template-axis contrast; the
          identified NET goal modulation - two-color displays
          cannot separate enhancement from suppression, the
          g_T/g_D ridge in RESULTS)
  g_F     goal-shape gain (template match)
  beta_T  pull toward past target locations
  beta_D  push from past distractor locations
  eta_T   weight on recent target locations (leaky accumulator)
  eta_D   weight on recent distractor locations

Fitted values (200 epochs, subject split seed 0, greyscale units):
alpha_P -2.35, g_C -0.653, g_F +0.72, beta_T +2.13, beta_D
-0.48, eta_T 0.60, eta_D 0.16; held-out NLL 1.39499.

Weights live in weights_final.json (written by fit.py).
"""

import json

import torch
import torch.nn as nn

NLOC = 6


class SearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_P = nn.Parameter(torch.tensor(0.0))  # sensory gain
        self.g_C = nn.Parameter(torch.tensor(0.5))   # target-color gain
        self.g_F = nn.Parameter(torch.tensor(1.0))   # template-shape gain
        self.beta_T = nn.Parameter(torch.tensor(0.5))
        self.beta_D = nn.Parameter(torch.tensor(-0.1))
        self.raw_eta_T = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_D = nn.Parameter(torch.tensor(0.0))

    @property
    def eta_T(self):
        return torch.sigmoid(self.raw_eta_T)

    @property
    def eta_D(self):
        return torch.sigmoid(self.raw_eta_D)

    def compute_traces(self, eT, eD):
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

    def _split_evidence(self, A):
        """Return P, C_T, S_T."""
        if A.shape[-1] != 3:
            raise ValueError(f"expected A[..., 3], got {A.shape}")
        return A[..., 0], A[..., 1], A[..., 2]

    def sensory_field(self, A):
        P, _, _ = self._split_evidence(A)
        return self.alpha_P * P

    def goal_field(self, A):
        _, CT, ST = self._split_evidence(A)
        return self.g_C * CT + self.g_F * ST

    def history_field(self, hT, hD, BH):
        vals = self.beta_T * hT + self.beta_D * hD
        return vals @ BH.T

    def field(self, A, hT, hD, BH6, BH4, m6):
        """Point-sensing readout: F_i = M sensed at item i's center.

        A [N,6,3]: per-saccade sensed fields (P, C_T, S_T) in
        dataset units. BH6/BH4: history kernel sensed between item
        centers; m6: set-size-6 mask. No attention window (see the
        module docstring)."""
        stimulus = self.sensory_field(A) + self.goal_field(A)
        F = torch.zeros(A.shape[0], 6)
        for msk, BH in ((m6, BH6), (~m6, BH4)):
            F[msk] = stimulus[msk] + self.history_field(hT[msk], hD[msk], BH)
        return F

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            alpha_P=self.alpha_P.item(),
            g_C=self.g_C.item(), g_F=self.g_F.item(),
            beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
            eta_T=self.eta_T.item(),
            eta_D=self.eta_D.item()).items()}

    def load_values(self, w):
        import numpy as np
        with torch.no_grad():
            self.alpha_P.copy_(torch.tensor(float(w.get("alpha_P", 0.0))))
            for n in ("g_C", "g_F", "beta_T", "beta_D"):
                getattr(self, n).copy_(torch.tensor(float(w[n])))
            self.raw_eta_T.copy_(torch.logit(torch.tensor(float(w["eta_T"]))))
            self.raw_eta_D.copy_(torch.logit(torch.tensor(float(w["eta_D"]))))
        return self


def load_final(path="weights_final.json"):
    return SearchModel().load_values(json.load(open(path)))
