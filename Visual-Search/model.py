"""The final search model - first fixations only, one construction.

The model builds a PRE-WINDOW priority map over the display -

  map(x) = g_T * relu(target-color contrast at x)
         - g_D_eff * relu(distractor-color contrast at x)
         + g_F * shape match at x
         + beta_T * h_T field(x) + beta_D * h_D field(x)

(the history fields are each item's history value smoothed by a
fixed kernel with PEAK HEIGHT 1, PAINT_SIG in build_contexts.py - a
stated model assumption) - multiplies it by the ego-anchored
attention window sigmoid(k*(r0 - d)), and SENSES it at each item's
center: F_i = P(x_i), the point-sensing readout (adopted 2026-09-11;
see RESULTS). Softmax over the six sensed values predicts the first
saccade.

Implementation: the sensed channel values are precomputed
(build_contexts.py: A[ctx, item, channel], with the distractor
channel stored as MINUS relu(distractor contrast), so the fitted
g_D > 0 means suppression), the history kernel is the sensed matrix
BH (~identity), and

  F_i = sigmoid(k*(r0 - d_i)) *
        [ g_T*A[i,0] + g_D*A[i,1] + g_F*A[i,2]
          + ((beta_T*h_T + beta_D*h_D) @ BH.T)[i] ]

With all items on one ring, the window weight is one shared scalar -
a pure softmax temperature - so k, r0 are not separately
identifiable in scope and stay on theoretical definition.

Notation: g_* stimulus gains, beta_* history gains, eta_* memory
speeds, k/r0 the attention window.

Weights live in weights_final.json (written by fit.py).
"""

import json

import torch
import torch.nn as nn

NLOC = 6


class SearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_T = nn.Parameter(torch.tensor(0.5))        # enhance target color
        self.g_D = nn.Parameter(torch.tensor(0.3))        # suppress distractor color
        self.g_F = nn.Parameter(torch.tensor(1.0))   # template-shape gain
        self.raw_k = nn.Parameter(torch.tensor(1.0))    # window steepness
        self.r0 = nn.Parameter(torch.tensor(0.5))       # window reach
        self.beta_T = nn.Parameter(torch.tensor(0.5))
        self.beta_D = nn.Parameter(torch.tensor(-0.1))
        self.raw_eta_T = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_D = nn.Parameter(torch.tensor(0.0))
        self.raw_sigma = nn.Parameter(torch.tensor(-12.0))  # trace spread (off)

    @property
    def k(self):
        return nn.functional.softplus(self.raw_k)

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

    def field(self, A, hT, hD, BH6, BH4, m6, D6, D4):
        """Point-sensing readout: F_i = w(d_i) * P-sensed-at-item-i.

        A [N,6,3]: per-saccade sensed channels (relu target-color
        contrast, MINUS relu distractor-color contrast, shape match)
        in dataset units - g_D > 0 therefore means suppression.
        BH6/BH4: history kernel sensed between item centers; D6/D4:
        sensed distances from fixation; m6: set-size-6 mask."""
        vals = self.beta_T * hT + self.beta_D * hD
        sal = (self.g_T * A[..., 0] + self.g_D * A[..., 1]
               + self.g_F * A[..., 2])
        F = torch.zeros(A.shape[0], 6)
        for msk, BH, D in ((m6, BH6, D6), (~m6, BH4, D4)):
            win = torch.sigmoid(self.k * (self.r0 - D))
            F[msk] = win * (sal[msk] + vals[msk] @ BH.T)
        return F

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            g_T=self.g_T.item(), g_D=self.g_D.item(),
            g_F=self.g_F.item(), k=self.k.item(), r0=self.r0.item(),
            beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
            eta_T=self.eta_T.item(), eta_D=self.eta_D.item(),
            sigma=self.sigma.item()).items()}

    def load_values(self, w):
        import numpy as np
        with torch.no_grad():
            for n in ("g_T", "g_D", "g_F", "r0",
                      "beta_T", "beta_D"):
                getattr(self, n).copy_(torch.tensor(float(w[n])))
            self.raw_k.copy_(torch.tensor(float(np.log(np.expm1(w["k"])))))
            self.raw_eta_T.copy_(torch.logit(torch.tensor(float(w["eta_T"]))))
            self.raw_eta_D.copy_(torch.logit(torch.tensor(float(w["eta_D"]))))
        return self


def load_final(path="weights_final.json"):
    return SearchModel().load_values(json.load(open(path)))
