"""The final search model (one class, one loader).

  F_i = window(dist_i) is implicit in the sensor for the stimulus part:
  stim_i = sum_bins sigmoid(k*(r0 - r)) * relu(g_T*D_T + g_O*D_O + w_p*D_P)
           + g_form*FORM_i
  F_i = stim_i + sigmoid(k*(r0 - dist_i)) * (beta_T*hT_i + beta_D*hD_i
                                             + g_I*visited_i)
  P(saccade -> i) = softmax over the choice set

Goal-early single priority map; ES2-form sigmoid attention window
(fitted k, r0) gating stimulus evidence AND memory (history-inside,
decided by held-out comparison); feature-level suppression is
relegation-only; signed writing is reserved for the spatial memories.
Weights are stored in weights_final.json by fit.py; load_final() reads
them back.
"""

import json

import torch
import torch.nn as nn

NLOC = 6


class SearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_T = nn.Parameter(torch.tensor(0.5))      # template-color gain
        self.g_O = nn.Parameter(torch.tensor(0.0))      # orthogonal axis (~0)
        self.w_p = nn.Parameter(torch.tensor(0.3))      # presence gain
        self.g_form = nn.Parameter(torch.tensor(1.0))   # template-shape gain
        self.raw_k = nn.Parameter(torch.tensor(1.0))    # window steepness
        self.r0 = nn.Parameter(torch.tensor(0.5))       # window reach
        self.beta_T = nn.Parameter(torch.tensor(0.5))   # target-trace weight
        self.beta_D = nn.Parameter(torch.tensor(-0.1))  # distractor-trace weight
        self.raw_eta_T = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_D = nn.Parameter(torch.tensor(0.0))
        self.g_I = nn.Parameter(torch.tensor(-0.5))     # IoR penalty
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
        """Replay the trial sequences; return the memories AS OF each
        trial. Optional dmat [S, 6, 6] spreads updates over ring
        neighbors with width sigma (the sigma_h variant)."""
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

    def field(self, P, FORM, dist, visited, hT, hD, radii):
        win_ray = torch.sigmoid(self.k * (self.r0 - radii))
        mix = (self.g_T * P[..., 0] + self.g_O * P[..., 1]
               + self.w_p * P[..., 2])
        stim = (torch.relu(mix) * win_ray).sum(-1) + self.g_form * FORM
        win_item = torch.sigmoid(self.k * (self.r0 - dist))
        return stim + win_item * (self.beta_T * hT + self.beta_D * hD
                                  + self.g_I * visited.float())

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            g_T=self.g_T.item(), g_O=self.g_O.item(), w_p=self.w_p.item(),
            g_form=self.g_form.item(), k=self.k.item(), r0=self.r0.item(),
            beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
            eta_T=self.eta_T.item(), eta_D=self.eta_D.item(),
            g_I=self.g_I.item(), sigma=self.sigma.item()).items()}

    def load_values(self, w):
        import numpy as np
        with torch.no_grad():
            for n in ("g_T", "g_O", "w_p", "g_form", "r0",
                      "beta_T", "beta_D", "g_I"):
                getattr(self, n).copy_(torch.tensor(float(w[n])))
            self.raw_k.copy_(torch.tensor(float(np.log(np.expm1(w["k"])))))
            self.raw_eta_T.copy_(torch.logit(torch.tensor(float(w["eta_T"]))))
            self.raw_eta_D.copy_(torch.logit(torch.tensor(float(w["eta_D"]))))
        return self


def load_final(path="weights_final.json"):
    return SearchModel().load_values(json.load(open(path)))
