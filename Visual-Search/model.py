"""The final search model - minimal perceptual module.

Stimulus terms, all computed from the color-contrast maps:

  mix(x) = a * (target-color contrast at x)
         - b * (distractor-color contrast at x)
  stim_i = sum over ray bins of  sigmoid(k*(r0 - r)) * relu(mix)
         + g_form * FORM_i
  F_i    = stim_i + sigmoid(k*(r0 - dist_i)) * (beta_T*hT_i + beta_D*hD_i
                                                + g_I*visited_i)

a = enhance the target color; b = suppress the distractor color (the
two are nearly yoked within any single color-pair study - only their
combination is well identified there; separating them needs >=3-color
displays). The target-color projection is the stored D_T profile; the
distractor-color projection is a fixed per-display rotation of
(D_T, D_O) by the angle between the two colors. No object-presence
term: it is invisible to the softmax up to small shape-area
differences (dropping it costs ~24 total held-out NLL).

Weights live in weights_final.json (written by fit.py).
"""

import json

import torch
import torch.nn as nn

NLOC = 6


class SearchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))        # enhance target color
        self.b = nn.Parameter(torch.tensor(0.3))        # suppress distractor color
        self.g_form = nn.Parameter(torch.tensor(1.0))   # template-shape gain
        self.raw_k = nn.Parameter(torch.tensor(1.0))    # window steepness
        self.r0 = nn.Parameter(torch.tensor(0.5))       # window reach
        self.beta_T = nn.Parameter(torch.tensor(0.5))
        self.beta_D = nn.Parameter(torch.tensor(-0.1))
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

    def field(self, P, FORM, dist, visited, hT, hD, radii, cphi, sphi,
              rect="after"):
        """cphi/sphi: per-observation cosine/sine of the angle between
        the distractor color and the target color in opponency space
        (0 on singleton-absent trials). rect: "after" rectifies the
        goal-weighted sum (suppression saturates at zero - relegation);
        "before" rectifies the color-tuned channels first and applies
        the signed gains to them (feature-level suppression can go
        below zero)."""
        d_proj = (cphi[:, None, None] * P[..., 0]
                  + sphi[:, None, None] * P[..., 1])
        win_ray = torch.sigmoid(self.k * (self.r0 - radii))
        if rect == "before":
            drive = self.a * torch.relu(P[..., 0]) - self.b * torch.relu(d_proj)
        else:
            drive = torch.relu(self.a * P[..., 0] - self.b * d_proj)
        stim = (drive * win_ray).sum(-1) + self.g_form * FORM
        win_item = torch.sigmoid(self.k * (self.r0 - dist))
        return stim + win_item * (self.beta_T * hT + self.beta_D * hD
                                  + self.g_I * visited.float())

    def named_values(self):
        return {n: round(v, 4) for n, v in dict(
            a=self.a.item(), b=self.b.item(),
            g_form=self.g_form.item(), k=self.k.item(), r0=self.r0.item(),
            beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
            eta_T=self.eta_T.item(), eta_D=self.eta_D.item(),
            g_I=self.g_I.item(), sigma=self.sigma.item()).items()}

    def load_values(self, w):
        import numpy as np
        with torch.no_grad():
            for n in ("a", "b", "g_form", "r0",
                      "beta_T", "beta_D", "g_I"):
                getattr(self, n).copy_(torch.tensor(float(w[n])))
            self.raw_k.copy_(torch.tensor(float(np.log(np.expm1(w["k"])))))
            self.raw_eta_T.copy_(torch.logit(torch.tensor(float(w["eta_T"]))))
            self.raw_eta_D.copy_(torch.logit(torch.tensor(float(w["eta_D"]))))
        return self


def color_angles(sacc):
    """Per-saccade (cos, sin) of the distractor color's angle relative
    to the target color's direction in opponency space."""
    import numpy as np
    from build_contexts import template_axis
    cphi = np.zeros(len(sacc))
    sphi = np.zeros(len(sacc))
    pairs = sacc[["targCol", "singCol"]].drop_duplicates()
    for _, r in pairs.iterrows():
        if r.singCol == "none":
            continue
        uT = template_axis(r.targCol)
        uS = template_axis(r.singCol)
        m = (sacc.targCol.values == r.targCol) & (sacc.singCol.values == r.singCol)
        cphi[m] = uT[0] * uS[0] + uT[1] * uS[1]
        sphi[m] = -uT[1] * uS[0] + uT[0] * uS[1]
    return torch.tensor(cphi, dtype=torch.float32), \
        torch.tensor(sphi, dtype=torch.float32)


def load_final(path="weights_final.json"):
    return SearchModel().load_values(json.load(open(path)))
