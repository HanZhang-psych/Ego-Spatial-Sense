"""The final search model - first fixations only, one construction.

The model builds a PRE-WINDOW priority map over the display -

  map(x) = g_T * relu(target-color contrast at x)
         - g_D * relu(distractor-color contrast at x)
         + g_F * shape match at x
         + beta_T * h_T field(x) + beta_D * h_D field(x)

(the history fields are each item's history value smoothed by a
fixed spatial kernel, PAINT_SIG in build_contexts.py - a stated
model assumption) - multiplies it by the ego-anchored attention
window sigmoid(k*(r0 - d)), and reads it out as the average over
each item's sector (a pie slice of the display). Softmax over the
sector averages predicts the first saccade.

Implementation: the readout is precomputed as sector x distance-bin
area averages (build_contexts.py), so

  F_i = sum over bins d of  sigmoid(k*(r0 - radii_d)) *
        [ g_T*relu(P_T[i,d]) - g_D*relu(P_dist[i,d]) + g_F*FP[i,d]
          + beta_T*HTP[i,d] + beta_D*HDP[i,d] ]

which equals the sector average of window x map, discretized over
the distance bins. HTP/HDP = HM @ h with HM the precomputed binned
history-kernel geometry.

Notation: g_* are the stimulus gains (g_T target-color enhancement,
g_D distractor-color suppression, g_F shape/form), beta_* the
history gains, eta_* the memory speeds, k/r0 the attention window.

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

    def field(self, P, FP, HTP, HDP, radii, cphi, sphi):
        """Window x (salience + history), sector-averaged, per item.

        P [N,6,NBINS,3]: binned color contributions (target axis,
        orthogonal, presence); FP [N,6,NBINS]: binned shape map;
        HTP/HDP [N,6,NBINS]: binned history fields (HM @ h).
        cphi/sphi: per-observation rotation of the distractor color
        axis relative to the target axis (0 if singleton absent).
        Channels are rectified before their gains: g_T = pure
        enhancement, g_D = pure suppression."""
        d_proj = (cphi[:, None, None] * P[..., 0]
                  + sphi[:, None, None] * P[..., 1])
        window = torch.sigmoid(self.k * (self.r0 - radii))
        pre_window = (self.g_T * torch.relu(P[..., 0])
                      - self.g_D * torch.relu(d_proj)
                      + self.g_F * FP
                      + self.beta_T * HTP + self.beta_D * HDP)
        return (pre_window * window).sum(-1)

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
