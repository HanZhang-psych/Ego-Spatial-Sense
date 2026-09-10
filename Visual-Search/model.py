"""Search instantiation of the signed priority field (v1).

Same structured network as the agent (model/goal_es2.py), degenerate
where the task pins an input (docs/priority_field_visual_search_model.md):

  F_i = env(d_i) * (1 + g_T*isT_i + g_S*isS_i) + beta_T*hT_i + beta_D*hD_i
  P(saccade -> i) = softmax(F)   over the current choice set

- env(d) = sigmoid(k * (1 - d)): the agent's per-ray sigmoid envelope
  with steepness k TIED across directions (weight sharing); d is the
  item's distance from the current fixation, normalized by the display
  ring diameter. The item-presence gain is fixed at 1 (unit of F, with
  softmax temperature fixed at 1).
- g_T, g_S: static top-down channel gains (task set: attend template,
  willfully ignore the singleton). Scalars because the task evaluates
  the gain blocks at a single input point each.
- hT, hD: presence-driven leaky location traces (runtime state), rates
  eta_T, eta_D learned; expression weights beta_T, beta_D signed.
  Updated every trial: h <- (1-eta)*h + eta*e (e marks where a target /
  singleton appeared; distractor e = 0 on singleton-absent trials).
- g_I: optional inhibition-of-return penalty on already-visited items
  (one weight, frozen at 0 in the base model) — added as a model
  comparison after the pre-registered refixation diagnostic failed
  (observed 1.2% revisits vs. 6.3% predicted without it).
- No latency terms, no lapse, no per-subject parameters.
"""

import torch
import torch.nn as nn

NLOC = 6


class SearchEs2Model(nn.Module):
    def __init__(self, combine="mul"):
        """combine: how the envelope enters the field (the master
        equation's estimable combination-rule fork). "mul" (default,
        agent-inherited): env(d) scales the stimulus drive. "add": a
        linear distance penalty -k*d added to every item's utility,
        independent of item identity (minimal additive form; amplitude
        and shape folded into the one slope)."""
        super().__init__()
        self.combine = combine
        self.raw_k = nn.Parameter(torch.tensor(1.0))       # envelope steepness (softplus)
        self.g_T = nn.Parameter(torch.tensor(1.0))         # template gain
        self.g_S = nn.Parameter(torch.tensor(0.0))         # salience/rejection gain
        self.beta_T = nn.Parameter(torch.tensor(0.5))      # target-trace weight
        self.beta_D = nn.Parameter(torch.tensor(-0.1))     # distractor-trace weight
        self.raw_eta_T = nn.Parameter(torch.tensor(0.0))   # trace rates (sigmoid)
        self.raw_eta_D = nn.Parameter(torch.tensor(0.0))
        self.g_I = nn.Parameter(torch.tensor(0.0))         # IoR penalty (0 = off)

    @property
    def k(self):
        return nn.functional.softplus(self.raw_k)

    @property
    def eta_T(self):
        return torch.sigmoid(self.raw_eta_T)

    @property
    def eta_D(self):
        return torch.sigmoid(self.raw_eta_D)

    def envelope(self, d):
        return torch.sigmoid(self.k * (1.0 - d))

    def compute_traces(self, eT, eD, present):
        """Scan the trial sequence once; returns traces AS OF each trial.

        eT, eD: [S, T, NLOC] one-hot event maps (target / singleton
        location per trial); present: [S, T] singleton presence.
        Trace used on trial t reflects trials < t only.
        """
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

    def field(self, d, isT, isS, hT, hD, visited=None):
        stim = 1.0 + self.g_T * isT + self.g_S * isS
        if self.combine == "add":
            F = stim - self.k * d + self.beta_T * hT + self.beta_D * hD
        else:
            F = self.envelope(d) * stim + self.beta_T * hT + self.beta_D * hD
        if visited is not None:
            F = F + self.g_I * visited.float()
        return F

    def log_prob(self, d, isT, isS, hT, hD, valid, choice, visited=None):
        """valid: [N, NLOC] choice-set mask (setsize + fixated-item
        exclusion); choice: [N] index of the landed item."""
        F = self.field(d, isT, isS, hT, hD, visited)
        F = F.masked_fill(~valid, -1e9)
        return torch.log_softmax(F, dim=1).gather(1, choice[:, None]).squeeze(1)

    def named_values(self):
        return dict(k=self.k.item(), g_T=self.g_T.item(), g_S=self.g_S.item(),
                    beta_T=self.beta_T.item(), beta_D=self.beta_D.item(),
                    eta_T=self.eta_T.item(), eta_D=self.eta_D.item(),
                    g_I=self.g_I.item())
