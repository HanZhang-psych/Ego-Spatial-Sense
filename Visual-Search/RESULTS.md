# Pooled fit results (v1, run 2026-09-10)

Data: 217,595 saccades (indices 1–5; 124,834 first saccades), 333
subjects, 11 studies; color-singleton present vs. absent trials only
(onset/motion-distractor trials excluded entirely); practice conditions
the traces but is not scored. Evaluation: subject-level 80/20 split
(267 train / 66 held-out subjects, seed 0) — held-out numbers are on
people the model never saw. `results_fit.json`.

## Nested model comparison (held-out NLL per saccade)

| Model | free weights | train | **test** | Δtest total (vs prev) |
| --- | --- | --- | --- | --- |
| null (gains + envelope only) | 3 | 1.3739 | 1.3665 | — |
| + location traces | 7 | 1.2892 | **1.2737** | **3,995** (4 wts) |
| + IoR (visited-item penalty) | 8 | 1.2722 | **1.2565** | **743** (1 wt) |

Both additions earn their keep on held-out subjects, not just in-sample.

## Estimates (traces + IoR model)

| Weight | Estimate | Reading |
| --- | --- | --- |
| g_T | **+3.01** | template gain: attend the target's features |
| g_S | **−1.81** | salience gain: singleton written *below* a plain nontarget — suppression as negative writing |
| beta_T / eta_T | **+1.87 / 0.63** | strong, fast-turnover attraction to prior target locations |
| beta_D / eta_D | **−0.27 / 0.17** | weaker, ~4x slower suppressive trace on prior singleton locations |
| g_I | **−2.01** | already-visited items are strongly penalized |
| k | 0.41 | envelope falloff shallow (see caveat) |

Estimates are stable across the with/without-IoR variants (g_T 3.18 vs
3.01, g_S −1.81 both, β/η essentially unchanged) — the IoR weight
absorbs refixation structure without disturbing the suppression or
history story.

## Model vs. observed, held-out subjects

First saccades, singleton present:

| Destination | Observed | Model (traces+IoR) |
| --- | --- | --- |
| target | 40.3% | 51.8% |
| singleton | 7.0% | 4.4% |
| nonsingleton (per item) | 13.6% | 11.4% |

Suppression-below-baseline ordering reproduced (source paper: 42.0 /
7.9 / 14.2 under slightly different exclusions); the model over-guides
— one g_T shared across saccade indices compromises between first
saccades and the better-guided later ones.

Refixations (saccades 2+, held-out): observed **1.15%**; no-IoR model
predicted 6.15%; with the single g_I weight the model predicts
**1.23%** — the diagnostic that motivated the term is now matched.

## Caveats on record

- **k ≈ 0.4 (shallow envelope).** Under the v1 separability assumption
  (one multiplicative envelope shared by all channels), the likelihood
  prefers a weak distance falloff. Either proximity matters little for
  between-item choice on these dense rings, or the shared-envelope form
  is strained (far targets are chosen anyway, dragging k down).
  Per-channel falloffs are the discriminating extension.
- Single split seed; no bootstrap intervals yet.
- Parameter recovery on synthetic data not yet run.
