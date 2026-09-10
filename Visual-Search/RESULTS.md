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

## The combination-rule fork (⊗): multiplicative vs. additive

The master equation leaves the envelope's entry rule estimable. Both
8-weight variants were fit on the same split:

| Variant | train NLL | test NLL | distance parameter |
| --- | --- | --- | --- |
| multiplicative env(d) x stim | 1.2722 | 1.25647 | k = 0.41 (near-flat) |
| additive −k·d penalty | 1.2672 | 1.25630 | k = 2.96 (strong slope) |

Held-out performance is a statistical tie (Δ ≈ 7 total NLL over 42k
test saccades) — **this design barely discriminates the combination
rule** — but the parameter stories differ sharply: the additive form
recovers a substantial uniform proximity preference (adjacent vs.
opposite item ≈ 1.5 utility units, comparable to the target gain),
which the multiplicative form cannot express without also crushing
far-target choices, and therefore flattens away (k ≈ 0.4). The
additive form's in-sample advantage (~875 NLL) does not survive
transfer to held-out subjects, so neither rule wins on prediction.

Consequences: (1) the earlier "wide attention window" reading is
rule-conditional — flat under the agent-inherited multiplicative
entry, a real distance cost under additive entry; (2) history and IoR
conclusions are rule-invariant (beta, eta, g_I essentially identical
across the two variants); (3) g_T is not comparable across rules
(multiplicative g_T multiplies through the envelope). Deciding ⊗
needs geometry these iso-eccentric rings do not provide (larger or
non-ring displays, or per-channel falloffs).

## Caveats on record

- **The envelope's width is combination-rule-conditional** (see the ⊗
  section above); statements about the attention window from these
  fits must name the rule they assume.
- Single split seed; no bootstrap intervals yet.
- Parameter recovery on synthetic data not yet run.
