# Pooled fit results (run 2026-09-10)

> **Consolidation note (2026-09-10):** the scripts for superseded
> model generations (v1 role-flag fits, goal-late v2, the
> window-form/history-ordering/sigma comparison runs) were removed;
> the current pipeline is `build_contexts.py` → `fit.py` →
> `reproduce.py` with the final model in `model.py`. Every number
> below stands as the record of those comparisons; the git history
> holds the original scripts. Sections below name the scripts that
> produced them at the time.
>
> **Pooling revision (2026-09-10):** practice trials are now excluded
> altogether (previously they conditioned the traces without being
> scored) and the excluded other-distractor-type trials (onsets,
> singleton-onsets, motion singletons) are reported at pooling time.
> Effect on results: none to three decimals (held-out NLL 1.2335 vs
> 1.2339; all reproduction numbers unchanged) — the fast trace rates
> make the cold start at each subject's first experimental trial
> immaterial.

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

## Parameter recovery (recover.py; results_recovery.json)

The fitted traces+IoR weights generated a synthetic pool (real 333
subjects' actual trial sequences driving the traces; generative
scanpaths per trial on the iso-eccentric ring; 273,406 synthetic
saccades), refit from scratch with the same pipeline:

| Weight | generating | recovered | rel. err |
| --- | --- | --- | --- |
| g_T | 3.010 | 2.944 | 2.2% |
| g_S | −1.808 | −1.788 | 1.1% |
| beta_T | 1.868 | 1.862 | 0.3% |
| beta_D | −0.270 | −0.241 | 10.6% |
| eta_T | 0.627 | 0.630 | 0.4% |
| eta_D | 0.173 | 0.181 | 4.7% |
| g_I | −2.005 | −1.973 | 1.6% |
| k | 0.411 | 0.528 | 28.6% |

Every substantive weight comes back within a few percent, signs and
ordering exact. The two larger relative errors are the two weakly
identified quantities, as expected: k (a near-flat envelope leaves
little curvature to estimate) and beta_D (the smallest weight). The
estimation machinery — including learning both eta's through the
trace recursion — is validated.

Note: the synthetic pool shows a higher raw refixation opportunity
(6.3% observed = 6.6% predicted internally) than the real data (1.15%)
because simulated searches always run to the target or 5 saccades,
while real trials are censored by the response deadline and the end of
the fixation report — a design difference, not a recovery failure.

## Generated predictions: high-probability distractor location
(v1-era prediction run, script since superseded by
reproduce_wang_theeuwes.py on the final model — see that section below;
simulated from the fitted weights, no new parameters)

Design: singleton present 70%; when present, at one HP location 65%
(LP locations 7% each); 400 biased + 200 unbiased trials.

1. **Location-specific suppression**: first saccades to the singleton
   at the HP location 3.2% vs. 4.6% at LP locations (~30% relative
   reduction) at asymptote.
2. **Source-blind spillover (architectural signature)**: targets
   appearing at the HP location are selected on 43.1% of first
   saccades vs. 50.7% elsewhere — a 7.5-point target cost at the
   suppressed location. The single signed field *forces* this
   prediction: the trace suppresses a location, and the readout cannot
   see who wrote the negative value. The human paradigm's hallmark
   finding (impaired target processing at the suppressed location)
   falls out with no added assumptions.
3. **Build-up is fast**: at eta_D = 0.17 per exposure, suppression
   reaches ~asymptote within ~6 HP-singleton appearances (~13 trials
   at these rates) — already at ceiling in the first 25-trial bin.
4. **Extinction is equally fast — a falsifiable discrepancy**: the
   former-HP location returns to baseline within ~25 unbiased trials.
   Human reports of much longer-lived statistical-learning effects
   would contradict the single fast trace as fitted (eta_D estimated
   mostly from short-range intertrial structure in roving designs) and
   would argue for a second, slower trace timescale — a concrete,
   pre-registrable model comparison for future data.

## v2: pixel front-end (no shortcut) — results_fit_v2.json

Displays reconstructed per trial (ring geometry; all items in the
target color, singleton in the opposite color per the data's
targCol/singCol; template shape at the target), rendered to pixels,
passed through the Itti & Koch front-end, sampled as ray scans from
the current fixation, wedge-integrated into per-item channel evidence
(526 unique displays, 3,062 unique display-fixation contexts). Same
split, 10 weights.

| Model | held-out NLL/saccade |
| --- | --- |
| v2 pixels, no traces | 1.4371 |
| **v2 pixels, full** | **1.3288** |
| v1 flags, traces+IoR (reference) | 1.2565 |

- **The cost of honest perception is ~0.07 NLL/saccade.** Role flags
  are a noiseless ceiling; sensor-derived evidence carries rendering
  and reconstruction noise. The v2 model still beats v1's no-trace
  null (1.3665) using no flags at all.
- **History and IoR are downstream of perception, as claimed**: their
  weights survive the front-end swap essentially unchanged (beta_T
  1.71 vs 1.87; beta_D -0.24 vs -0.27; eta_T 0.643 vs 0.627; eta_D
  0.196 vs 0.173; g_I -2.39 vs -2.01).
- **Suppression-mechanism comparison (nested, held-out):** full model
  (g_simS and w_sal both free) 1.32879; color-rejection only 1.32941;
  salience only 1.32978 — differences of ~27–43 total NLL, weak
  evidence. More telling: in the single-mechanism fits the dedicated
  suppression weight collapses toward zero (g_simS −0.05 alone;
  w_sal −0.01 alone) while the *template-color enhancement* g_simT
  rises to absorb the effect (0.25 → 0.41/0.44). In two-color
  displays every nontarget shares the target's color, so SIM_T and
  SIM_S are complementary — "boost green" and "penalize red" differ
  only by a constant the softmax ignores, and salience marks the same
  single odd item. Conclusion: (1) below-baseline oculomotor
  suppression in these displays is carried almost entirely by
  **template-color enhancement relegating the mismatching singleton**,
  with dedicated suppression weights adding only marginal held-out
  gain; (2) enhancement-vs-suppression and rejection-vs-salience are
  **structurally unidentifiable in two-color displays** — separating
  them requires ≥3 colors (heterogeneous nontarget colors), a concrete
  design prescription. v1's single g_S was the flag-level projection
  of this confounded bundle.
- k lands large (~9.8): in v2 the envelope acts on the sensor's range
  reading, effectively gating wedges by whether/where they contain
  energy — not comparable to v1's k on geometric distance.

## v2.1: single goal-modified salience map (goal-early) — results_fit_v21.json

Architecture revision (user decision): no task-blind salience channel;
the task set enters the feature channels *before* the contrast stage.
Fixed opponency axes are rotated into template-referenced coordinates
(the goal supplies the direction of the target color; the gain
supplies only strength), the gain-weighted contrast is RECTIFIED, and
the result is one goal-modified map. Feature-level suppression can
therefore only attenuate/relegate — negative writing is reserved for
the spatial sources (traces, IoR).

| Model (same split, 10 weights each) | held-out NLL/saccade |
| --- | --- |
| v2 goal-late (linear gains + salience channel) | 1.3288 |
| v1 role flags, traces+IoR | 1.2565 |
| **v2.1 goal-early (rectified, no salience channel)** | **1.2414** |

- **Goal-early wins decisively over goal-late** (~3,800 total held-out
  NLL at equal weight count), and even beats the noiseless role-flag
  model (~650): the graded, geometry-carrying sensory profiles plus
  the built-in near-weighting express structure the flags cannot.
- **Sanity check passed**: the orthogonal-axis gain idles at ~0
  (g_O = −0.005); the template direction alone carries the color work.
- **Third front-end, same history/IoR weights**: beta_T 1.87,
  beta_D −0.27, eta_T 0.630, eta_D 0.193, g_I −2.15 — the selection-
  history and IoR layer is invariant to every perception swap tried.
- **The attention window, estimated in its sensor-native place.**
  The initial v2.1 hand-set the sensor's radial falloff (exp(−2r))
  and carried a separate fitted envelope, which collapsed to ~0 — the
  hidden hand-set window was doing its job. Revision: the falloff
  decay is now the fitted window parameter and the redundant envelope
  is retired. Estimated k = 2.15 (held-out 1.2408, a further small
  gain): at the ring geometry this weights an adjacent item ~3x an
  opposite-side item — a genuinely graded functional viewing field,
  overturning the earlier "flat window" reading, which was an
  artifact of estimating the window in the wrong place (multiplying
  item utilities) under the wrong combination rule. The hand-set
  constant (2.0) happened to be near-optimal, which is why freeing it
  changed little else (history/IoR weights again unchanged: beta_T
  1.87, beta_D -0.26, eta_T 0.630, eta_D 0.201, g_I -2.15).

## Window-form comparison (results_window_sigmoid/free.json)

Three shapes for the sensor-readout attention window, same split:

| Form (free params) | held-out NLL | fitted shape |
| --- | --- | --- |
| exponential decay (1) | 1.24082 | k = 2.15 |
| **sigmoid, ES2's form (2)** | **1.24048** | r0 ≈ 0, k = 3.21 |
| nonparametric 24-bin profile (24) | 1.24128 | jagged; overfits |

- The **sigmoid (ES2's own functional form) fits best**, making the
  window correspondence with the agent literal; adopted as canonical.
- Its reach parameter collapses to r0 ≈ 0: no plateau — the window
  declines from fixation immediately (acuity-like falloff, not a
  spotlight with a rim). The "zoom-lens" reach is simply not exercised
  at these display scales.
- The **free profile loses out-of-sample despite 23 extra weights**:
  its bins align with item chord distances and become covert
  item-selectors — overfitting display geometry. This settles the
  "net vs. one k" question empirically: given the same freedom the
  agent's 360-k window has, held-out prediction on these displays
  *chooses* the low-parameter form. The restriction is selected by
  the data, not imposed.
- History/IoR weights unchanged across all three (sixth front-end or
  window variation with beta_T 1.87, beta_D -0.26, eta_T 0.63,
  eta_D 0.20, g_I -2.15).

## History ordering: inside vs outside the window
(results_history_order.json)

Does memory (traces + IoR) enter the field at full strength regardless
of eccentricity (outside the window), or is its expression gated by
the same attention window as the stimulus (inside)? Equal-weight fits,
r0 free in both:

| Ordering | held-out NLL | held-out, saccades 2+ |
| --- | --- | --- |
| history outside (prior commitment) | 1.2407 | 1.0050 |
| **history inside (gated)** | **1.2283** | **0.9875** |

Δ ≈ 540 total held-out NLL — decisive. **The architectural commitment
is revised: everything is read through the window.** The final field
is F = window(d) * [goal-modified salience + beta*traces + g_I*visited],
i.e., one priority map assembled from all three sources and then
window-gated — which is both what the data prefer and the cleaner
statement (a single gate over the whole map). Notes:

- With history inside, the window recovers a genuine reach:
  r0 = 0.49 (half-height at the ring radius), k = 3.4 — a plateau it
  did not show when forced to serve the stimulus alone. History
  amplitudes rescale accordingly (beta_T 4.45, beta_D -0.53,
  g_I -6.30 raw; effective strengths at item distances comparable to
  before).
- Anticipation survives: pre-onset the window is centered at fixation
  and covers the display gradedly, so the prior is attenuated, not
  abolished; on first saccades from center the orderings are
  equivalent up to a constant.
- This also resolves the search/agent asymmetry in the agent's favor:
  the agent's runtime trace was always injected through its
  distance-sensitive goal-gain machinery. The two instantiations now
  agree - memory expresses through the same windowed readout as
  perception.

## Reproduction battery vs the source paper (reproduce_gaspelin.py)

Final model (v2.1, history-inside), held-out subjects, no refitting.
Paper analyses out of scope by design: manual RT, latency quantiles,
continuous gaze, practice-block capture.

1. **Oculomotor suppression** (paper: 42.0 / 7.9 / 14.2): observed
   40.3 / 7.0 / 13.6; model 51.0 / 6.3 / 11.0. Ordering and
   below-baseline suppression reproduced; known miscalibration: the
   model over-guides the target (~+11 pts).
2. **Suppression across saccades 1–5** (paper: positive suppression at
   every index): observed effect −6.6 to −10.6 points, persisting
   strongly; model reproduces indices 1–3 (−4.8, −4.4, −2.8) but
   under-persists at 4–5 (−1.1, +0.9; n = 1164/386). A named residual:
   late-saccade suppression in humans outlives what window-gated color
   relegation plus IoR produce in the model.
3. **Intertrial location priming** (paper: target 73.3 repeat vs 36.6
   change; singleton 4.6 vs 10.1): observed 71.1/35.2 and 2.9/7.6 —
   the pooled subset closely matches the paper; model 82.1/44.1 and
   3.7/6.7 — both signatures reproduced (target doubling; singleton
   halving on repeat), with the target effect overshot.

Verdict: every in-scope qualitative signature reproduces from one
fitted parameter set; the three quantitative residuals (target
over-guidance, late-saccade suppression persistence, priming
overshoot) are specific and named, and all point at the same missing
ingredient family: saccade-index-dependent guidance (a gain that
grows as search proceeds) and/or the scoped-out motor-repetition term.

## Reproduction: Wang & Theeuwes distractor-location probability cueing
(reproduce_wang_theeuwes.py; final model, no refitting)

HP-location design (singleton 65% at one location when present),
first saccades, learned regime:

1. **Less capture at the HP location**: 4.73% vs 6.65% at LP (~29%
   relative reduction) — reproduced.
2. **Impaired target selection at the HP location**: 46.0% vs 50.1%
   elsewhere — reproduced, from the same trace value (source-blind
   spillover).
(A third item here formerly claimed the model's flat spatial
gradient around the HP location as a committed divergence from
W&T's observed spillover to neighbors; removed with the W&T battery
- see "The Wang & Theeuwes battery is a transplant" below. The
slot-indexed traces' lack of spatial spread stands on its own as an
architectural fact.)

Framing caveat: W&T ran a capture regime (singleton-detection mode);
our weights come from suppression paradigms, so absolute capture
levels are not comparable.

## Trace spread sigma_h (fit_sigma.py; results_fit_sigma.json)

Adding a mass-preserving ring-Gaussian spread to the trace updates
(sigma -> 0 nests the slot-exact final model): the free fit collapses
to sigma = 0.13 slots (nothing reaches even the nearest neighbor),
held-out NLL identical to the null (1.22797 both; r0 refit jointly to
0.55, marginally improving on the earlier 1.22833). Conclusion: **no
measurable trace spread in roving-location designs** — neighbor
repeats are rare and unstructured, so the kernel has no variance to
bite on. sigma_h is measurable only where location statistics
concentrate mass - biased designs - which is where the agent-side
spread trace earned its keep. A biased-design human fit is the
experiment that would estimate it.

## Reconstruction correction: feature-search displays

The initial reconstruction rendered the target diamond among identical
circles - accidentally a shape-singleton display, which the source
studies' inclusion criteria specifically avoid. Corrected: nontarget
shapes are heterogeneous (circle/square/triangle/cross), matching the
feature-search paradigm. Refit on rebuilt contexts: every
psychological parameter unchanged to two decimals (beta_T 1.87,
beta_D -0.26, eta_T 0.63, eta_D 0.20, g_I -2.10); held-out NLL 1.2471
(vs 1.2408 before) - shape enters the model only through the analytic
form channel, so the fit is invariant as the architecture predicts.
Parameter audit: set size (6 per study; Hamblin 4), colors (including
Stilwell's within-study singleton-salience colors and Gaspelin 2019's
four counterbalanced pairs), and item distances are all taken
per-trial from the data files; shapes, item size, and background are
paper-sourced reconstruction assumptions (the display-parameter
details are not in the OSF trial files or summary workbook).

## Unsigned color-oddity salience w_S (adopted into the final model)

Motivated by a structural critique (Han, 2026-09-10): the rectified
goal-weighted form could not express direction-blind color salience -
"this item's color differs from its neighbors, whichever direction" -
so bottom-up color capture had no dedicated route (the presence map
covers intensity oddity only). Added w_S * |off-goal color contrast|
as a nested term (w_S = 0 recovers the prior model).

Result: **w_S = -0.32**, held-out NLL 1.2335 -> 1.2317 (~73 total for
one weight). Reading: beyond the goal's up/down color weighting, odd
colors are *actively avoided* - a signal-suppression-style penalty on
salience itself, now carried inside the goal-early architecture.
Adopted into the final model (fit.py fits it by default; the nested
test is --variant no_salience). Side effect: the late-saccade
suppression residual improves markedly (model effect at indices 1-4:
-5.7/-5.0/-3.4/-1.5, was -4.0/-3.7/-2.2/-0.6) - oddity suppression,
unlike goal weighting, keeps acting at every fixation. Note: in
green/red studies the off-goal magnitude loads mainly on the
singleton, so w_S there acts as singleton-specific suppression;
cross-color studies give it generality. The yoked up/down goal gain
and the >=3-color prescription are discussed in the two-color
identifiability section above.

## Presence-channel correction (semantic, not predictive)

Rendering each model term from actual displays exposed a defect: the
palette's colors are equiluminant with the gray background, so the
intensity-based presence channel fired on almost nothing (mostly an
image-border artifact). Corrected to contrast of color deviation from
background ("any visible object"). Effects after rebuild + refit:
held-out NLL essentially unchanged (1.2325 vs 1.2317); w_p collapses
toward 0 - with a real signal to weight, object presence per se adds
almost nothing once the color terms exist (its earlier weight partly
leveraged the artifact). Second note: within any single color-pair
study the off-goal axis is one-signed, so g_O and w_S partially trade
(the pooled fit can land on offsetting values, e.g. +0.46/-0.48);
their per-study NET is the identified quantity, and separating them
cleanly needs cross-color designs.

## The minimal module (final form; run 2026-09-10)

A minimalization pass (driven by Han) settled the perceptual module at
two color terms plus shape:

  F = window x relu( a*(target-color contrast) - b*(distractor-color
      contrast) ) + g_form*shape + windowed( beta_T*h_T + beta_D*h_D
      + g_I*visited )

Ten weights. Decisions and their held-out prices (test NLL; noise
floor ~7 total):

- **(a, b) reparameterization** of the goal/off-goal gains: exactly
  equivalent (two parameters of the same linear family, different
  basis); adopted for interpretability - a = enhance target color,
  b = suppress distractor color. Within a two-color study the pair is
  nearly yoked; >=3-color displays separate them.
- **Presence dropped** (-24 total): near-equal across items, hence
  softmax-invisible; its channel had also been intensity-based and
  artifact-prone before correction.
- **w_S (odd-color salience) dropped** by design decision for
  minimality (it had earned ~73 and part of the late-saccade
  improvement; within a study it duplicates the b term). The
  bottom-up-salience route is thereby out of the final model; the
  novel-singleton-color transfer design is the experiment that would
  force it back.
- **Shape kept** (dropping it: test NLL 1.5153 vs 1.2333 - ~11,700
  total, the largest effect in this ledger): shape is the target
  template's only carrier; without it the model cannot prefer the
  target over same-colored items.
- **Rectification placement fork - a tie (~10 total)**: relu after
  the goal weighting (suppression saturates at zero; relegation) vs
  relu on the tuned channels before signed gains (feature suppression
  can go below zero) fit identically; rectify-after kept as
  convention, the fork recorded as empirically open here.
- **b ~ 0 in the fitted model** (0.02): with two-color displays,
  a*(target color) alone relegates the distractor (it scores negative
  on the target axis and the relu clips it) - enhancement carries
  suppression, now visible at the parameter level.

Final ten-weight model: held-out NLL 1.23328 (vs 1.23249 for the
twelve-weight version - the whole simplification cost ~33 total);
suppression, priming, and W&T reproductions intact; notebook trains
the identical form.

## Window-gating of the shape term: tested, rejected, with a caveat

Han's unified notation - window x [color + shape + memory] - exposed
that the shape term entered ungated. Gating it was tested: held-out
NLL 1.25597 vs 1.23328 ungated (~950 total, decisive; the fit even
flattened the window trying to rescue it). Shape stays OUTSIDE the
window:

  F = window x [ relu(a*targetColor - b*distractorColor)
                 + beta_T*h_T + beta_D*h_D + g_I*visited ] + g_form*shape

Architecturally this converges with the agent, which never had one
shared window (the obstacle channel runs through the per-ray
envelope; the goal channel has its OWN learned distance gain): in
both instantiations the seek/template channel's distance profile is
separate from the scan channel's, and effectively far-reaching.

**Caveat (Han):** FORM is the analytic label channel - it marks the
template-shape item using task knowledge, not pixels - and a label
trivially ignores eccentricity. The ~950 result therefore conflates
"template guidance is window-free" (interesting) with "our shape
channel is a label" (artifact). The clean test is a pixel-derived
shape channel (template matching with the known shape kernel: still
top-down in WHAT it looks for, bottom-up in the evidence), then
re-running this fork. Until then, the safe claim is: template
evidence, however delivered, fits best un-gated in these data.

## From label to pixels: the shape channel made honest (and where that ends)

Han asked the pointed question: "Is the 'shape' channel simply
encoding the correct target location?" It was - FORM marked the
template item analytically, using task knowledge, not the display.
Rebuilding it from pixels took three moves:

**1. Canonical displays (Han's call).** In every pooled study the
subject's target shape and the color scheme were fixed for the whole
session, so only the match/mismatch structure matters, not the
specific assignment. All displays are therefore reconstructed as:
target = circle, nontargets drawn from {square, triangle, cross,
diamond}, all items green, singleton red. This collapses 526 unique
displays to 52 (36 setsize-6 + 16 setsize-4) and 3,062 contexts to
232. Shape audit backing this (from the papers; the trial files
record no shape columns, so per-subject assignments are unknowable):
Gaspelin 2017 counterbalanced diamond/circle targets among
diamond/circle/square/hexagon; Adams 2023 / Adams & Gaspelin 2024
fixed a diamond target among hexagons and triangles; Stilwell
counterbalanced circle/diamond; Hamblin-Frohman 2022 used a diamond
target. Known cost: Stilwell's singleton-salience color manipulation
becomes invisible to the model.

**2. A pixel-derived matcher.** The display is rendered (256px, 2x
for matching), the background-deviation map binarized (shape, not
color amplitude - unbinarized, the red singleton's larger deviation
leaked color into "shape"), and normalized cross-correlation with
the circle kernel computed by FFT. Plain NCC is nearly blind at this
item scale: filled same-radius shapes share ~95% of their area, so
circle scored 0.822 vs square 0.809 - a margin that standardization
turns to noise. Fit with that channel, held-out NLL was 1.4627 and
the model's target-fixation rate collapsed to 26.6% (observed
40.3%). Resolution does not fix this (1x/2x/4x renders: margin
+1.6/+2.1/+2.3%); the overlap is geometric. A contour-based matcher
was worse (the cross's edges beat the circle's). What fixes it is a
discriminative readout - score = circle NCC minus the best competing
shape's NCC, i.e. template matching with competitive normalization
among shape detectors. Target then scores ~6x the best nontarget.

**3. The refit.** Held-out NLL 1.23764, against 1.23328 for the
analytic label - 99.98% of the gap recovered; fitted weights nearly
unchanged (g_form 0.62); suppression/priming reproductions intact
(target rate 51.5 vs label-era 52.5). And the gating fork, rerun as
promised in the caveat above: with the weak matcher, gating had
FLIPPED (gated 1.4487 beat ungated 1.4627 - the window acting as
damage control on a noisy channel); with the sharp matcher it flips
back (gated 1.2595, decisively worse). Shape stays outside the
window, now on pixel-derived evidence.

**Where honesty ends (on record):** the datasets never log item
shapes, so the reconstruction places the circle at targLoc by
construction. In a noiseless canonical display, any sufficiently
good shape matcher must therefore converge on the analytic label -
1.2376 vs 1.2333 is the entire remaining daylight. The pixel channel
changes the pathway, not the information: guidance now flows through
a fixed, shape-general computation with measurable confusability,
rather than a location-indexed label. A shape channel carrying
genuinely LESS information than the label would need a principled
degradation (e.g. eccentricity-dependent acuity); the gated_shape
result is that test at window granularity, and the data reject it.

## Scope narrowed to first fixations (Han's decision)

The model of record now predicts the FIRST saccade of each trial
only, launched from central fixation. Two structural consequences:
the IoR term is removed (items can only be "already visited" from
the second saccade on - g_I was unidentifiable in scope), and EVERY
term, shape included, is gated by the attention window. From central
fixation all ring items are equidistant, so the window is flat
across items - a shared gain doing no selective work - and is kept
because it is theoretically defined (the ego-anchored window), not
because first-saccade data constrain it. This also dissolves the
gating fork: with one vantage point, gated and ungated shape are
reparameterizations of each other. Nine weights remain: a, b,
g_form, k, r0, beta_T, beta_D, eta_T, eta_D.

Fits (124,834 first saccades; same subject split):

- final: held-out NLL 1.38849 - better than the all-saccade model
  evaluated on first saccades (1.4091): the dedicated fit no longer
  compromises to serve later saccades.
- no_shape: 1.59033 (still the largest single ablation)
- no_traces: 1.52374
- no_color (a=b=0): 1.39468 - only ~155 total. But the small
  price is a lesson in what likelihood measures, not evidence the
  channel is idle (Han's challenge). Rerunning the battery with the
  no-color model: the suppression effect VANISHES - singleton 12.1%
  vs plain 12.0% (final model: 10.2 vs 12.4; observed: 7.0 vs 13.6)
  - leaving only the h_D location-priming component. Color's
  target-enhancement role is redundant with shape here (the circle
  is unique, so "find green circle" = "find circle"), but its
  relegation role - keeping the eyes off the red item, below the
  plain-item baseline - is color's alone. Singleton fixations are
  ~7% of first saccades, so abolishing suppression costs almost no
  likelihood while destroying the dataset's signature phenomenon.
  This also makes sense of the ridge basis the free fit chose
  (a = -0.116, b = +0.078): that combination is the one whose relu
  clips the RED item - the fit spends color where it has unique
  work (relegation) and leaves target-attraction to shape.
- rectify_first: 1.38594; no relu at all: 1.38547. Fork RESOLVED
  (Han's question "did we try no relu?"): held-out NLL is monotone
  in rectification strength (rectify-after 1.38849 > rectify-first
  1.38594 > linear 1.38547), and the linear field also reproduces
  suppression best - singleton 8.5% vs plain 12.7% (rectify-after:
  10.2 vs 12.4; observed: 7.0 vs 13.6) - because the signed -b term
  can push the red item BELOW baseline instead of saturating at
  zero. The relu is dropped from the model of record: the field is
  fully linear before the softmax. Interpretively this is a
  reversal: the data side with ACTIVE SUPPRESSION below baseline
  over relegation, at least at the first saccade. The fitted b also
  becomes cleanly positive again (0.185), ending the ridge
  weirdness: signed suppression is what the color channel is for.

Reproductions (held-out subjects): suppression present (model
41.8 / 10.2 / 12.4 vs observed 40.3 / 7.0 / 13.6 - the singleton
sits below plain items, though the margin is weaker than observed);
priming clean (target repeat/change 76.3/35.1 vs 71.2/35.2;
distractor-repeat drop reproduced); W&T HP-location capture
reduction intact (7.99% HP vs 10.39% LP).

**Color-term identifiability, sharpened.** The fitted (a, b) came
out sign-flipped (a = -0.116, b = +0.078). Diagnosis: the known
two-color ridge. Freezing b = 0 refits to a = +0.045 at held-out
1.38998 - within ~37 total of the free fit - so (a, b) is a nearly
flat direction and individual signs mean nothing here; only the
combined green-vs-red contrast is identified. The small a under
b = 0 also shows the color channel carries little at the first
saccade once shape and the traces are in; its full price is the
no-color ablation above.

## Is 1.385 good? Oracle benchmarks and the ceiling (Han's question)

Nonparametric memorization benchmarks on the same held-out first
saccades (n=24,843; Laplace-smoothed tables built from training
subjects unless noted):

  chance                                     1.754
  cross-person oracle: context table         1.563
  within-person oracle: own context table    1.539  (leave-one-out,
                                                     same subject)
  cross-person: context x tRep x sRep        1.525
  model of record                            1.385

The model beats every constructible memorization benchmark -
including tables that know the display perfectly, the discrete
priming conditions, or the individual person. The margin comes from
the graded trace history: h_T/h_D accumulate over many trials with
learned decays, and that continuous state predicts more than any
discrete conditioning can tabulate. Consequence: these oracles are
floors for good models here, not ceilings.

The true ceiling (conditional entropy of behavior) is therefore not
empirically bounded from above by any table. Bracketing it: behavior
is intrinsically high-entropy (even on target-repeat trials people
hit the target only ~71%), and the model's residual misses in the
battery are small distortions (suppression 8.5 vs 7.0, priming
slightly compressed), not absent phenomena. Estimated reachable
ceiling for a subject-held-out model: roughly 1.30-1.35, via
per-study gains, spatially spreading traces, or
richer history kernels; the remaining ~1.3 nats look like genuine
first-saccade stochasticity.

## Stilwell's salience-graded suppression: reproduced out-of-sample

Han's question: can the model reproduce Stilwell's finding that MORE
salient singletons receive GREATER suppression? The canonicalization
had made this manipulation invisible to the fit (all singletons
"red"). Test (`stilwell_salience.py`): apply weights_final,
unchanged, to Stilwell's displays rendered with their TRUE colors,
zero traces:

  high salience: observed  7.0%   model  8.3%
  low  salience: observed 11.3%   model 11.4%

Per-pair structure tracks too: blue targ/red sing 7.8 (obs 7.0);
blue/teal 11.2 (11.0); pink/blue 8.7 (7.6); red/pink - the least
salient pairing - 15.5 (obs 14.8), above everything else. Mechanism:
a near-target-color singleton projects partly onto the target axis
and weakly onto its own, escaping the signed -b suppression; a
chromatically distant one takes the full hit. Nothing was added or
refitted - the gradient falls out of the two-axis color module on
colors the fit never saw, buying back most of what canonicalization
discarded. Caveats: our pink/teal/blue RGBs are plausible guesses,
not Stilwell's calibrated coordinates; the small teal counterbalance
cells (n~270) carry a high/low label the model cannot distinguish
(same color pair) and are missed.

**Tightened (Han's catch):** the first version of this test was not
fully out-of-sample - Stilwell's low-salience trials sat in the
training pool (canonicalized to red/green), so their weaker
suppression could have leaked into the gains. Fix: pool_data.py now
EXCLUDES Stilwell low-salience singleton trials from the dataset
(the canonical salient-red reconstruction only matches the
high-salience displays anyway); 124,834 -> 114,232 first saccades,
refit test NLL 1.39892 (new dataset - not comparable to 1.38547),
weights essentially unchanged (b 0.196, g_form 0.76), all other
reproductions intact. The battery rerun with the clean weights:
high 8.2% (obs 7.0), low 11.5% (obs 11.3) - the low-salience
condition is now predicted by a model that never saw it in any
form, colors or choices. The earlier oracle/ceiling benchmarks and
ablation prices in this ledger refer to the pre-exclusion dataset.

## The "Wang & Theeuwes" battery is a transplant, not a simulation
(Han's challenge)

The wang() battery applies W&T's STATISTICAL manipulation - a 65%
predictable distractor location - to THIS task's displays and task
set, and tests only the location-learning traces, the one component
the paradigms plausibly share (traces live over slots, not
features). It is not a simulation of the additional-singleton task
itself, and the model as constituted cannot represent that task:
their target is defined by shape UNIQUENESS (a local-contrast /
heterogeneity computation, not a fixed template - g_form does not
apply in kind), and their capture is bottom-up salience - exactly
the w_S channel dropped for minimality. Modeling W&T's actual
paradigm would need the salience channel reinstated plus a
local-uniqueness shape-contrast channel; until then the claims this battery made were qualitative and
trace-borne: reduced capture at the HP location and a target cost
there. Decision
(Han): even the transplant is misleading - the wang() battery is
REMOVED from reproduce.py; the prior-evolution figure remains as an
illustration of the trace mechanism under a generic biased-location
sequence, with the W&T attribution dropped.

## Caveats on record

- **Window anchoring: display-relative vs fixed-size - untestable
  here, tested anyway (Han's question).** Distances are normalized by
  each study's own empirically measured ring radius (median landing
  positions; spread across studies 190-237 px, i.e. about +/-11%),
  which implicitly assumes the attention window scales with display
  eccentricity. Refitting with distances re-expressed in a common
  screen frame (each study scaled by its radius over the grand mean)
  - a fixed-size window - gives held-out NLL 1.23814 vs 1.23764
  ring-relative: a tie (~21 total), parameters unmoved (r0 0.60 vs
  0.59). An 11% eccentricity spread cannot separate the two
  anchorings; deciding this needs designs varying eccentricity
  severalfold. The ring-relative convention stands, now as a recorded
  choice rather than a silent one.
- **The envelope's width is combination-rule-conditional** (see the ⊗
  section above); statements about the attention window from these
  fits must name the rule they assume.
- Single split seed; no bootstrap intervals yet.
