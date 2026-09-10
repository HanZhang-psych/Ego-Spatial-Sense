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
3. **Spatial gradient around the HP location**: model is FLAT (6.66 /
   6.76 / 6.41 at ring distances 1/2/3) where W&T observed graded
   spillover to neighbors — a committed divergence: the fitted traces
   are slot-indexed with no spatial spread. The missing ingredient is
   the trace kernel width sigma_h — exactly the component the
   agent-side experiments already identified and implemented (the
   spread-matched trace), specified in the model doc but not yet in
   the search fits. The same parameter fixes the same limitation in
   both domains.

Framing caveat: W&T ran a capture regime (singleton-detection mode);
our weights come from suppression paradigms, so absolute capture
levels are not comparable — the reproduced content is the location
modulation, its spillover to targets, and (negatively) the gradient.

## Trace spread sigma_h (fit_sigma.py; results_fit_sigma.json)

Adding a mass-preserving ring-Gaussian spread to the trace updates
(sigma -> 0 nests the slot-exact final model): the free fit collapses
to sigma = 0.13 slots (nothing reaches even the nearest neighbor),
held-out NLL identical to the null (1.22797 both; r0 refit jointly to
0.55, marginally improving on the earlier 1.22833). Conclusion: **no
measurable trace spread in roving-location designs** — neighbor
repeats are rare and unstructured, so the kernel has no variance to
bite on. The Wang & Theeuwes gradient divergence therefore stands as
a prediction about *biased* designs specifically: sigma_h is
measurable only where location statistics concentrate mass, which is
exactly where W&T observed the gradient and where the agent-side
spread trace earned its keep. A biased-design human fit is the
experiment that would estimate it.

## Pre-onset gaze bias: a floor-limited null (analyze_preonset.py)

Prediction tested: the pre-onset prior (window x history) should
displace gaze at display onset toward recent target locations and
away from recent singleton locations. Measure: each trial's initial
fixation position relative to the subject's own median, projected
onto lag-1 and fitted-trace directions; 116,500 trials, 287 subjects.

Result: **null on every projection** (all effects 0.02-0.06 px against
a 38.5 px median offset; all |t| < 0.6) — including the *positive
control*: projection toward the previous trial's final gaze position
(+0.02 px, t = 0.27). Return-saccade undershoot is a robust
oculomotor phenomenon; its complete absence says initial-fixation
position in these datasets carries no previous-trial structure of ANY
kind, motor included — consistent with enforced refixation
(drift-correct / fixation checks before trial start) clamping exactly
this variable. Verdict: a measurement-floor null, not a refutation —
the paradigm actively resets the DV. The discriminating test needs
the continuous sample reports (pre-onset microsaccade direction and
drift within the inter-trial epoch), which exist on OSF (~9M gaze
samples) but are not in the local fixation reports. The prediction
stands, sharpened: it now specifies the measure (pre-onset
microsaccades, not fixation position) and predicts the control
(undershoot) should appear there too, validating the channel.

## Caveats on record

- **The envelope's width is combination-rule-conditional** (see the ⊗
  section above); statements about the attention window from these
  fits must name the rule they assume.
- Single split seed; no bootstrap intervals yet.
