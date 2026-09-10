# Reach-Avoid Results (5 seeds × 6000 steps ≈ 2 min each, CPU)

Goal-conditioned model: `pretrained/goal_es2.pth` (722-input `GoalEs2Model`,
trained 150 epochs on 12k goal-directed expert demonstrations, final MSE 0.145).
Reactive baseline: the original `pretrained/es2.pth` run in the same environment
(never sees the goal). Expert reference: 65.85 goals/min, 0 collisions/min.

## Primary metric: Pareto trade-off (goals/min vs. collisions/min)

| Agent | goals/min | collisions/min |
| --- | --- | --- |
| Goal-directed expert (reference) | 65.85 | 0.00 |
| **goal_es2 (goal-conditioned)** | **67.60** | **0.40** |
| goal_transformer (goal-conditioned) | 36.10 | 0.00 |
| goal_mlp (goal-conditioned) | 29.50 | 4.40 |
| es2 (reactive baseline) | 0.00 | 0.00 |
| mlp (reactive baseline, 3 seeds) | 0.00 | 14.17 |
| transformer (reactive baseline, 3 seeds) | 0.33 | 1.00 |

All three goal-conditioned models share the same demonstrations, dataloader and
722-feature observation; only the architecture differs (`goal_mlp` /
`goal_transformer` trained with train_goal_baseline.py; the transformer
checkpoint is its best-loss save from epoch ~57 of a 60-epoch run that was
stopped just before completion). ES2 essentially reproduces the expert's
frontier point; the Transformer is safe but half as fast; the MLP is dominated
on both axes (final imitation MSE: ES2 0.145, MLP 2.15; the transformer's loss
log was not saved because its run was stopped).

Side-finding: the continuous (non-terminating) evaluation exposes the pretrained
MLP as a far weaker avoider (14 collisions/min) than ES2 or the Transformer —
invisible under the original first-collision-ends-the-run protocol.

The goal-conditioned agent matches the expert's throughput at near-zero collision
cost and trivially dominates the frontier: the reactive baseline's throughput is
exactly zero (it survives but never approaches the destination). Per-seed
goals/min for goal_es2: 63.0, 67.5, 72.0, 62.5, 73.0; collisions per 2-minute
rollout: 1, 2, 0, 0, 1.

## Attention-field asymmetry

`field_asym` (mean attention-field value on rays within ±60° of the goal bearing
minus rays pointing away) is consistently ≈ −0.19 across all seeds — the learned
field is systematically asymmetric toward the goal sector, i.e. the top-down goal
channel visibly reshapes F(q | s_e). (The sign convention is learned; the stable
nonzero magnitude is the signal. For the goal-blind model the statistic is
undefined.)

## Goal-swap diagnostic (`goal_swap_probe.py`)

goal_es2 (goal-conditioned):

| condition | true-goals/min | fed-goals/min | collisions/min |
| --- | --- | --- | --- |
| correct | 67.60 | — | 0.40 |
| opposite (mirrored) | 0.40 | 102.40 | 0.20 |
| random | 2.00 | 65.80 | 0.40 |

es2 (goal-blind control): 0.00 true-goals/min in every condition,
0.00–0.20 fed-goals/min, 0.00 collisions/min.

- **Failure form (scored against TRUE goal):** sensitivity gap = 67.2 goals/min
  (−99.4%) under mirrored goals and 65.6 goals/min (−97.0%) under random goals —
  the agent chases wherever the goal channel points. The goal-blind control shows
  a 0.00 gap, as expected.
- **Positive form (scored against FED goal):** the agent reaches randomly
  assigned fed goals at 65.8/min — statistically indistinguishable from its
  correct-goal rate (67.6/min): it reaches whatever goal it is handed. The
  goal-blind control reaches fed goals at 0.2/min (chance). (The 102.4/min under
  `opposite` is a geometry artifact: successive mirrored goals are closer
  together on average than freshly sampled ones.)
- **Entanglement check:** collisions/min stay at 0.2–0.4 across all conditions —
  the goal input steers *direction* while collision avoidance is unaffected.
  This is the clean decomposition the method predicts: goal-reaching collapses
  under a wrong goal while avoidance does not.

## Architecture comparison under the goal-swap probe

All goal-conditioned models pass the goal-sensitivity test (they use the goal),
but only the structured architectures keep goal-steering and avoidance
decomposed:

| Model | true-goals/min correct→wrong | fed-goals/min (random) | collisions/min correct→wrong |
| --- | --- | --- | --- |
| goal_es2 | 67.6 → 0.4 / 2.0 | 65.8 | 0.4 → 0.2 / 0.4 (flat) |
| goal_transformer | 36.1 → 0.0 / 0.7 | 32.5 | 0.0 → 0.0 / 0.0 (flat) |
| goal_mlp | 29.5 → 0.2 / 1.2 | 29.6 | 4.4 → 8.6 / 10.6 (**doubles**) |

The MLP entangles the goal input with avoidance — feeding it a wrong goal
roughly doubles its collision rate — while ES2 and the Transformer keep
avoidance untouched under the same manipulation.

## Selection-history experiment (hazard-biased training): an informative null

Design: 8 of 10 balls confined to the left half of the arena during
demonstration generation (`expert_goal.py --hazard_side left`); a fresh
`goal_es2_hazard.pth` trained on those demonstrations (final MSE 0.29) and
probed in the STANDARD unbiased world against the unbiased model
(`selection_history_probe.py`, 5 seeds x 6000 steps).

Result: **no credible asymmetry transfers.** k by sector west/east =
0.01055/0.01067 (control 0.01045/0.01022); baseline-field west−east = +0.012
(control −0.010); nearest-ball clearance west−east = −2.1 px on ~170 px means
(control −0.2); steps-per-goal left vs right half = 47.2 vs 45.4 (control
43.9 vs 45.0). The hazard model is slightly more conservative overall
(64.6 goals/min, 0.00 collisions/min).

Interpretation: this null is the theoretically expected outcome for behavior
cloning — the expert is memoryless (its action is a pure function of the
current scan and goal), so a biased world biases only the distribution of
observations, not the observation→action mapping being imitated. The
architecture provides the slot for selection history (the per-ray gain), but
the slot is filled by outcome-driven learning, not imitation. Predicted
positive condition (untested): the same architecture trained or fine-tuned
with outcome signals (collision penalties / an online gain-map update rule)
in the hazard world should develop the asymmetry.

## Anticipation experiment (online target-history trace): positive, with an instructive reversal

Design (`anticipation_experiment.py`): a presence-driven leaky trace of goal
spawn positions (leaky centroid, rate η=0.05 per spawn) is bolted onto the
frozen goal_es2 agent and written into the field through the model's own
goal-field machinery at gain β=0.15. Exposure block: 120 goals, 70% in one
quadrant, each followed by a 100-step goal-free period (zero goal vector fed —
this exactly silences the phasic goal channel, so goal-free behavior reads out
the trace alone). Test block: 80 unbiased goals. Control arm: β=0, same
seeds (42–44, 3 seeds).

**Anticipatory drift — confirmed.** Goal-free distance to the frequent-quadrant
center falls across exposure in the trace arm (162 → ~110 px over 120 goals;
control fluctuates around 150–185 with no trend), and the agent is
pre-positioned for frequent goals: mean spawn distance 235 px vs. 282 px
(control) / 320 px (rare goals). The frozen action head translated a field
source it was never trained on into coherent drift — the source-blindness
commitment passing a behavioral generalization test.

**Persistence — confirmed, with an honest decomposition.** Early-test
goal-free bias remains (193 px in the first 10 test goals) and decays over
~20–30 unbiased goals, consistent with the trace time constant (1/η = 20
events). The residual late-test offset (~210–215 vs. control ~230) is a
centering artifact — under unbiased goals the trace centroid converges to the
arena center, which still pulls the agent centerward — not residual quadrant
history.

**Speed effect — a congruency structure, hidden by the region average.** The
frequent/rare split initially suggested pure interference (frequent goals
30.2 steps/100px vs. 13.8 control). Separating goals by their distance from
the trace centroid (re-run with position logging, `anticipation_v2_*.csv`;
identical seeds/RNG) reveals the real profile (exposure block):

| goal vs. centroid | trace | control | effect |
| --- | --- | --- | --- |
| <100px (history valid, n=45/51) | 13.8 steps (9.7/100px) | 27.4 (10.7/100px) | ~2x faster |
| 100–250px (near miss, n=203/228) | 79.3 (31.7/100px) | 41.2 (14.2/100px) | ~2x slower |
| >250px (clearly wrong, n=112/81) | 70.4 (22.0/100px) | 47.3 (15.1/100px) | moderate headwind |

History-valid goals get a compound benefit — pre-positioning (spawn distance
124 vs. 254px) plus tailwind, confirmed per-distance — while the cost
concentrates in the near-miss ring, where the centroid bump competes with
the goal in the endgame (the displaced-minimum hover). This is the
congruency profile of human probability cueing: benefit at the frequent
location, cost at near-misses. "Residual influence during pursuit can only
distort" is therefore too strong: an informationally redundant prior is
mechanically helpful exactly to the degree it points where the goal is; the
distortion comes from its error component only.

The net-negative *average* traces to a resolution mismatch: the true goal
distribution is a quadrant, but the leaky-centroid trace is a point prior,
so most frequent goals land in its near-miss ring (203 vs. 45). The model
spec's trace kernel width σ_h is the missing piece; prediction: net payoff
improves monotonically as the trace's spread approaches the true spawn
spread.

**σ_h follow-up — spread-matched trace: confirmed.** Replacing the point
centroid with a nonparametric spread trace (exponentially weighted recent
spawn positions, each contributing its own goal-field bump;
`--trace_kind spread`, same β=0.15/η/seeds; `anticipation_spread_*.csv`)
removes most of the interference while preserving the anticipation
(exposure block, steps/100px, control in parentheses):

| goal vs. centroid | point trace | spread trace | control |
| --- | --- | --- | --- |
| <100px (valid) | 9.7 | 12.8 | (10.7) |
| 100–250px (near miss) | 31.7 | **19.0** | (14.2) |
| >250px (wrong) | 22.0 | **15.7** | (15.1) |

Mean absolute steps per exposure goal: point 68.3 → spread 46.7 (control
40.6) — ~75% of the excess cost removed — while goal-free drift is fully
preserved (134 px vs. control 160, same as the point trace's 133) and the
far-goal headwind disappears entirely (inside the spread, opposing bumps
cancel into a plateau; from outside, the basin still guides). In the test
block the spread trace is essentially costless (frequent goals 12.7 vs.
control 13.0 steps/100px). As predicted, matching the trace's spread to the
true spawn spread converts near-miss interference into tailwind; the small
residual cost is consistent with the still-imperfect spread match and the
absence of pursuit-time normalization.

Interpretation: the payoff structure follows from *where in the trial the
uncertainty sits*. During the goal-free period the agent knows nothing about
the next goal except through the trace — genuine anticipatory uncertainty,
which the prior reduces, and its benefit is realized physically as goalless
drift toward the likely region (the shorter spawn distances). At goal onset,
however, the goal's exact location enters the observation and the prior
becomes informationally redundant — everything it knows is superseded — so
any residual influence during pursuit can only distort (and does, because
this implementation leaves the trace on at constant strength). Search
differs precisely in that uncertainty *persists after onset*: the target
must still be found, so the prior stays informative within the trial.

Human-testable prediction (two-part): adding a fully valid location cue to a
probability-cueing paradigm moves it from the search regime to this one —
the within-trial frequent-location benefit should vanish (the cue supersedes
the prior), while any residual history effect should survive only
pre-onset, e.g. as anticipatory gaze bias toward the frequent region before
the display appears.

Collisions: 11 (trace) vs. 6 (control) across ~230k steps — avoidance largely
intact. Raw data: `anticipation_goals.csv`, `anticipation_goalfree.csv`.

**β=0.05 follow-up — a gain window, bounded by an actuation floor.** Re-run
at a third of the trace gain (`anticipation_b005_*.csv`): the drift
disappears (goal-free distance 153 px vs. control 160, no acquisition trend;
spawn distance 279 ≈ control 282) while a small pursuit drag remains
(15.9 vs. 13.8 steps/100px on frequent goals). Cause: the environment
truncates sub-pixel forces (`player.x += int(fx)`), so a trace-induced force
below 1 px/step actuates nothing — β=0.05 sits under that floor during
goal-free periods, yet still perturbs pursuit where it sums with larger
forces. So with a static, always-on injection this environment offers no β
with net-positive throughput: the pre-positioning payoff is bounded
(~47 px ≈ 7 steps saved) while expressible-β interference costs ~35 steps
on frequent goals. The principled refinement this motivates — and the
plausible reason human history effects are small during explicit
goal-directed action — is pursuit-time normalization: the prior should
yield when a fully observed goal supersedes it, leaving anticipation nearly
free in idle periods and nearly invisible during pursuit.

## Fixation-start (reset) structure — the definitive anticipation result

The continuous design confounds anticipatory drift with positional
carryover (the agent simply stays where the last goal was). The
`--reset_agent` variant adopts the human trial structure: teleport to the
arena center (= return to fixation; previous-scan buffer reset so the
teleport is not a looming transient), a 100-step anticipation period with
no goal, then goal onset. Spread trace, β=0.15, η=0.05,
`--min_spawn_dist 150`, seeds 42–44, both arms
(`anticipation_reset_*.csv`).

**Per-trial anticipatory excursion, from a standardized start.** The start
is 184 px from the frequent-quadrant center. By the end of the anticipation
period the trace agent has moved *toward* the region (end distance ~153–176
px across exposure, learning visible in early bins), while the control
moves *away* (215–240 px — avoidance pressure alone). In the unbiased test
block the trace excursion decays across trials (187 → 214 px, approaching
the control's ~225) — per-trial persistence-with-decay, now measured as an
active excursion rather than a residue of where the agent happened to be.

**Net speed benefit for history-congruent goals via the head start**
(exposure block, absolute steps to goal):

| goal vs. centroid | trace | control | mechanism |
| --- | --- | --- | --- |
| <100px (valid) | 22.5 | 24.6 | shorter onset distance (205 vs. 239 px); per-distance speed equal |
| 100–250px | 32.8 | 38.8 | head start again (240 vs. 286 px); near-miss cost eliminated |
| >250px (incongruent) | 42.8 | 36.9 | headwind + longer onset distance |

Region view: frequent goals 31.7 vs. 33.6 steps (trace faster), rare 34.4
vs. 37.1. In the unbiased test block the trace arm is *slower* (frequent
47.0 vs. 40.8) — the now-invalid prior still pulls, i.e.
persistence-despite-cost, decaying at the trace rate.

Together with the spread-trace fix this recovers the full
probability-cueing phenomenology in the agent: active anticipatory
excursion from a fixed start that builds with exposure, faster acquisition
of history-congruent targets (via pre-positioning, not per-distance speed),
cost confined to history-incongruent targets and to the post-bias test
block, and trial-timescale decay. Collisions are elevated equally in both
arms by the teleports (24 vs. 22 over ~260k steps) — a property of the
reset protocol, not of the trace.

**Why the excursion stops at the region's edge (diagnostic).** Controlled
anticipation periods with a pre-built trace show the excursion is an
*equilibrium*, not slow progress: with balls removed the agent settles at
141 ± 5 px from the region center whether given 80 or 240 steps —
essentially the region boundary (half-extent 130 px). Balls are not the
limiter (with balls: 121 ± 64 — slightly closer on average, with jostling
variance; avoidance adds noise, not resistance), and forces are well above
the 1 px actuation floor. The cause is the spread trace's plateau: inside
the mass of remembered locations the bumps cancel, so the inward gradient
dies at the edge, and β sets the equilibrium depth (β=0.40 would settle at
77 px, inside the region). With β fixed at 0.15 — a deliberate
parsimony/no-tuning decision — the model's honest prediction is that the
agent **waits at the threshold of the likely region rather than at its
center**, short of the expected-distance-optimal waiting point.
Context-dependent expression (gating β by goal presence, i.e. pursuit-time
normalization) remains documented future work, not implemented.

## Reproduce

See README_reach_avoid.md; all runs used `--device cpu`, seeds 42–46.
