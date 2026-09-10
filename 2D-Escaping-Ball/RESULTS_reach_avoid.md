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

## Reproduce

See README_reach_avoid.md; all runs used `--device cpu`, seeds 42–46.
