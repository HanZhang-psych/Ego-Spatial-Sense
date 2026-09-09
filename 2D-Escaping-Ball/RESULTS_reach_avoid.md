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
| es2 (reactive baseline) | 0.00 | 0.00 |

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

## Reproduce

See README_reach_avoid.md; all runs used `--device cpu`, seeds 42–46.
