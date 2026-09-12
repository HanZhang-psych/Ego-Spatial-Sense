# Goal-History Reach-Avoid Results

This note records the direct-geometric goal/history experiment for the
trial-based reach-avoid task.

## Model

The final model is `GoalHistoryEs2Model` in `model/goal_history_es2.py`.
It keeps the original ES2 LiDAR pathway as the obstacle-perception module,
then adds direct geometric fields for the visible goal and the world-space
history trace:

```text
obstacle_field = ES2(previous_scan, current_scan)
goal_field = goal_gain(distance_to_goal) * cos(ray - goal_bearing)
history_field = history_gain(distance_to_history) * cos(ray - history_bearing)

combined_field = obstacle_field + goal_field + beta_H * history_field
action = planner(combined_field)
```

The history trace is a world-space leaky accumulator over reached target
locations:

```text
trace = (1 - eta_H) * trace + eta_H * reached_goal_xy
```

### Conceptual note: the gain MLPs are psychophysical transfer functions

All three channels translate a physical variable into an internal signal
strength before anything is combined — the same job, done with different
machinery:

- **Obstacle side**: a fixed sigmoid family with learned per-ray
  sensitivity `k` maps distance to perceived closeness; the looming delta
  then passes through `delta_proj` (two stacked linears with no
  nonlinearity, i.e. an implicit scalar gain + offset). Constrained
  psychophysics: the curve's shape is assumed, only its sensitivity is
  learned.
- **Goal and history sides**: a free-form ReLU MLP (`goal_gain`,
  `history_gain`) maps distance to field strength, multiplied by fixed
  cosine geometry for direction. Nonparametric psychophysics: the curve's
  shape itself is learned.

So the goal/history gain MLPs are the conceptual analog of the obstacle
sigmoid — a learned transfer function from a physical quantity (distance)
to signal strength. The difference is what the output means: the obstacle
curve outputs *sensory* strength (perceived proximity/looming), while the
goal and history curves output *relevance* (top-down pull). One transduces
physics; the others transduce task value — but structurally all three are
psychophysical transforms feeding fields in the same 360-ray space.

A consequence: unlike the visual-search model, there is no single scalar
channel gain (no `g_C`/`g_F` analog) — channel strength is
distance-dependent by construction, so comparing channels means comparing
curves, not numbers. `beta_H` is the one named scalar (analogous to the
search model's `beta`s), kept so the history strength stays readable. We
keep this asymmetric structure deliberately, for simplicity.

How the goal reaches the sensory input also differs between the two
models, in a precise way. In one sense both do the same thing — convert
sensed physics into task-signed relevance: the search model's `D_T` takes
the raw color-contrast the retina delivers and re-signs it by the target
color, and the agent's goal field takes a sensed physical object (the
visible goal's distance and bearing) and converts it into signed pull.
Neither goal conjures evidence from nowhere; both re-weight evidence
originating in physical stimuli. The residual difference is architectural:
in the search model the goal modulates a *shared* sensory channel — the
color-contrast map exists without the goal, which only flips which pole is
positive — whereas in the agent model the goal object has its own
dedicated pathway that never passes through the obstacle psychophysics, so
there is no goal-free version of that channel being modified. Search:
goal-modulation of a general channel; agent: goal-dedicated transduction.

## Reproduction

Run commands from `2D-Escaping-Ball/`.

Generate the full 360-ray unbiased demonstration dataset:

```bash
python3 expert_goal_history.py \
  --num_features 360 \
  --num_episodes 20 \
  --max_steps_per_episode 3000 \
  --record_every 5 \
  --goal_free_steps 0 \
  --goal_bias none \
  --output dataset/data_goal_history_cont_unbiased_360.csv
```

Train the 360-ray geometric goal/history model:

```bash
python3 train_goal_history_es2.py \
  --data_path dataset/data_goal_history_cont_unbiased_360.csv \
  --model_path pretrained/goal_history_geometric_trial_unbiased_360_150.pth \
  --log_path loss_goal_history_geometric_trial_unbiased_360_150.csv \
  --num_features 360 \
  --num_epochs 150 \
  --device cpu
```

Evaluate on the continuous reach-avoid task (the evaluation of record —
it mirrors the expert-demo structure: a persistent world where reaching a
goal spawns the next one, no resets; add `--disable_history` for the
ablated model):

```bash
python3 evaluate_goal_history.py \
  --model_path pretrained/goal_history_geometric_trial_unbiased_360_150.pth \
  --goal_bias none \
  --goal_free_steps 0 \
  --max_steps 6000 \
  --num_seeds 3
```

## Results

Dataset generation:

```text
dataset/data_goal_history_cont_unbiased_360.csv
360 rays
12,000 recorded rows
expert: 1316 goals, 0 collisions over 60,000 simulated steps
expert rate: 65.80 goals/min, 0.00 collisions/min
```

Training:

```text
checkpoint: pretrained/goal_history_geometric_trial_unbiased_360_150.pth
loss: 62.88191 -> 21.31034
eta_H: 0.0462
beta_H: 0.1688
```

Continuous evaluation (2026-09-12, the evaluation of record; 3 seeds x
6,000 steps, unbiased goals, no goal-free periods — the expert-demo
structure, so the models are tested in the distribution they were trained
on):

```text
with history:    54.5 goals/min, 1.5 collisions/min
expert baseline: 65.8 goals/min, 0.0 collisions/min
```

The original pretrained `GoalEs2Model` under its own continuous evaluator
(`evaluate_reach_avoid.py`, same seeds/steps; protocol differs slightly —
no per-goal timeout and default spawn distance):

```text
plain GoalEs2:   67.5 goals/min, 0.5 collisions/min
```

The with-history model is slower than both the expert and the plain goal
model on this unbiased task (see the ablation below: the gap is the
history channel's fault).

An earlier trial-reset evaluation (each trial reset player, obstacles,
and target; superseded because it mismatched the continuous expert-demo
structure) gave: with-history 100/100 success, 0 collisions, 60.4 mean
steps; plain GoalEs2 100/100, 42.4 mean steps.

## Ablation: selection history disabled (2026-09-12)

The trial-reset evaluation resets the target with unbiased sampling every
trial, so a location-history trace has nothing systematic to exploit.  We
therefore retrained the same architecture on the same dataset with history
disabled: `--disable_history` freezes `beta_H` at 0, so the history field
never enters the sum (the history-gain MLP then receives no gradient and
`eta_H` stays at its 0.05 init).  Both trainer and evaluator take the
flag.

Training (150 epochs, cpu, identical data and settings):

```text
checkpoint: pretrained/goal_history_geometric_trial_unbiased_360_150_nohist.pth
wall clock: 37.8 s
loss: 62.96960 -> 8.64123 (best 8.58004)
```

The with-history run on the same data ended at loss 21.31 after the same
150 epochs (its wall clock was not recorded).  Disabling history more than
halves the final imitation loss — the randomly-initialized history channel
is pure input noise on this task, and the optimizer spends capacity
suppressing it rather than fitting the expert.

Continuous evaluation (`evaluate_goal_history.py --goal_bias none
--goal_free_steps 0 --max_steps 6000 --num_seeds 3 --disable_history`):

```text
no history:      68.0 goals/min, 0.0 collisions/min
with history:    54.5 goals/min, 1.5 collisions/min
plain GoalEs2:   67.5 goals/min, 0.5 collisions/min
expert baseline: 65.8 goals/min, 0.0 collisions/min
```

With history disabled the model matches the plain goal model and the
expert's throughput, with zero collisions; the with-history model is ~20%
slower and collides.  On this unbiased task, then, the history channel
costs training fit, throughput, and safety while buying nothing — the
expected result, and the motivation for a future evaluation with
goal-location structure (e.g. biased or repeating goal locations) where
history could help.  (The superseded trial-reset evaluation agreed:
no-history 40.5 mean steps vs with-history 60.4, both 100/100.)

## Stress test: 40 balls (2026-09-12)

The model of record (no-history checkpoint, `--disable_history`) under
the same continuous protocol but with `--num_balls 40` — 4x the training
density of 10:

```text
40 balls: 25.8 goals/min, 6.0 collisions/min
10 balls: 68.0 goals/min, 0.0 collisions/min
```

Throughput drops to ~38% and collisions rise from zero to 6/min.  The
agent was trained entirely at 10-ball density, so this measures
out-of-distribution crowding: paths lengthen (~30 steps/100px vs ~12.6
at 10 balls) and the obstacle field frequently saturates with no safe
gap toward the goal.

## Biased-world comparison: history enabled vs disabled (2026-09-12)

Both models trained from scratch (150 epochs) on a quadrant-biased
demonstration set with anticipation periods —
`dataset/data_goal_history_cont_biased_360.csv`: 80% of goals in the
fixed quadrant (140,140)-(400,400), 25 goal-free steps before each spawn
during which the expert steers toward its trace; expert 823 goals, 0
collisions, 41.15 goals/min (throughput includes the goal-free waits).
Evaluated in the matching biased world (3 seeds x 6,000 steps,
`--goal_free_steps 25`):

```text
                    train loss   goals/min   coll/min   free drift   freq/rare steps per 100px
with history        17.03        27.5        28.3       +1.26        27.9 / 21.5
history disabled    18.35        35.3         1.0       +1.42        23.8 / 12.8
```

Two findings, both against the history channel here:

1. The with-history model is catastrophically collision-prone (28.3/min
   vs 1.0) and slower.  Its training loss is lower (17.0 vs 18.3), so
   the trace helps imitate the expert in-sample, but at rollout the
   trace-following field apparently drags the agent through traffic it
   would otherwise skirt — imitation gain, control loss.
2. The no-history model drifts toward the frequent region during
   goal-free periods just as strongly (+1.42 px/step) WITHOUT any trace
   input.  This exposes a confound in the design: the frequent region is
   at a FIXED world location, and the agent can infer its position in
   the arena from wall distances in the scans — so the spatial bias is
   learnable as a static policy prior, no memory required.  A fixed
   frequent region cannot dissociate selection history (a trace updated
   by experience) from a learned constant preference.

The dissociating experiment is a frequent region that MOVES between
episodes (or mid-episode): a static prior then fails, and only a model
whose trace tracks recent goals can anticipate correctly.

### Respawn-at-center redesign (2026-09-12, fixation-start structure)

Han's correction to the design above: every goal cycle should begin with
the agent teleported back to the center (the fixation point), so
anticipation always starts from a common origin.  Implemented as
`--respawn_center` in the expert and the evaluator; demonstrations carry
a `respawn` column and the trainer excludes scan pairs that straddle a
teleport (the looming delta would otherwise compare two positions as if
continuous).  Dataset: `data_goal_history_biased_respawn_360.csv`
(not committed - 76 MB class of file); expert 798 goals, 6.0
collisions/min - the teleport can land the expert next to traffic, so
even the expert now collides.  Same biased world, 150 epochs each,
eval 3 seeds x 6,000 steps with `--respawn_center`:

```text
                    train loss   goals/min   coll/min   free drift   freq/rare steps per 100px
expert (demos)      -            39.9        6.0        -            -
with history        14.19        21.0        1.3        -0.43        33.9 / 13.9
history disabled    18.32        40.3        6.3        +0.12        17.4 / 15.5
```

The redesign sharpened the imitation story and reversed nothing at
rollout:

1. History now genuinely explains the demonstrations: the training-loss
   gap widened to 4.1 (14.2 vs 18.3; it was 1.3 in the no-respawn
   design).  From a fixed start, where the expert drifts during
   anticipation is determined by its trace, and only the history channel
   can represent that.
2. The no-history model reproduces the expert almost exactly at rollout
   (40.3 vs 39.9 goals/min, 6.3 vs 6.0 collisions/min) but shows NO
   anticipatory drift (+0.12 px/step, frequent/rare path efficiency
   nearly equal) - with respawn, the static-prior shortcut that
   contaminated the previous design is gone.
3. The with-history model again fits better and performs worse: half the
   throughput (21.0), drift AWAY from the frequent region (-0.43), and
   markedly inefficient paths to frequent-region goals (33.9 steps per
   100 px).  Its trace field helps predict expert actions one step at a
   time but destabilizes the closed loop - compounding-error brittleness
   of behavioral cloning concentrated in the newly-added channel.

Standing conclusion so far: in this task family the history channel
buys in-sample imitation fit, not closed-loop competence.  The honest
open question is whether that reflects the channel (geometric trace
field) or the training method (single-step behavioral cloning with no
rollout correction, e.g. no DAgger).

### Staged training: freeze the policy, then train only history (2026-09-12)

Han's proposal, exploiting that the history field feeds the same frozen
planner as the other channels: stage 1 trains everything with history
disabled (the `_nohist` checkpoint above); stage 2 warm-starts from it,
freezes all parameters except the history ones (`beta_H`, the
`history_gain` MLP, `raw_eta_H` - 26 scalars), and trains only on the
goal-free anticipation rows, where the trace is the sole driver of the
expert's actions.  New trainer flags: `--train_history_only`,
`--goal_free_only`.  Stage-2 loss (goal-free rows) 18.0 -> 17.07;
fitted eta_H 0.060, beta_H -0.081 (sign not interpretable alone - it
composes with the free-signed gain curve).  Evaluation as before:

```text
                    goals/min   coll/min   free drift   freq/rare steps per 100px
history disabled    40.3        6.3        +0.12        17.4 / 15.5
joint history       21.0        1.3        -0.43        33.9 / 13.9
staged history      36.0        4.2        +0.59        19.3 / 23.0
```

Staging rescues the history channel.  The staged model is the first
with the anticipation signature and near-full competence: drift toward
the frequent region during goal-free periods (+0.59 px/step - 5x the
ablated model's residual, opposite in sign to the joint model's), and
frequent-region goals now reached MORE efficiently than rare ones (19.3
vs 23.0 steps/100px; every other variant had the advantage absent or
inverted), at a modest throughput cost (36.0 vs 40.3 goals/min).

Interpretation: joint training lets history gradients reshape the
planner and destabilize the whole policy; freezing the competent
policy and letting only the 26 history parameters adapt confines the
selection-history machinery to the anticipation behavior it is meant
to explain.  This mirrors the search model's separation between fixed
salience/goal machinery and the small set of history parameters fitted
on top.

### Split by target location: likely vs unlikely (2026-09-12)

Per-region reach rates added to the evaluator (goals reached / goals
spawned per region; the efficiency metric alone misses timeouts).
Staged-history vs history-disabled, same protocol:

```text
                         staged history      history disabled
frequent (likely):
  steps per 100 px       19.3                17.4
  reach rate             0.978               0.986
rare (unlikely):
  steps per 100 px       23.0                15.5
  reach rate             1.000               1.000
freq/rare efficiency     0.84                1.12
```

The signature is an interaction, not a main effect.  The
history-disabled model treats the two regions alike (in fact slightly
faster to rare goals, 15.5 vs 17.4).  The staged model reverses the
ordering: frequent goals are reached ~16% more efficiently than rare
ones (19.3 vs 23.0).  In absolute terms the staged model pays a cost in
both regions relative to the ablated model, but the cost is
concentrated on unlikely targets (+7.5 steps/100px at rare vs +1.9 at
frequent) - anticipatory drift toward the frequent region leaves the
agent poorly positioned when the goal appears elsewhere.  Reach rates
are at ceiling for both models in both regions, so the effect is purely
in path efficiency, not success.

This is the agent-model analog of the search paradigm's
likely-vs-unlikely location effect: history helps where the
environment's statistics repeat and costs where they break - the
biased-attention trade-off, now in closed-loop navigation.

### Center-referenced spawn rule (2026-09-12)

Han's objection to the split above: if the agent drifts toward (thus
closer to) the upcoming target, likely-position goals should get
cheaper - and they structurally could not, because `min_spawn_dist`
(250 px) was measured from the PLAYER, so goals always spawned >=250 px
from wherever the agent stood, and the efficiency metric normalizes by
spawn distance anyway.  Fix: `--spawn_from_center` measures the rule
from the arena center (the respawn point) in expert and evaluator, so
goal positions are defined relative to the arena, like a search
display.  Full staged pipeline re-run on the new dataset
(`data_goal_history_biased_respawn_cs_360.csv`, not committed; expert
990 goals, 49.5 goals/min):

```text
                    goals/min   coll/min   free drift   freq/rare steps per 100px
staged history      35.8        3.7        -0.31        19.7 / 20.0
history disabled    36.7        3.0        -0.75        18.4 / 20.4
```

The proximity benefit still did not materialize: throughput is now
equal (35.8 vs 36.7), and the staged model is no better at frequent
goals in absolute terms.  The relative history effect on goal-free
drift is stable (+0.44 px/step vs the ablated baseline, matching the
+0.47 relative effect of the previous design), but both baselines
shifted negative.  Two candidate reasons, recorded as open issues:

1. Magnitude shrinkage: the models drift at well under 1 px/step while
   the expert covers up to 10 px/step toward its trace - MSE cloning
   averages over uncertain directions and shrinks the anticipation
   vector, so ~10-15 px of approach against >=250 px goal distances is
   a few percent, invisible in throughput.
2. Spawn geometry: with the 250 px rule measured from the center, much
   of the quadrant (whose center is only ~180 px from the arena
   center) is ineligible, so frequent goals concentrate in the
   quadrant's outer corner - drifting toward the quadrant center aims
   at a spot goals cannot occupy.

The honest summary: the spawn-rule artifact is fixed, and with it
fixed, the anticipation the cloned models actually express is too weak
and too misaimed to pay.  Making history pay would need some
combination of a stronger/faster expert drift in the demos, a smaller
min-spawn distance, or a frequent region positioned so its eligible
zone is where the trace points.

### min_spawn_dist 150 (2026-09-12): worse for everyone, and no history effect

Same center-referenced staged pipeline with `--min_spawn_dist 150`
throughout (dataset `data_goal_history_biased_respawn_cs150_360.csv`,
not committed; expert 1069 goals, 53.45 goals/min):

```text
                    goals/min   coll/min   free drift   freq/rare steps per 100px   freq/rare reach
staged history      20.0        8.2        -0.06        53.0 / 24.5                 0.98 / 0.90
history disabled    19.8        5.7        -0.03        54.6 / 24.7                 0.99 / 0.90
```

Lowering the spawn floor backfired: BOTH models collapse to ~20
goals/min (from ~36) while the expert improves to 53.5 - the
imitation gap triples.  Frequent-region efficiency balloons to ~54
steps/100px for both models (near goals inflate the normalized metric,
but ~80 steps for a 150 px goal is genuinely poor control), and the
staged model's fitted beta_H shrinks to -0.05 with drift ~0: in this
regime stage 2 finds almost nothing for history to explain that
survives cloning.  No history benefit, no history cost - the
bottleneck this variant exposes is near-goal closed-loop control, not
anticipation.  Conclusion: the spawn floor is not the lever; the
magnitude shrinkage of the cloned anticipation (and near-goal control
generally) is where the imitation gap lives.

## Target-location priming: mechanism-sufficiency demonstration (2026-09-12)

Reframing (Han): the end goal is the agent-side analog of the search
model's target-location priming (tutorial Sec. 9c), not goalless drift
per se - and behavioral cloning from a history-blind expert cannot
produce it, while training the expert to drift measures probability
cueing instead.  So this is a sufficiency demonstration, not a training
evaluation: add a history field to a competent trained policy and show
the human signature appears, dose-dependently, with no training.

Construction (`evaluate_priming_goal_history.py`): base policy = the
trained history-disabled respawn checkpoint (its own history channel is
exactly zero, asserted).  The previous goal location (one-back trace)
is fed through the TRAINED goal-gain machinery as a faint goal -
`beta_H * geometric_field(trace - player, goal_gain)` added to the
obstacle + goal fields.  The single free parameter is beta_H, swept
with beta_H = 0 as the built-in control.  Trials (fixation-start,
~300/beta over 3 seeds): teleport to center; 25 goal-free steps (drift
toward the remembered location logged); goal onset - REPEAT trials at
the previous goal's location, CHANGE trials at matched eccentricity
rotated 90-270 deg; steps-to-goal and first-step heading error
recorded.

```text
beta   repeat steps   change steps   priming(chg-rep)   drift px/step   timeouts
0.0    63.9           56.9            -7.0              -0.12             2
0.1    50.2           80.0           +29.8              +0.64            13
0.2    42.8          152.8          +110.0              +1.90            50
0.4    31.8          183.7          +151.9              +4.29           143
```

The full human signature, from one parameter:

1. No mechanism, no effect (beta 0: repeat ~ change; the -7 baseline
   asymmetry is noise-scale relative to the effects).
2. Repeat facilitation grows monotonically (63.9 -> 31.8 steps).
3. Change cost grows faster (56.9 -> 183.7), with timeouts exploding at
   high beta - the agent is captured by the remembered location and
   fails to disengage: the navigation analog of attentional capture by
   selection history.
4. Anticipatory goalless drift emerges from the SAME mechanism and
   scales with the same beta (-0.1 -> +4.3 px/step) - history as
   anticipation and history as priming unified under one parameter,
   with no drift-specific training.

First-step heading error barely moves at low beta (9.8-10.5 deg): the
visible goal dominates the initial turn, and the priming lives in the
en-route dynamics.  beta ~ 0.1 is the human-plausible regime - clear
repeat benefit, moderate change cost - before the mechanism tips into
capture.  This parallels the search model's history term exactly: a
small parametric bolt-on entering through the same spatial machinery as
the goal, whose strength determines both the benefit and the bias.

## Stress test: ball speed sweep (2026-09-12)

`--speed_multiplier` in `evaluate_goal_history.py` scales every ball's
velocity after world creation.  Training speeds are 1-3 px/step and never
change, so multiplied speeds are outside anything in the demonstrations.
Model of record, continuous protocol (3 seeds x 6,000 steps, 10 balls):

```text
x1: 68.0 goals/min, 0.0 collisions/min   (ball speeds 1-3)
x2: 72.2 goals/min, 0.5 collisions/min   (2-6)
x3: 73.3 goals/min, 0.7 collisions/min   (3-9)
x4: 71.0 goals/min, 4.8 collisions/min   (4-12, some balls now outrun
                                          the ego's max speed of 10)
```

The looming channel extrapolates: throughput holds (even ticks up
slightly — fast balls vacate corridors sooner) and collisions stay near
zero through x3, where the fastest balls nearly match the ego's own
speed.  Degradation begins at x4, where 1-3 px/step balls become 4-12
and the fastest of them are strictly faster than the ego, so some
contacts stop being avoidable even in principle.  Contrast with the
density stress test (40 balls: throughput ~38%, 6 collisions/min) and
the pursuer test (collapse): speed generalization is the axis this
model handles best, consistent with the looming delta scaling linearly
and monotonically with obstacle speed.

## Adversarial test: seeded pursuer (2026-09-12)

`evaluate_adversarial_goal_history.py` transplants the stressor of
`evaluate_adversarial.py` (which serves the goal-blind escape models and
cannot consume this model's goal/history input) into the continuous goal
task: at a seeded trigger (5-30 s window), one ball switches to erratic
pursuit of the ego — homing heading with ±0.6 rad noise, speed equal to
the ego's max so escape stays possible.  Metrics are goals/min and
collisions/min split at the trigger.  Model of record, 3 seeds x 6,000
steps, 10 balls:

```text
before pursuit: 64.8 goals/min,  0.0 collisions/min (mean 511 steps/seed)
during pursuit:  8.1 goals/min, 19.5 collisions/min (mean 5489 steps/seed)
```

Under pursuit the model collapses: throughput drops ~8x and the pursuer
catches it ~20 times/min (each sustained contact counts once).  Expected,
and diagnostic: the policy is pure imitation of a goal-seeking expert
that never faced a pursuer, so it treats the hunter as an ordinary
obstacle to skirt while pressing toward the goal — it has no flee
behavior and no concept of a threat worth abandoning the goal for.  The
original ES2 escape models handle this stressor because escaping is all
they do; the goal model shows the complementary failure.  A future
goal-vs-threat arbitration (e.g. a negatively-weighted looming channel
strong enough to override the goal field) is the obvious fix.

## Notes

- The evaluation task uses unbiased target sampling.
- The evaluation of record is continuous, mirroring the expert demo: one
  persistent world per seed; reaching a goal (or a 300-step timeout)
  spawns the next; collisions are counted, not terminal.
- The history trace persists across goals within a seed and updates on
  each goal spawn.
- The earlier trial-reset evaluation (world reset after every goal) is
  superseded; its script `evaluate_trial_reach_avoid_history.py` was
  removed.
