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
