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

Evaluate on the trial-reset reach-avoid task:

```bash
python3 evaluate_trial_reach_avoid_history.py \
  --model_path pretrained/goal_history_geometric_trial_unbiased_360_150.pth \
  --num_features 360 \
  --num_trials 100 \
  --max_steps_per_trial 1000 \
  --device cpu
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

Trial-reset evaluation:

```text
success:   100 / 100 = 1.000, mean_steps = 60.4
collision:   0 / 100 = 0.000
timeout:     0 / 100 = 0.000
```

For comparison, the original pretrained `GoalEs2Model` with direct goal
injection and no history also reached:

```text
success:   100 / 100 = 1.000, mean_steps = 42.4
collision:   0 / 100 = 0.000
timeout:     0 / 100 = 0.000
```

The geometric goal/history model therefore matches the original model on
success and collision rate in this trial-reset evaluation, but reaches the
target more slowly.

## Notes

- The evaluation task uses unbiased target sampling.
- Each trial resets the player, obstacles, and target.
- Success means reaching the target before collision or timeout.
- Collision ends the trial as a failure.
- The history trace persists across successful trials during evaluation.
