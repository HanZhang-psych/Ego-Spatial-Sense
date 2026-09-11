# Priority-Weighted, Fixed-Duration Evaluation

An add-on evaluation mode where the game **does not stop on collision**. Instead
it runs for a fixed duration and accumulates a **priority-weighted collision
score**: balls come in classes (e.g. red / amber / green), and hitting a
higher-priority ball costs more. Lower total score is better.

Provided by `evaluate_scored.py`; the original `evaluate.py` is unchanged.

## How it works

- **Fixed duration.** The episode always runs `--duration` seconds; survival
  time is no longer the metric.
- **Priority classes.** Each background ball is assigned a class (color +
  penalty), sampled from `--class_probs` and seeded by `--random_seed`.
- **Count once, then respawn.** On contact, the ball's penalty is added **once**
  and that ball is teleported to a random spot away from the ego. This keeps the
  ball count constant and prevents a single contact from being scored every
  frame.
- **Difficulty** is raised with `--num_balls` and/or the ball-speed range
  (`--ball_speed_min` / `--ball_speed_max`).
- The ego ball is drawn **dark** so it is never confused with a red
  (high-priority) ball; its action arrow is magenta.

> **Perception caveat.** The pretrained models observe only a 720-dim LiDAR
> *distance* scan (2 frames x 360 rays), which carries **no color/class**
> information. A pretrained agent therefore avoids every ball equally and cannot
> act on priority — the weighted score is an evaluation **metric** over existing
> models, not a behavior they were trained to optimize. Making the agent
> priority-aware would require extending the observation and retraining.

## Prerequisites

Same environment as the main task (see `README.md`). Use `--device cpu` (or
`mps`) on machines without an NVIDIA GPU.

## Run

```bash
# Ego spatial sense model, 20 balls, 60 s episode, with animation
python3 evaluate_scored.py --model_path "pretrained/es2.pth" --random_seed 42 --num_balls 20 --duration 60 --device "cpu" --render

# MLP model
python3 evaluate_scored.py --model_path "pretrained/mlp.pth" --random_seed 42 --num_balls 20 --duration 60 --device "cpu" --render

# Transformer model
python3 evaluate_scored.py --model_path "pretrained/transformer.pth" --random_seed 42 --num_balls 20 --duration 60 --device "cpu" --render

# Harder: 40 fast balls, more high-priority balls, custom penalties
python3 evaluate_scored.py --model_path "pretrained/es2.pth" --random_seed 42 --num_balls 40 --ball_speed_min 3 --ball_speed_max 6 --penalties 10 3 1 --class_probs 0.5 0.3 0.2 --duration 60 --device "cpu"
```

At the end, the script prints per-class hit counts, subtotals, total hits, and
the weighted score:

```
  class0: penalty=10.0 hits=6    subtotal=60
  class1: penalty=3.0  hits=1    subtotal=3
  class2: penalty=1.0  hits=2    subtotal=2
  total hits      : 9
  WEIGHTED SCORE  : 65  (lower is better)
```

During rendering, the HUD shows time remaining, the running score, total hits,
and per-class counts (`c0 c1 c2 ...`).

## Flags

Inherited from `evaluate.py`:

- `--model_path` — pretrained model to load (`es2` / `mlp` / `transformer`
  inferred from the filename).
- `--random_seed` — fixes initial positions **and** class assignments.
- `--num_balls` — number of background balls (difficulty).
- `--device` — `cpu`, `mps`, or `cuda`.
- `--render` — enable animation.
- `--max_speed` — ego per-step speed cap (default `10.0`).

Scored-mode specific:

- `--duration` — episode length in **seconds** (default `60`); the run never
  ends early.
- `--ball_speed_min` / `--ball_speed_max` — inclusive integer range for each
  ball's speed in px/step (default `1`–`3`); raise for difficulty.
- `--penalties` — space-separated per-class penalties, **highest priority
  first** (default `5 2 1`). The number of values sets the number of classes
  (max 5: red, amber, green, blue, purple).
- `--class_probs` — spawn probability per class, same length as `--penalties`
  (default: uniform). Auto-normalized, so it need not sum to 1.

Headless helpers (for scripting/sweeps):

- `--no_throttle` — run as fast as possible instead of capping at `--frequency`
  steps/sec.
- `--quiet` — suppress per-run prints.

## Notes

- Because the score depends on the seeded ball layout and class assignment, use
  several seeds and average when comparing models — a single seed is noisy.
- To compare models fairly, keep `--random_seed`, `--num_balls`, speed range,
  `--penalties`, and `--class_probs` identical across runs; each model then
  faces the same ball population.
