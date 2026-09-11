# Adversarial Edge-Case Evaluation

This is an add-on to the 2D Escaping Ball task that introduces a **long-tail
stressor never seen during training**: at a randomly chosen (but *seeded*)
moment, one of the background balls switches into a pursuit mode and starts
moving erratically toward the ego ball.

It directly probes Feature #1 of ego spatial sense — *"Reflexive avoidance of
imminent collision threats … even in previously unseen scenarios"* — and the
attention field's sensitivity to fast-approaching entities. In training, balls
only move at constant velocity and bounce off walls; none of them chase, so a
homing pursuer is a genuinely new **behavior mode**, not just a change in ball
count.

Two scripts are provided; the original `evaluate.py` is unchanged:

| Script | Purpose |
|---|---|
| `evaluate_adversarial.py` | Run **one** model on the adversarial scenario (optionally with animation). |
| `sweep_adversarial.py` | Sweep **many seeds per model** and print a survival-time summary table. |

## Design notes

- **Reproducible, not wall-clock random.** The trigger time, the pursuer, and
  its per-step jitter are all drawn from `--random_seed`, so every model faces
  the *identical* perturbation on a given seed. The pre-trigger ball layout uses
  the same RNG sequence as `evaluate.py`, so the base scenario matches that
  script for the same seed.
- **Solvable by construction.** The pursuer's speed is bounded (default: the
  ego's own `--max_speed`), so a competent evader can still escape. A pursuer
  strictly faster than the ego would make collision unavoidable and the test
  would discriminate between no models.
- **"Erratic" = homing heading + bounded noise.** Each step the pursuer aims at
  the ego, then perturbs its heading by a random angle in
  `[-pursuit_noise, +pursuit_noise]` radians. Speed magnitude stays fixed at the
  pursuer speed, so the bound above always holds.
- Report this as its **own condition** — a separate survival-time row — rather
  than folding it into the 10/20/30/40-ball averages from the paper.
- Because the expert potential-field policy already reacts to approaching balls,
  a model handling this is a legitimate **extrapolation** of demonstrated
  behavior, not a wholly novel skill.

## Prerequisites

Same environment as the main task (see `README.md`). On a CPU-only or Apple
Silicon machine, pass `--device cpu` (or `mps`); `cuda` requires an NVIDIA GPU.

---

## 1. Run a single model

```bash
# Ego spatial sense model, with animation
python3 evaluate_adversarial.py --model_path "pretrained/es2.pth" --random_seed 42 --num_balls 10 --device "cpu" --render

# MLP model
python3 evaluate_adversarial.py --model_path "pretrained/mlp.pth" --random_seed 42 --num_balls 10 --device "cpu" --render

# Transformer model
python3 evaluate_adversarial.py --model_path "pretrained/transformer.pth" --random_seed 42 --num_balls 10 --device "cpu" --render
```

When the pursuit activates, the pursuing ball turns **orange** and the on-screen
status switches from `[calm]` to `[PURSUIT]`. The run ends on the first
collision or at `--max_steps`, and the survival time is printed in steps and
seconds.

### Flags

Inherited from `evaluate.py`:

- `--model_path` — path to the pretrained model to load (`es2` / `mlp` /
  `transformer` is inferred from the filename).
- `--random_seed` — fixes the initial conditions **and** the adversarial trigger.
- `--num_balls` — number of background balls (must be ≥ 1).
- `--device` — runtime device: `cpu`, `mps`, or `cuda`.
- `--render` — enable animation.
- `--max_speed` — ego's per-step speed cap (default `10.0`).
- `--max_steps` — end the run after this many steps (default `15000` = 300 s at
  the default `--frequency 50`).

Adversarial-specific:

- `--pursuer_speed` — speed of the pursuing ball once active. `<= 0` (the
  default) falls back to `--max_speed`. **Keep it ≤ `--max_speed`** to keep the
  task solvable; lower values (e.g. `5`–`7`) give clearer separation between
  models.
- `--pursuit_noise` — maximum heading perturbation in **radians** (default
  `0.6`). Higher = more erratic; `0` = pure straight-line homing.
- `--trigger_min` / `--trigger_max` — the seeded time window (in **seconds**)
  during which the pursuit may start (default `5`–`30`).
- `--pursuer_index` — index of the ball that turns hostile; `< 0` (default)
  picks one at random (seeded).

Headless helpers (mainly for the sweep):

- `--no_throttle` — headless only; run as fast as possible instead of capping at
  `--frequency` steps/sec.
- `--quiet` — suppress per-run progress prints.

---

## 2. Sweep seeds across models

`sweep_adversarial.py` runs the adversarial scenario for every seed against every
model and prints a summary table. It always runs headless and un-throttled.

```bash
# Compare all three pretrained models across 10 seeds (0–9)
python3 sweep_adversarial.py --device cpu --num_seeds 10 --num_balls 10

# A harder pursuer, explicit seed list, write per-run CSV, show each seed
python3 sweep_adversarial.py --device cpu --seeds 0 1 2 3 4 --pursuer_speed 6 --csv results.csv --verbose
```

Example output:

```
model            n   mean(s)      std      min      max   #maxed
--------------------------------------------------------------------------
es2              8     33.34    17.20    14.54    72.66        0
mlp              8      4.70     5.16     0.52    16.56        0
transformer      8     19.40     6.01    12.16    31.02        0
```

`mean` is the average survival time in seconds; `std` is the population standard
deviation over the seeds; `#maxed` counts runs that reached `--max_steps` without
ever being caught.

### Flags

Sweep control:

- `--models` — space-separated list of checkpoint paths to compare (default:
  `pretrained/es2.pth pretrained/mlp.pth pretrained/transformer.pth`).
- `--seeds` — explicit list of seeds (e.g. `--seeds 0 1 2 7 42`). Overrides the
  two options below.
- `--num_seeds` / `--seed_start` — generate the seed range
  `[seed_start, seed_start + num_seeds)` (default `10` seeds from `0`).

Scenario settings (same meaning as in `evaluate_adversarial.py`):
`--num_balls`, `--max_speed`, `--max_steps`, `--pursuer_speed`,
`--pursuit_noise`, `--trigger_min`, `--trigger_max`, `--pursuer_index`,
`--device`, `--frequency`, `--width`, `--height`, `--sensing_range`, and the
Transformer options `--d_model` / `--nhead` / `--num_layers` /
`--dim_feedforward`.

Output:

- `--csv PATH` — write per-run rows (`model, seed, steps, seconds, reached_max`).
- `--verbose` — print every per-seed result, not just the summary.

> **Tip:** with only a handful of seeds the standard deviation is large. Use
> ≥ 10–20 seeds for a stable mean before quoting numbers, and sweep a few
> `--pursuer_speed` values (e.g. `5` / `7` / `9`) if you want a difficulty curve.
