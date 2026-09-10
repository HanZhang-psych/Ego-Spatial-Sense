# Goal-Directed Reach-Avoid Extension

The original escaping-ball task tests only *reactive* collision avoidance — the ego
ball has no destination, so the goal component of the ego state `s_e` is degenerate
("survive"), symmetric in all directions. Ego spatial sense, however, is defined as a
spatial attention field `F(q | s_e)` generated a priori from the ego's own state **and
task/goal** (top-down attention). This extension adds a minimal sandbox that makes the
goal non-trivial: the ego must **reach a destination while avoiding the moving balls**.

It stays a REACTIVE, single-step spatial-pressure benchmark: an open arena with a
destination and moving balls — no corridors, dead-ends or path planning.

`evaluate.py` and the original models/data are untouched.

## Task

Same arena, LiDAR, and ball dynamics as `evaluate.py`, plus a goal:

- A goal position is sampled at least 140 px from the walls and 250 px from the ego.
- The ego "reaches" the goal within 30 px; a new goal then spawns (continuous
  throughput). With `--single_goal` the episode instead ends at the first goal.
- Collisions do **not** end evaluation runs; collision *events* are counted
  (an overlap lasting several frames counts once).
- **Primary metric: goals-reached-per-minute vs. collisions-per-minute** — a
  Pareto trade-off. (Single-shot form: steps-to-destination + collisions.)

## Observation & model

The observation is the usual two concatenated scans plus the relative goal vector
(pixels): `prev_scan(360) | scan(360) | goal_dx, goal_dy` → 722 features. This changes
the input dimension, so the original 720-input checkpoints cannot be reused for the
goal-conditioned agent; `pretrained/goal_es2.pth` is trained from scratch.

`model/goal_es2.py` (`GoalEs2Model`) keeps the ES2 `SpatialSenseBlock` intact for the
obstacle field and adds a top-down goal field: per-ray alignment
`cos(theta_i - goal_bearing)` scaled by a learned gain conditioned on goal distance.
The two fields are summed into a single attention field (asymmetric toward the goal
sector, inspectable via `compute_fields`) which the usual action layers map to
`(fx, fy)`.

## Expert

`expert_goal.py` extends the potential-field expert of `expert.py` with the textbook
attractive term: constant-magnitude pull toward the goal (tapered near the goal),
on top of the unchanged inverse-square repulsion from every LiDAR ray. It writes
demonstrations to `dataset/data_goal.csv` with columns
`fx, fy, scan_0..359, goal_dx, goal_dy, episode` (the episode id keeps the dataloader
from pairing frames across resets).

## Files

| File | Purpose |
| --- | --- |
| `reach_avoid_common.py` | Shared env: balls, LiDAR, goal sampling, expert policy, collision-event counting |
| `expert_goal.py` | Goal-directed expert; generates `dataset/data_goal.csv` |
| `dataset/dataloader_goal.py` | Goal-augmented dataset (episode-boundary aware) |
| `model/goal_es2.py` | `GoalEs2Model` (722-input goal-conditioned ES2) |
| `train_goal_es2.py` | Imitation training (same alternating-k schedule as `train_es2.py`) |
| `evaluate_reach_avoid.py` | Closed-loop evaluation, multi-seed, Pareto metrics |
| `goal_swap_probe.py` | Goal-swap diagnostic (see below) |
| `model/goal_mlp.py`, `model/goal_transformer.py` | Goal-conditioned MLP / Transformer baselines (same 722-feature input) |
| `train_goal_baseline.py` | Trainer for the two baselines (`--model mlp|transformer`) |
| `selection_history_probe.py` | Selection-history probe: hazard-biased training vs. unbiased control (see below) |
| `anticipation_experiment.py` | Online presence-driven target-history trace on the frozen agent: goal-biased exposure with goal-free periods, unbiased test block, β=0 control arm |

## Reproduce

```bash
# 1. Generate demonstrations (headless, ~1 min)
python expert_goal.py --num_episodes 20 --max_steps_per_episode 3000

# 2. Train the goal-conditioned model (CPU is fine)
python train_goal_es2.py --device cpu

# 3. Evaluate: goal agent vs. reactive baseline in the same reach-avoid env
python evaluate_reach_avoid.py --model_path pretrained/goal_es2.pth --device cpu
python evaluate_reach_avoid.py --model_path pretrained/es2.pth --device cpu

# watch it (draws the goal in green)
python evaluate_reach_avoid.py --model_path pretrained/goal_es2.pth --device cpu --num_seeds 1 --render

# 4. Goal-swap diagnostic (after training)
python goal_swap_probe.py --model_path pretrained/goal_es2.pth --device cpu
python goal_swap_probe.py --model_path pretrained/es2.pth --device cpu   # control
```

The baseline `es2.pth`/`mlp.pth`/`transformer.pth` models run in the same environment
but never see the goal — they are the reactive frontier anchor (near-zero goals/min,
low collisions/min). The claim to verify is that the goal-directed agent **dominates
the frontier** (large goals/min at comparable collisions/min), not that it merely
survives.

`evaluate_reach_avoid.py` also reports `field_asym` for the goal model: the mean
attention-field difference between rays within ±60° of the goal bearing and rays
pointing away — nonzero means the visualizable field is asymmetric toward the goal
sector, as the top-down attention story predicts.

## Goal-swap diagnostic (`goal_swap_probe.py`)

A causal ablation verifying the agent actually *uses* the goal input, in two forms:

- **Failure form** — feed a goal different from the scored one (`opposite`: mirrored
  through the arena center; `random`: independent, resampled when reached) and score
  against the TRUE goal. Sensitivity = true-goals/min(correct) − true-goals/min(wrong),
  reported as a continuous effect size. Goal-conditioned agent: large gap. Goal-blind
  agent: ~zero gap. Wrong goals are mirrored/random and averaged over several seeds so
  an accidental alignment with the true goal cannot masquerade as goal-blindness.
- **Positive form (stronger)** — score against the FED goal across many randomly
  assigned goals: a goal-conditioned agent reaches whatever goal it is handed; a
  goal-blind one reaches fed goals only at chance rate.
- **Entanglement check** — collisions/min per condition. The goal input should steer
  *direction*, not avoidance, so collision rates should stay comparable across
  conditions; if avoidance also collapses under a wrong goal, that entanglement is
  itself a finding.

## Architecture comparison

`train_goal_baseline.py --model mlp` / `--model transformer` trains
goal-conditioned MLP / Transformer baselines on the same demonstrations, and
`evaluate_reach_avoid.py` / `goal_swap_probe.py` dispatch on the checkpoint
name (`goal_mlp.pth`, `goal_transformer.pth`). Headline result: all three use
the goal, but only ES2 (and the Transformer) keep goal-steering decomposed
from avoidance under the goal-swap manipulation — the MLP's collision rate
doubles when fed a wrong goal — and only ES2 matches the expert's
throughput/safety frontier point.

## Selection-history experiment

`expert_goal.py --hazard_side left` generates demonstrations in a world where
80% of the balls are confined to one half of the arena;
`selection_history_probe.py` then tests a model trained on them in the
STANDARD world against the unbiased control (learned per-ray gains, baseline
field by sector, clearance and goal-approach asymmetries). Result so far: an
informative null — behavior cloning from a memoryless expert does not produce
history effects; see RESULTS_reach_avoid.md for numbers and interpretation.

## Training provenance (what learns, from what, and when)

Four layers, deliberately separated:

1. **Expert — never trained.** A hand-designed potential-field control law
   (`reach_avoid_common.py`): inverse-square repulsion from every LiDAR ray
   plus constant attraction to the goal. Exists to generate demonstrations
   and as the performance reference.
2. **Demonstrations — generated once per condition** by `expert_goal.py`
   (~12k rows standard world; ~11k hazard-biased world).
3. **Networks — behavior-cloned once, offline; frozen ever after.**
   Supervised MSE regression onto the expert's actions. `GoalEs2Model`
   keeps the original SpatialSenseBlock *identical* and adds one goal
   channel (input 720 → 722; learned distance gain × per-ray cosine
   alignment summed into the field); it is trained from scratch (the
   original 720-input checkpoints cannot accept a goal and serve only as
   goal-blind baselines). No reward, no RL, no fine-tuning, no test-time
   weight updates anywhere.
4. **Selection-history trace — not trained at all.** Runtime *state*, not
   weights: a leaky, exponentially weighted record of goal spawn positions
   (presence-driven, rate η=0.05/spawn), injected into the frozen network's
   field through the model's own goal-field machinery at fixed weight
   β=0.15 ("ghost goals where goals have tended to be, at 15% strength").
   η and β are hand-set constants, not fitted.

The separation is load-bearing: behavior cloning of a memoryless expert
provably cannot contain history (the hazard-null result), so history must
— and does — live in the runtime layer; conversely the trace has no
competence without the frozen policy. The trace steers a network that
never saw a trace during training because everything communicates through
the one shared field (the source-blindness result).

`render_anticipation_gif.py` renders the learning visualization (early vs.
late trials, fixation-start structure; note the excursion settles at the
region's *edge* — the β=0.15 plateau equilibrium documented in
RESULTS_reach_avoid.md).

## Results (5 seeds × 6000 steps, CPU)

See `RESULTS_reach_avoid.md` (generated by the runs above).
