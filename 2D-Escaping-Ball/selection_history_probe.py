"""Selection-history probe: does hazard-biased experience leave a lasting,
direction-specific trace in the learned field — and in behavior?

An agent is trained on demonstrations from a hazard-biased world (most balls
confined to one half of the arena, see expert_goal.py --hazard_side) and then
tested in the STANDARD unbiased world, against an agent trained unbiased.
This is the locomotor analog of learned spatial distractor suppression
(Wang & Theeuwes): the statistics of past threats are no longer valid at
test time, so any asymmetry is history, not stimulus.

Reported, for each model:
  1. Learned per-ray k profile of the SpatialSenseBlock by sector
     (the architecture's selection-history slot).
  2. Baseline attention field by sector in a neutral scene (ego at center,
     no balls, goal placed orthogonal to the hazard axis).
  3. Behavior in the unbiased world: mean clearance to the nearest ball on
     the hazard side vs. the safe side, mean ego x position, and
     goals/min + collisions/min (competence check).
  4. Collateral cost: mean steps between goal arrivals for goals in the
     hazard-side half vs. the other half.

Usage:
    python selection_history_probe.py \
        --model_path pretrained/goal_es2_hazard.pth \
        --control_path pretrained/goal_es2.pth --device cpu
"""

import argparse
import math
import random

import torch

from evaluate_reach_avoid import add_common_args, load_model
from reach_avoid_common import (
    GOAL_RADIUS,
    CollisionTracker,
    lidar_scan,
    make_world,
    sample_goal,
)

WEST = [i for i in range(360) if 135 <= i <= 225]
EAST = [i for i in range(360) if i <= 45 or i >= 315]
NORTH = [i for i in range(360) if 225 < i < 315]  # y is down in screen coords
SOUTH = [i for i in range(360) if 45 < i < 135]


def sector_means(vec):
    return {
        "west": sum(vec[i] for i in WEST) / len(WEST),
        "east": sum(vec[i] for i in EAST) / len(EAST),
        "north": sum(vec[i] for i in NORTH) / len(NORTH),
        "south": sum(vec[i] for i in SOUTH) / len(SOUTH),
    }


def k_profile(model):
    return sector_means(model.spatial_sense_block.k.detach().tolist())


def baseline_field(model, args):
    """Combined field for the ego alone at the arena center (walls only),
    with the goal due south so the goal field is orthogonal to east/west."""
    random.seed(0)
    player, balls = make_world(0, args.width, args.height)
    d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
    goal = (args.width / 2, args.height - 150)
    obs = torch.tensor(
        d + d + [goal[0] - player.x, goal[1] - player.y], dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        of, gf = model.compute_fields(obs)
    return sector_means((of + gf)[0].tolist())


def behavior_probe(model, args, seed):
    random.seed(seed)
    player, balls = make_world(args.num_balls, args.width, args.height)
    goal = sample_goal(player, args.width, args.height)
    tracker = CollisionTracker()
    prev = None
    goals_reached = 0
    clear_w, clear_e = [], []
    xs = []
    goal_steps = {"left": [], "right": []}
    steps_since_goal = 0

    for step in range(args.max_steps):
        for b in balls:
            if b is not player:
                b.move()
        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if prev is None:
            prev = d.copy()
        obs = torch.tensor(
            prev + d + [goal[0] - player.x, goal[1] - player.y], dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            a = model(obs)
        fx = min(max(a[0][0].item(), -args.max_speed), args.max_speed)
        fy = min(max(a[0][1].item(), -args.max_speed), args.max_speed)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        prev = d.copy()
        tracker.update(player, balls)
        steps_since_goal += 1
        xs.append(player.x)

        west_d = [
            math.hypot(b.x - player.x, b.y - player.y)
            for b in balls
            if b is not player and b.x < player.x
        ]
        east_d = [
            math.hypot(b.x - player.x, b.y - player.y)
            for b in balls
            if b is not player and b.x >= player.x
        ]
        if west_d:
            clear_w.append(min(west_d))
        if east_d:
            clear_e.append(min(east_d))

        if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
            goals_reached += 1
            side = "left" if goal[0] < args.width / 2 else "right"
            goal_steps[side].append(steps_since_goal)
            steps_since_goal = 0
            goal = sample_goal(player, args.width, args.height)

    minutes = args.max_steps / 50 / 60
    return {
        "goals_per_min": goals_reached / minutes,
        "collisions_per_min": tracker.count / minutes,
        "clearance_west": sum(clear_w) / len(clear_w) if clear_w else float("nan"),
        "clearance_east": sum(clear_e) / len(clear_e) if clear_e else float("nan"),
        "mean_x": sum(xs) / len(xs),
        "steps_per_left_goal": (
            sum(goal_steps["left"]) / len(goal_steps["left"])
            if goal_steps["left"]
            else float("nan")
        ),
        "steps_per_right_goal": (
            sum(goal_steps["right"]) / len(goal_steps["right"])
            if goal_steps["right"]
            else float("nan")
        ),
        "n_left_goals": len(goal_steps["left"]),
        "n_right_goals": len(goal_steps["right"]),
    }


def report(name, model, args):
    print(f"\n===== {name} =====")
    kp = k_profile(model)
    print("k by sector:        " + "  ".join(f"{s}={v:.5f}" for s, v in kp.items()))
    bf = baseline_field(model, args)
    print("baseline field:     " + "  ".join(f"{s}={v:+.4f}" for s, v in bf.items()))

    keys = [
        "goals_per_min", "collisions_per_min", "clearance_west", "clearance_east",
        "mean_x", "steps_per_left_goal", "steps_per_right_goal",
        "n_left_goals", "n_right_goals",
    ]
    acc = {k: [] for k in keys}
    for i in range(args.num_seeds):
        r = behavior_probe(model, args, args.random_seed + i)
        for k in keys:
            acc[k].append(r[k])
    mean = {k: sum(v) / len(v) for k, v in acc.items()}
    print(
        f"behavior (unbiased world, {args.num_seeds} seeds x {args.max_steps} steps):\n"
        f"  goals/min={mean['goals_per_min']:.2f}  collisions/min={mean['collisions_per_min']:.2f}  "
        f"mean_x={mean['mean_x']:.1f}\n"
        f"  nearest-ball clearance: west={mean['clearance_west']:.1f}px  "
        f"east={mean['clearance_east']:.1f}px  "
        f"(west-east={mean['clearance_west'] - mean['clearance_east']:+.1f})\n"
        f"  steps per goal: left-half={mean['steps_per_left_goal']:.1f} "
        f"(n={mean['n_left_goals']:.1f}/seed)  right-half={mean['steps_per_right_goal']:.1f} "
        f"(n={mean['n_right_goals']:.1f}/seed)"
    )
    return kp, bf, mean


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--control_path", type=str, default="pretrained/goal_es2.pth",
        help="Unbiased-trained model for comparison",
    )
    args = parser.parse_args()

    model, is_goal = load_model(args)
    assert is_goal and hasattr(model, "compute_fields"), (
        "selection_history_probe expects a goal-conditioned ES2 model"
    )
    report(f"{args.model_path} (hazard-trained)", model, args)

    control_args = argparse.Namespace(**vars(args))
    control_args.model_path = args.control_path
    control, _ = load_model(control_args)
    report(f"{args.control_path} (control)", control, args)

    print(
        "\nInterpretation: a selection-history effect appears as a west/east "
        "asymmetry in k, baseline field, or clearance for the hazard-trained "
        "model that the control lacks — measured in a world whose statistics "
        "no longer contain the bias."
    )


if __name__ == "__main__":
    main()
