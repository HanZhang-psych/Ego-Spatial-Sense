"""Anticipation (target selection-history) experiment on the frozen agent.

A presence-driven leaky trace of goal-spawn positions is bolted onto the
trained goal_es2 agent (no retraining): at every goal spawn the trace
centroid updates toward the spawn position at rate eta; at every step the
trace writes into the attention field exactly as a goal-presence channel
would — the model's own goal-field machinery evaluated at the centroid,
scaled by beta — on top of the obstacle field and the (real) phasic goal
field.  During goal-free periods a zero goal vector is fed, which exactly
zeroes the phasic goal field, so behavior is obstacle avoidance + trace
alone: drift is a pure readout of the trace through the frozen,
source-blind action head.

Blocks:
  exposure — goals spawn in a frequent quadrant with probability bias_p,
             each goal followed by a goal-free period;
  test     — goals spawn unbiased (statistics no longer contain the bias).

Arms: trace (beta > 0) vs. control (beta = 0), same seeds.

Predictions: (a) goal-free drift toward the frequent region (trace arm
only); (b) faster time-to-goal for frequent-region goals per unit
distance, collateral cost for rare-region goals; (c) both persist into
the test block and decay at rate eta.
"""

import argparse
import csv
import math
import random

import torch

from evaluate_reach_avoid import load_model
from reach_avoid_common import (
    GOAL_RADIUS,
    GOAL_WALL_MARGIN,
    CollisionTracker,
    lidar_scan,
    make_world,
    sample_goal,
)

# Frequent-goal quadrant (respecting the goal wall margin of 140)
QUAD = (140, 140, 400, 400)  # x0, y0, x1, y1
QUAD_CENTER = ((QUAD[0] + QUAD[2]) / 2, (QUAD[1] + QUAD[3]) / 2)


def in_quadrant(g):
    return QUAD[0] <= g[0] <= QUAD[2] and QUAD[1] <= g[1] <= QUAD[3]


def sample_goal_biased(player, args, biased):
    if biased and random.random() < args.bias_p:
        for _ in range(200):
            gx = random.uniform(QUAD[0], QUAD[2])
            gy = random.uniform(QUAD[1], QUAD[3])
            if math.hypot(gx - player.x, gy - player.y) >= 250:
                return gx, gy
        # Player is inside the quadrant: accept a closer goal there.
        return random.uniform(QUAD[0], QUAD[2]), random.uniform(QUAD[1], QUAD[3])
    return sample_goal(player, args.width, args.height)


def run_arm(model, args, seed, beta):
    random.seed(seed)
    torch.manual_seed(seed)
    player, balls = make_world(args.num_balls, args.width, args.height)
    tracker = CollisionTracker()
    prev = None
    # Trace state: leaky centroid of goal spawn positions, starts at center.
    cx, cy = args.width / 2, args.height / 2

    goal_rows = []  # per-goal: block, index, region, spawn_dist, steps, reached
    free_rows = []  # per goal-free period: block, index, mean dist to quad center

    def step(goal):
        nonlocal prev
        for b in balls:
            if b is not player:
                b.move()
        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if prev is None:
            prev = d.copy()
        gdx, gdy = (goal[0] - player.x, goal[1] - player.y) if goal else (0.0, 0.0)
        obs = torch.tensor(
            prev + d + [gdx, gdy], dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            of, gf = model.compute_fields(obs)
            if beta != 0.0:
                obs_h = torch.tensor(
                    prev + d + [cx - player.x, cy - player.y], dtype=torch.float32
                ).unsqueeze(0)
                _, gf_h = model.compute_fields(obs_h)
                field = of + gf + beta * gf_h
            else:
                field = of + gf
            a = model.sense_action_layers(field)
        fx = min(max(a[0][0].item(), -args.max_speed), args.max_speed)
        fy = min(max(a[0][1].item(), -args.max_speed), args.max_speed)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        prev = d.copy()
        tracker.update(player, balls)

    def run_block(block, num_goals, biased):
        nonlocal cx, cy
        for gi in range(num_goals):
            goal = sample_goal_biased(player, args, biased)
            # Presence-driven trace update at spawn (both arms track it;
            # only beta>0 injects it).
            cx = (1 - args.eta) * cx + args.eta * goal[0]
            cy = (1 - args.eta) * cy + args.eta * goal[1]
            spawn_px, spawn_py = player.x, player.y
            spawn_dist = math.hypot(goal[0] - player.x, goal[1] - player.y)
            steps = 0
            reached = False
            while steps < args.goal_timeout:
                step(goal)
                steps += 1
                if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                    reached = True
                    break
            goal_rows.append(
                [block, gi, "frequent" if in_quadrant(goal) else "rare",
                 round(spawn_dist, 1), steps, int(reached),
                 round(goal[0], 1), round(goal[1], 1),
                 round(spawn_px, 1), round(spawn_py, 1),
                 round(cx, 1), round(cy, 1)]
            )
            # Goal-free period: zero goal fed.
            dists = []
            for _ in range(args.goal_free_steps):
                step(None)
                dists.append(
                    math.hypot(player.x - QUAD_CENTER[0], player.y - QUAD_CENTER[1])
                )
            free_rows.append([block, gi, round(sum(dists) / len(dists), 1)])

    run_block("exposure", args.num_exposure_goals, biased=True)
    run_block("test", args.num_test_goals, biased=False)
    return goal_rows, free_rows, tracker.count


def summarize(goal_rows, free_rows, label):
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n--- {label} ---")
    for block in ["exposure", "test"]:
        fr = [r[2] for r in free_rows if r[0] == block]
        # split goal-free distances into halves to show build-up / decay
        h = len(fr) // 2
        print(
            f"[{block}] goal-free mean dist to frequent-quadrant center: "
            f"{mean(fr):.0f}px (first half {mean(fr[:h]):.0f}, "
            f"second half {mean(fr[h:]):.0f})"
        )
        for region in ["frequent", "rare"]:
            rows = [r for r in goal_rows if r[0] == block and r[2] == region and r[5]]
            spd = [r[4] / r[3] * 100 for r in rows if r[3] > 0]
            print(
                f"[{block}] {region:8s}: n={len(rows):3d}  "
                f"steps per 100px = {mean(spd):.2f}"
            )
        timeouts = sum(1 for r in goal_rows if r[0] == block and not r[5])
        if timeouts:
            print(f"[{block}] timeouts: {timeouts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/goal_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--bias_p", type=float, default=0.7)
    parser.add_argument("--num_exposure_goals", type=int, default=120)
    parser.add_argument("--num_test_goals", type=int, default=80)
    parser.add_argument("--goal_free_steps", type=int, default=100)
    parser.add_argument("--goal_timeout", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--out_prefix", type=str, default="anticipation")
    args = parser.parse_args()

    model, is_goal = load_model(args)
    assert is_goal and hasattr(model, "compute_fields")

    all_rows = {"trace": ([], [], 0), "control": ([], [], 0)}
    for i in range(args.num_seeds):
        seed = args.random_seed + i
        for arm, beta in [("trace", args.beta), ("control", 0.0)]:
            g, f, coll = run_arm(model, args, seed, beta)
            ag, af, ac = all_rows[arm]
            ag.extend(g)
            af.extend(f)
            all_rows[arm] = (ag, af, ac + coll)
            print(f"seed {seed} arm {arm}: collisions={coll}")

    for arm in ["trace", "control"]:
        g, f, coll = all_rows[arm]
        summarize(g, f, f"{arm} arm (beta={args.beta if arm == 'trace' else 0}, "
                        f"{args.num_seeds} seeds, total collisions={coll})")

    with open(f"{args.out_prefix}_goals.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "block", "goal_idx", "region", "spawn_dist", "steps",
                    "reached", "goal_x", "goal_y", "spawn_px", "spawn_py",
                    "trace_cx", "trace_cy"])
        for arm in ["trace", "control"]:
            for r in all_rows[arm][0]:
                w.writerow([arm] + r)
    with open(f"{args.out_prefix}_goalfree.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "block", "goal_idx", "mean_dist_to_quad_center"])
        for arm in ["trace", "control"]:
            for r in all_rows[arm][1]:
                w.writerow([arm] + r)
    print(f"\nRows written to {args.out_prefix}_goals.csv / {args.out_prefix}_goalfree.csv")


if __name__ == "__main__":
    main()
