"""Evaluate the history-aware goal agent in biased/unbiased goal worlds."""

import argparse
import math
import random

import torch

from model.goal_es2 import GoalEs2Model
from environment import (
    GOAL_RADIUS,
    CollisionTracker,
    lidar_scan,
    make_world,
    sample_goal,
)
from expert_goal import QUAD, sample_goal_biased


def in_quad(goal):
    return QUAD[0] <= goal[0] <= QUAD[2] and QUAD[1] <= goal[1] <= QUAD[3]


def run_seed(model, args, seed):
    random.seed(seed)
    player, balls = make_world(args.num_balls, args.width, args.height)
    for b in balls:
        if b is not player:
            b.dx *= args.speed_multiplier
            b.dy *= args.speed_multiplier
    tracker = CollisionTracker()
    prev = None
    trace = torch.tensor([args.width / 2, args.height / 2], dtype=torch.float32)
    goals = collisions = steps = 0
    spawns = {True: 0, False: 0}
    reached_n = {True: 0, False: 0}
    free_drift = []
    frequent_steps = []
    rare_steps = []

    def step(goal):
        nonlocal prev, steps
        for b in balls:
            if b is not player:
                b.move()
        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if prev is None:
            prev = d.copy()
        gdx, gdy = (goal[0] - player.x, goal[1] - player.y) if goal else (0.0, 0.0)
        hist = trace - torch.tensor([player.x, player.y], dtype=torch.float32)
        obs = torch.tensor(prev + d + [gdx, gdy] + hist.tolist(),
                           dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(obs)
        fx = min(max(action[0, 0].item(), -args.max_speed), args.max_speed)
        fy = min(max(action[0, 1].item(), -args.max_speed), args.max_speed)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        prev = d.copy()
        steps += 1
        tracker.update(player, balls)

    while steps < args.max_steps:
        if args.respawn_center:
            # fixation-start structure: every cycle begins at the center
            player.x, player.y = args.width // 2, args.height // 2
            tracker.in_contact.clear()
            prev = None
        start_dist = math.hypot(player.x - QUAD[0], player.y - QUAD[1])
        for _ in range(args.goal_free_steps):
            if steps >= args.max_steps:
                break
            before = math.hypot(player.x - (QUAD[0] + QUAD[2]) / 2,
                                player.y - (QUAD[1] + QUAD[3]) / 2)
            step(None)
            after = math.hypot(player.x - (QUAD[0] + QUAD[2]) / 2,
                               player.y - (QUAD[1] + QUAD[3]) / 2)
            free_drift.append(before - after)

        goal = sample_goal_biased(player, args)
        trace = (1 - model.eta_H.detach().cpu()) * trace + model.eta_H.detach().cpu() * torch.tensor(goal)
        spawns[in_quad(goal)] += 1
        region = frequent_steps if in_quad(goal) else rare_steps
        spawn_dist = max(math.hypot(goal[0] - player.x, goal[1] - player.y), 1e-6)
        used = 0
        reached = False
        while steps < args.max_steps and used < args.goal_timeout:
            step(goal)
            used += 1
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                goals += 1
                reached = True
                break
        if reached:
            region.append(used / spawn_dist * 100)
            reached_n[in_quad(goal)] += 1

    minutes = steps / 50 / 60
    collisions = tracker.count
    return {
        "goals_per_min": goals / minutes,
        "collisions_per_min": collisions / minutes,
        "free_drift_px_per_step": (
            sum(free_drift) / len(free_drift) if free_drift else float("nan")
        ),
        "frequent_steps_per_100px": sum(frequent_steps) / len(frequent_steps) if frequent_steps else float("nan"),
        "rare_steps_per_100px": sum(rare_steps) / len(rare_steps) if rare_steps else float("nan"),
        "frequent_reach_rate": reached_n[True] / max(spawns[True], 1),
        "rare_reach_rate": reached_n[False] / max(spawns[False], 1),
        "frequent_spawns": spawns[True],
        "rare_spawns": spawns[False],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/goal_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument(
        "--speed_multiplier",
        type=float,
        default=1.0,
        help="scale every ball's velocity (training speeds are 1-3 px/step)",
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--goal_free_steps", type=int, default=25)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--goal_bias", choices=["none", "quadrant"], default="quadrant")
    parser.add_argument("--bias_p", type=float, default=0.8)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--spawn_from_center",
        action="store_true",
        help="measure min_spawn_dist from the arena center, not the player",
    )
    parser.add_argument(
        "--respawn_center",
        action="store_true",
        help="teleport the player to the center at the start of every goal cycle",
    )
    parser.add_argument(
        "--disable_history",
        action="store_true",
        help="zero beta_H so the history field drops out of the sum",
    )
    args = parser.parse_args()

    model = GoalEs2Model(
        num_features=args.num_features,
        num_actions=args.num_actions,
        sensing_range=args.sensing_range,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    if args.disable_history:
        with torch.no_grad():
            model.beta_H.zero_()
        print("history DISABLED (beta_H forced to 0)")
    print(f"eta_H={model.eta_H.item():.4f} beta_H={model.beta_H.item():+.4f}")

    rows = [run_seed(model, args, args.random_seed + i) for i in range(args.num_seeds)]
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"{k}: {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
