"""Trial-based reach-avoid evaluation for GoalHistoryEs2Model.

Each trial resets the player, obstacles, and unbiased target.  The agent
tries to reach the target before colliding or timing out.  A world-space
history trace is carried across successful trials and supplied to the model
as an additional spatial source.
"""

import argparse
import math
import random

import torch

from model.goal_history_es2 import GoalHistoryEs2Model
from reach_avoid_common import (
    GOAL_RADIUS,
    CollisionTracker,
    lidar_scan,
    make_world,
    sample_goal,
)


def clamp_action(action, max_speed):
    fx = min(max(action[0].item(), -max_speed), max_speed)
    fy = min(max(action[1].item(), -max_speed), max_speed)
    return fx, fy


def run_trial(model, args, trace):
    player, balls = make_world(args.num_balls, args.width, args.height)
    goal = sample_goal(player, args.width, args.height, min_dist=args.min_spawn_dist)
    tracker = CollisionTracker()
    previous_scan = None

    for step in range(1, args.max_steps_per_trial + 1):
        for ball in balls:
            if ball is not player:
                ball.move()

        distances, _ = lidar_scan(
            player, balls, args.width, args.height, args.num_features
        )
        if previous_scan is None:
            previous_scan = distances.copy()

        hist = trace - torch.tensor([player.x, player.y], dtype=torch.float32)
        obs = torch.tensor(
            previous_scan
            + distances
            + [goal[0] - player.x, goal[1] - player.y]
            + hist.tolist(),
            dtype=torch.float32,
            device=args.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action = model(obs)[0]

        fx, fy = clamp_action(action, args.max_speed)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        previous_scan = distances.copy()

        if tracker.update(player, balls):
            return "collision", step, trace

        if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
            eta = model.eta_H.detach().cpu()
            goal_xy = torch.tensor(goal, dtype=torch.float32)
            trace = (1 - eta) * trace + eta * goal_xy
            return "success", step, trace

    return "timeout", args.max_steps_per_trial, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/goal_history_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps_per_trial", type=int, default=1000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--disable_history",
        action="store_true",
        help="zero beta_H so the history field drops out of the sum",
    )
    args = parser.parse_args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    model = GoalHistoryEs2Model(
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

    trace = torch.tensor([args.width / 2, args.height / 2], dtype=torch.float32)
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    steps_by_outcome = {"success": [], "collision": [], "timeout": []}

    for _ in range(args.num_trials):
        outcome, steps, trace = run_trial(model, args, trace)
        outcomes[outcome] += 1
        steps_by_outcome[outcome].append(steps)

    total = sum(outcomes.values())
    print(f"eta_H={model.eta_H.item():.4f}")
    for key in ["success", "collision", "timeout"]:
        rate = outcomes[key] / max(total, 1)
        steps = steps_by_outcome[key]
        mean_steps = sum(steps) / len(steps) if steps else 0.0
        print(f"{key}: {outcomes[key]} ({rate:.3f}), mean_steps={mean_steps:.1f}")


if __name__ == "__main__":
    main()
