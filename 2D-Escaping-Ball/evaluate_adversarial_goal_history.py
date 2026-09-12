"""Adversarial evaluation for GoalHistoryEs2Model on the continuous goal task.

The original adversarial test (``evaluate_adversarial.py``) is for the
goal-blind escape models: scan-only input, survival time as the metric.
This variant transplants its stressor — at a seeded trigger time, one ball
switches to erratic pursuit of the ego (homing heading + bounded angular
noise, speed capped at the ego's max so escape stays possible) — into the
continuous reach-avoid task that the goal/history models are trained and
evaluated on.  The question changes from "can it survive?" to "can it keep
reaching goals while being hunted?", so the metrics are goals/min and
collisions/min, split into the pre-trigger and post-trigger phases of each
seed.

Example (model of record):
    python3 evaluate_adversarial_goal_history.py \
      --model_path pretrained/goal_history_geometric_trial_unbiased_360_150_nohist.pth \
      --disable_history --num_seeds 3 --max_steps 6000
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


def pursuer_move(ball, target, speed, noise, width, height):
    """Erratic homing step (evaluate_adversarial.py semantics)."""
    ball.prev_x, ball.prev_y = ball.x, ball.y
    angle = math.atan2(target.y - ball.y, target.x - ball.x)
    angle += random.uniform(-noise, noise)
    ball.x += speed * math.cos(angle)
    ball.y += speed * math.sin(angle)
    ball.x = max(ball.radius, min(width - ball.radius, ball.x))
    ball.y = max(ball.radius, min(height - ball.radius, ball.y))


def run_seed(model, args, seed):
    random.seed(seed)
    player, balls = make_world(args.num_balls, args.width, args.height)
    tracker = CollisionTracker()
    trace = torch.tensor([args.width / 2, args.height / 2], dtype=torch.float32)
    goal = sample_goal(player, args.width, args.height, min_dist=args.min_spawn_dist)
    prev = None

    pursuer_speed = args.pursuer_speed if args.pursuer_speed > 0 else args.max_speed
    others = [b for b in balls if b is not player]
    pursuer = others[random.randrange(len(others))]
    trig_lo = int(args.trigger_min * 50)
    trig_hi = max(int(args.trigger_max * 50), trig_lo)
    trigger_step = random.randint(trig_lo, trig_hi)

    phase = {False: {"goals": 0, "steps": 0},
             True: {"goals": 0, "steps": 0}}
    pre_coll_final = None
    goal_steps = 0

    for step in range(args.max_steps):
        pursuing = step >= trigger_step
        if pursuing and pre_coll_final is None:
            pre_coll_final = tracker.count

        for b in balls:
            if b is player:
                continue
            if pursuing and b is pursuer:
                pursuer_move(b, player, pursuer_speed, args.pursuit_noise,
                             args.width, args.height)
            else:
                b.move()

        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if prev is None:
            prev = d.copy()
        hist = trace - torch.tensor([player.x, player.y], dtype=torch.float32)
        obs = torch.tensor(
            prev + d + [goal[0] - player.x, goal[1] - player.y] + hist.tolist(),
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            a = model(obs)[0]
        fx = min(max(a[0].item(), -args.max_speed), args.max_speed)
        fy = min(max(a[1].item(), -args.max_speed), args.max_speed)
        player.x = max(player.radius, min(args.width - player.radius, player.x + int(fx)))
        player.y = max(player.radius, min(args.height - player.radius, player.y + int(fy)))
        prev = d.copy()
        tracker.update(player, balls)
        phase[pursuing]["steps"] += 1
        goal_steps += 1

        reached = math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS
        if reached or goal_steps >= args.goal_timeout:
            if reached:
                phase[pursuing]["goals"] += 1
                eta = model.eta_H.detach().cpu()
                trace = (1 - eta) * trace + eta * torch.tensor(goal, dtype=torch.float32)
            goal = sample_goal(player, args.width, args.height,
                               min_dist=args.min_spawn_dist)
            goal_steps = 0

    if pre_coll_final is None:
        pre_coll_final = tracker.count
    colls = {"pre": pre_coll_final, "post": tracker.count - pre_coll_final}
    out = {}
    for name, ph in (("pre", phase[False]), ("post", phase[True])):
        minutes = ph["steps"] / 50 / 60
        out[name] = (ph["goals"] / minutes if minutes else float("nan"),
                     colls[name] / minutes if minutes else float("nan"),
                     ph["steps"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="pretrained/goal_history_geometric_trial_unbiased_360_150_nohist.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--pursuer_speed", type=float, default=0.0,
                        help="<=0 uses --max_speed (task stays solvable)")
    parser.add_argument("--pursuit_noise", type=float, default=0.6)
    parser.add_argument("--trigger_min", type=float, default=5.0,
                        help="earliest pursuit start (s)")
    parser.add_argument("--trigger_max", type=float, default=30.0,
                        help="latest pursuit start (s)")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--disable_history", action="store_true",
                        help="zero beta_H so the history field drops out of the sum")
    args = parser.parse_args()

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

    rows = [run_seed(model, args, args.random_seed + i) for i in range(args.num_seeds)]
    for name in ("pre", "post"):
        g = [r[name][0] for r in rows]
        c = [r[name][1] for r in rows]
        s = [r[name][2] for r in rows]
        label = "before pursuit" if name == "pre" else "during pursuit"
        print(f"{label}: {sum(g)/len(g):.1f} goals/min, "
              f"{sum(c)/len(c):.1f} collisions/min "
              f"(mean {sum(s)/len(s):.0f} steps/seed)")


if __name__ == "__main__":
    main()
