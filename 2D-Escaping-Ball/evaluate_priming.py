"""Target-location priming: a mechanism-sufficiency demonstration.

The claim (mirroring Sec. 9c of the visual-search tutorial): adding a
history field to a competent goal-seeking policy is SUFFICIENT to produce
the human target-location priming signature - facilitation when the goal
repeats its location, cost when it changes, both scaling with the history
strength and both absent at beta_H = 0.

Construction (no arbitrary weights): the remembered location (the previous
goal, a one-back trace) is treated as a faint goal.  Its field is built
with the TRAINED goal-field machinery - `geometric_field(trace - player,
goal_gain)` - scaled by a single swept scalar BETA.  The base policy is a
trained history-disabled checkpoint; its own history channel contributes
nothing (beta_H = 0 in the checkpoint).  Nothing is trained here.

Trial structure (fixation-start): teleport to center; a goal-free
anticipation window (drift toward the remembered location is the
mechanism's second signature, logged from the same runs); then the goal
appears - on REPEAT trials at the previous goal's location, on CHANGE
trials at the same eccentricity but rotated 90-270 degrees away, so path
lengths are matched.  Measured: steps to goal and initial heading error.

Goal-vs-history precedence (default GATED): history is anticipation and
operates only in the absence of a goal; once a goal is visible it
overrides history (the history field is zeroed during pursuit).  Priming
then comes entirely from pre-positioning - the drift moved the agent
closer to a repeated goal and away from a changed one before it appeared.
--ungated restores the original always-on sum (history also tugs during
pursuit, producing capture at high beta).
"""

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
)


def run_seed(model, beta, args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cx, cy = args.width / 2, args.height / 2
    player, balls = make_world(args.num_balls, args.width, args.height)
    tracker = CollisionTracker()

    def act(goal_vec, hist_vec):
        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if act.prev is None:
            act.prev = d
        obs = torch.tensor(act.prev + d + list(goal_vec),
                           dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            of, gf = model.compute_fields(obs)
            hv = torch.tensor(hist_vec, dtype=torch.float32).unsqueeze(0)
            hf = model.geometric_field(hv, model.goal_gain)
            a = model.sense_action_layers(of + gf + beta * hf)[0]
        act.prev = d
        fx = min(max(a[0].item(), -args.max_speed), args.max_speed)
        fy = min(max(a[1].item(), -args.max_speed), args.max_speed)
        return fx, fy

    def step_world():
        for b in balls:
            if b is not player:
                b.move()

    def move(fx, fy):
        player.x = max(player.radius, min(args.width - player.radius, player.x + int(fx)))
        player.y = max(player.radius, min(args.height - player.radius, player.y + int(fy)))
        tracker.update(player, balls)

    prev_goal = None
    rows = []
    for trial in range(args.num_trials):
        player.x, player.y = int(cx), int(cy)
        tracker.in_contact.clear()
        act.prev = None

        # anticipation window: no goal; the remembered location is the only field
        drift = float("nan")
        if prev_goal is not None and args.goal_free_steps > 0:
            d0 = math.hypot(prev_goal[0] - player.x, prev_goal[1] - player.y)
            for _ in range(args.goal_free_steps):
                step_world()
                fx, fy = act((0.0, 0.0),
                             (prev_goal[0] - player.x, prev_goal[1] - player.y))
                move(fx, fy)
            d1 = math.hypot(prev_goal[0] - player.x, prev_goal[1] - player.y)
            drift = (d0 - d1) / args.goal_free_steps  # px/step toward remembered loc

        # goal placement
        if prev_goal is None:
            ecc = random.uniform(args.ecc_min, args.ecc_max)
            ang = random.uniform(0, 2 * math.pi)
            goal = (cx + ecc * math.cos(ang), cy + ecc * math.sin(ang))
            kind = "first"
        else:
            ecc = math.hypot(prev_goal[0] - cx, prev_goal[1] - cy)
            ang = math.atan2(prev_goal[1] - cy, prev_goal[0] - cx)
            if trial % 2 == 1:
                kind = "repeat"
                goal = prev_goal
            else:
                kind = "change"
                ang = ang + math.radians(random.uniform(90, 270))
                goal = (cx + ecc * math.cos(ang), cy + ecc * math.sin(ang))

        # pursue.  Gated (default): the visible goal OVERRIDES history -
        # the history field is zeroed, so priming comes entirely from the
        # pre-positioning drift of the anticipation window.  --ungated
        # keeps history active during pursuit (the original construction:
        # adds an in-flight tug, and capture at high beta).
        head_err = float("nan")
        used = None
        for t in range(1, args.goal_timeout + 1):
            step_world()
            fx, fy = act((goal[0] - player.x, goal[1] - player.y),
                         (0.0, 0.0) if (prev_goal is None or not args.ungated)
                         else (prev_goal[0] - player.x, prev_goal[1] - player.y))
            if t == 1 and (fx or fy):
                gang = math.atan2(goal[1] - player.y, goal[0] - player.x)
                aang = math.atan2(fy, fx)
                head_err = abs(math.degrees(
                    (aang - gang + math.pi) % (2 * math.pi) - math.pi))
            move(fx, fy)
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                used = t
                break
        rows.append((kind, used, head_err, drift))
        prev_goal = goal
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="pretrained/goal_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--goal_free_steps", type=int, default=50)
    parser.add_argument("--ecc_min", type=float, default=250.0)
    parser.add_argument("--ecc_max", type=float, default=330.0)
    parser.add_argument("--num_trials", type=int, default=101)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.4])
    parser.add_argument("--ungated", action="store_true",
                        help="keep the history field active while a goal is "
                             "visible (the original always-on sum)")
    args = parser.parse_args()

    model = GoalEs2Model(
        num_features=args.num_features,
        num_actions=args.num_actions,
        sensing_range=args.sensing_range,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    assert model.beta_H.item() == 0.0, "base policy must be a history-disabled checkpoint"

    print(f"{'beta':>5} | {'repeat steps':>12} {'change steps':>12} {'priming':>8} | "
          f"{'rep head-err':>12} {'chg head-err':>12} | {'drift px/step':>13} | "
          f"{'timeouts':>8}")
    for beta in args.betas:
        rows = []
        for i in range(args.num_seeds):
            rows += run_seed(model, beta, args, args.random_seed + i)
        def agg(kind, idx):
            v = [r[idx] for r in rows if r[0] == kind and r[idx] is not None
                 and not (isinstance(r[idx], float) and math.isnan(r[idx]))]
            return sum(v) / len(v) if v else float("nan")
        rs, cs = agg("repeat", 1), agg("change", 1)
        rh, ch = agg("repeat", 2), agg("change", 2)
        dr = agg("repeat", 3) if not math.isnan(agg("repeat", 3)) else agg("change", 3)
        drift_all = [r[3] for r in rows if not math.isnan(r[3])]
        dr = sum(drift_all) / len(drift_all) if drift_all else float("nan")
        to = sum(1 for r in rows if r[1] is None)
        print(f"{beta:>5.2f} | {rs:>12.1f} {cs:>12.1f} {cs - rs:>+8.1f} | "
              f"{rh:>12.1f} {ch:>12.1f} | {dr:>13.2f} | {to:>8d}")


if __name__ == "__main__":
    main()
