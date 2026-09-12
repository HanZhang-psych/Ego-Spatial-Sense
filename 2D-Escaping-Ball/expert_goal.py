"""Generate demonstrations for a goal-history-aware reach-avoid expert.

Rows include enough state to train a world-space leaky goal trace:
current scans, current goal vector, player position, and goal-spawn events.
During goal-free anticipation periods, the expert is attracted to its
world-space goal-history trace instead of a visible goal.
"""

import argparse
import csv
import math
import random

from environment import (
    GOAL_RADIUS,
    CollisionTracker,
    expert_goal_action,
    lidar_scan,
    make_world,
    sample_goal,
)

QUAD = (140, 140, 400, 400)


class _CenterRef:
    def __init__(self, args):
        self.x, self.y = args.width / 2, args.height / 2


def sample_goal_biased(player, args):
    # --spawn_from_center: the minimum-spawn-distance rule is measured from
    # the arena center (the respawn point), not the agent.  Goal positions
    # are then defined relative to the arena - like a search display - so an
    # agent that pre-positions near the frequent region genuinely starts
    # closer.  Measured from the player, drifting toward the region just
    # pushes eligible spawns to its far side and cancels the benefit.
    ref = _CenterRef(args) if getattr(args, "spawn_from_center", False) else player
    if args.goal_bias == "quadrant" and random.random() < args.bias_p:
        for _ in range(200):
            gx = random.uniform(QUAD[0], QUAD[2])
            gy = random.uniform(QUAD[1], QUAD[3])
            if math.hypot(gx - ref.x, gy - ref.y) >= args.min_spawn_dist:
                return gx, gy
    return sample_goal(
        ref, args.width, args.height, min_dist=args.min_spawn_dist
    )


def run_episode(args, writer, episode_id):
    player, balls = make_world(args.num_balls, args.width, args.height)
    tracker = CollisionTracker()
    trace_x, trace_y = args.width / 2, args.height / 2
    goals_reached = 0
    step_id = 0
    respawn_flag = [0]   # 1 on the first recorded row after a center respawn

    def write_step(goal, goal_spawn):
        nonlocal step_id
        for ball in balls:
            if ball is not player:
                ball.move()
        distances, _ = lidar_scan(
            player, balls, args.width, args.height, args.num_features
        )
        if goal is None:
            goal_dx, goal_dy = 0.0, 0.0
            drive_dx, drive_dy = trace_x - player.x, trace_y - player.y
        else:
            goal_dx, goal_dy = goal[0] - player.x, goal[1] - player.y
            drive_dx, drive_dy = goal_dx, goal_dy

        fx, fy = expert_goal_action(distances, drive_dx, drive_dy, args.num_features)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        tracker.update(player, balls)

        if step_id % args.record_every == 0:
            writer.writerow(
                [fx, fy]
                + distances
                + [
                    goal_dx, goal_dy,
                    trace_x - player.x, trace_y - player.y,
                    player.x, player.y,
                    goal[0] if goal else 0.0,
                    goal[1] if goal else 0.0,
                    int(goal is not None),
                    int(goal_spawn),
                    respawn_flag[0],
                    episode_id,
                ]
            )
            respawn_flag[0] = 0
        step_id += 1

    while step_id < args.max_steps_per_episode:
        if args.respawn_center:
            # fixation-start structure: every cycle begins at the center
            player.x, player.y = args.width // 2, args.height // 2
            tracker.in_contact.clear()
            respawn_flag[0] = 1
        for _ in range(args.goal_free_steps):
            if step_id >= args.max_steps_per_episode:
                break
            write_step(None, False)

        goal = sample_goal_biased(player, args)
        trace_x = (1 - args.expert_eta) * trace_x + args.expert_eta * goal[0]
        trace_y = (1 - args.expert_eta) * trace_y + args.expert_eta * goal[1]
        spawned = True
        for _ in range(args.goal_timeout):
            if step_id >= args.max_steps_per_episode:
                break
            write_step(goal, spawned)
            spawned = False
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                goals_reached += 1
                break

    return goals_reached, tracker.count, step_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps_per_episode", type=int, default=3000)
    parser.add_argument("--record_every", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--goal_free_steps", type=int, default=25)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--goal_bias", choices=["none", "quadrant"], default="none")
    parser.add_argument("--bias_p", type=float, default=0.8)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--expert_eta", type=float, default=0.05)
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
    parser.add_argument("--output", type=str, default="dataset/data_goal_unbiased_360.csv")
    args = parser.parse_args()

    random.seed(args.random_seed)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["fx", "fy"]
            + [f"scan_{i}" for i in range(args.num_features)]
            + [
                "goal_dx", "goal_dy", "hist_dx", "hist_dy",
                "player_x", "player_y", "goal_x", "goal_y",
                "goal_present", "goal_spawn", "respawn", "episode",
            ]
        )
        total_goals = total_collisions = total_steps = 0
        for ep in range(args.num_episodes):
            goals, collisions, steps = run_episode(args, writer, ep)
            total_goals += goals
            total_collisions += collisions
            total_steps += steps
            print(f"Episode {ep}: steps={steps}, goals={goals}, collisions={collisions}")

    minutes = total_steps / 50 / 60
    print(
        f"\nExpert summary: {total_goals} goals, {total_collisions} collisions "
        f"over {total_steps} steps ({total_goals / minutes:.2f} goals/min, "
        f"{total_collisions / minutes:.2f} collisions/min)"
    )
    print(f"Demonstrations written to {args.output}")


if __name__ == "__main__":
    main()
