"""Goal-directed potential-field expert for the reach-avoid task.

Runs the expert (repulsive from LiDAR rays + attractive toward the goal)
headlessly and records demonstrations to CSV.  Columns:

    fx, fy, scan_0..scan_{N-1}, goal_dx, goal_dy, episode

goal_dx/goal_dy are the goal position relative to the player (pixels).
`episode` marks episode boundaries so the dataloader never pairs frames
across a reset.  Reaching a goal spawns a new one (continuous throughput);
a collision ends the episode and resets the world.
"""

import argparse
import csv
import math
import random

from reach_avoid_common import (
    GOAL_RADIUS,
    CollisionTracker,
    expert_goal_action,
    lidar_scan,
    make_world,
    sample_goal,
)


def run_episode(args, csv_writer, episode_id):
    player, balls = make_world(args.num_balls, args.width, args.height)
    goal = sample_goal(player, args.width, args.height)
    tracker = CollisionTracker()
    goals_reached = 0

    for step in range(args.max_steps_per_episode):
        for ball in balls:
            if ball is not player:
                ball.move()

        distances, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        goal_dx = goal[0] - player.x
        goal_dy = goal[1] - player.y

        fx, fy = expert_goal_action(distances, goal_dx, goal_dy, args.num_features)

        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))

        if step % args.record_every == 0:
            csv_writer.writerow([fx, fy] + distances + [goal_dx, goal_dy, episode_id])

        if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
            goals_reached += 1
            goal = sample_goal(player, args.width, args.height)

        if tracker.update(player, balls) and args.reset_on_collision:
            break

    return goals_reached, tracker.count, step + 1


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
    parser.add_argument("--reset_on_collision", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="dataset/data_goal.csv")
    args = parser.parse_args()

    random.seed(args.random_seed)

    with open(args.output, "w", newline="") as f:
        csv_writer = csv.writer(f)
        header = (
            ["fx", "fy"]
            + [f"scan_{i}" for i in range(args.num_features)]
            + ["goal_dx", "goal_dy", "episode"]
        )
        csv_writer.writerow(header)

        total_goals, total_collisions, total_steps = 0, 0, 0
        for ep in range(args.num_episodes):
            goals, collisions, steps = run_episode(args, csv_writer, ep)
            total_goals += goals
            total_collisions += collisions
            total_steps += steps
            print(
                f"Episode {ep}: steps={steps}, goals={goals}, collisions={collisions}"
            )

    minutes = total_steps / 50 / 60  # simulation runs at 50 Hz
    print(
        f"\nExpert summary: {total_goals} goals, {total_collisions} collisions "
        f"over {total_steps} steps ({total_goals / minutes:.2f} goals/min, "
        f"{total_collisions / minutes:.2f} collisions/min)"
    )
    print(f"Demonstrations written to {args.output}")


if __name__ == "__main__":
    main()
