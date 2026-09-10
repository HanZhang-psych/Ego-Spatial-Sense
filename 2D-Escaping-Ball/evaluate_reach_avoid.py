"""Closed-loop evaluation for the goal-directed reach-avoid task.

The ego must reach a destination (green circle) while moving balls are
present.  Reaching a goal spawns a new one (continuous throughput); with
--single_goal the episode instead ends at the first goal.  Collisions do
NOT end the run — collision events are counted (an overlap lasting several
frames counts once) so both sides of the trade-off are measured:

    goals reached per minute   vs.   collisions per minute
    (--single_goal: steps to destination + collisions on the way)

Models:
  - "goal" in --model_path  -> GoalEs2Model, observes prev scan + scan + relative goal
  - "es2"/"mlp"/"transformer" -> the original 720-input goal-blind models,
    run in the same environment as reactive baselines (they never see the goal).

The rollout also supports feeding the model a goal different from the scored
one (used by goal_swap_probe.py) and reports the attention-field asymmetry
toward the fed goal for the goal-conditioned model.
"""

import argparse
import math
import random
import sys

import torch

from model.es2 import Es2Model
from model.goal_es2 import GoalEs2Model
from model.mlp import MLPModel
from model.transformer import TransformerModel
from reach_avoid_common import (
    GOAL_RADIUS,
    GOAL_WALL_MARGIN,
    CollisionTracker,
    lidar_scan,
    make_world,
    sample_goal,
)


def load_model(args):
    """Load a model by path.  Returns (model, is_goal_conditioned)."""
    if "goal" in args.model_path:
        if "mlp" in args.model_path:
            from model.goal_mlp import GoalMLPModel

            model = GoalMLPModel(
                num_features=args.num_features,
                num_actions=args.num_actions,
                sensing_range=args.sensing_range,
            ).to(args.device)
            print("Loading goal-conditioned MLP model...")
        elif "transformer" in args.model_path:
            from model.goal_transformer import GoalTransformerModel

            model = GoalTransformerModel(
                num_features=args.num_features,
                num_actions=args.num_actions,
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                sensing_range=args.sensing_range,
            ).to(args.device)
            print("Loading goal-conditioned Transformer model...")
        else:
            model = GoalEs2Model(
                num_features=args.num_features,
                num_actions=args.num_actions,
                sensing_range=args.sensing_range,
            ).to(args.device)
            print("Loading goal-conditioned ES2 model...")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.eval()
        return model, True

    if "es2" in args.model_path:
        model = Es2Model(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)
        print("Loading ES2 model (goal-blind baseline)...")
    elif "mlp" in args.model_path:
        model = MLPModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)
        print("Loading MLP model (goal-blind baseline)...")
    elif "transformer" in args.model_path:
        model = TransformerModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            sensing_range=args.sensing_range,
        ).to(args.device)
        print("Loading Transformer model (goal-blind baseline)...")
    else:
        print("Error: undefined model type.")
        sys.exit(1)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    return model, False


def mirror_goal(goal, args):
    """Reflect a goal through the arena center, clamped to the wall margin."""
    gx = args.width - goal[0]
    gy = args.height - goal[1]
    gx = max(GOAL_WALL_MARGIN, min(args.width - GOAL_WALL_MARGIN, gx))
    gy = max(GOAL_WALL_MARGIN, min(args.height - GOAL_WALL_MARGIN, gy))
    return gx, gy


def run_rollout(model, is_goal_model, args, seed, goal_mode="correct", render=False):
    """Run one closed-loop rollout.

    goal_mode:
      "correct"  - the model observes the true (scored) goal.
      "opposite" - the model observes the true goal mirrored through the center.
      "random"   - the model observes an independent random goal, resampled
                   whenever the ego reaches it.
    Returns a dict of counters (true/fed goals reached, collisions, steps,
    mean attention-field asymmetry toward the fed goal).
    """
    random.seed(seed)

    if render:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("Reach-Avoid")
        clock = pygame.time.Clock()

    player, balls = make_world(args.num_balls, args.width, args.height)
    true_goal = sample_goal(player, args.width, args.height)

    if goal_mode == "correct":
        fed_goal = true_goal
    elif goal_mode == "opposite":
        fed_goal = mirror_goal(true_goal, args)
    elif goal_mode == "random":
        fed_goal = sample_goal(player, args.width, args.height)
    else:
        raise ValueError(f"Unknown goal_mode: {goal_mode}")

    tracker = CollisionTracker()
    previous_scan = None
    true_goals_reached = 0
    fed_goals_reached = 0
    asym_sum, asym_n = 0.0, 0

    steps = 0
    while steps < args.max_steps:
        steps += 1

        if render:
            import pygame

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

        for ball in balls:
            if ball is not player:
                ball.move()

        distances, _ = lidar_scan(
            player, balls, args.width, args.height, args.num_features
        )
        if previous_scan is None:
            previous_scan = distances.copy()

        if is_goal_model:
            obs = previous_scan + distances + [
                fed_goal[0] - player.x,
                fed_goal[1] - player.y,
            ]
        else:
            obs = previous_scan + distances
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(args.device)

        with torch.no_grad():
            action = model(obs_t)
            if (
                is_goal_model
                and hasattr(model, "compute_fields")
                and steps % args.field_every == 0
            ):
                obstacle_field, goal_field = model.compute_fields(obs_t)
                field = (obstacle_field + goal_field)[0]
                bearing = math.degrees(
                    math.atan2(fed_goal[1] - player.y, fed_goal[0] - player.x)
                ) % 360
                idx = torch.arange(args.num_features, dtype=torch.float32)
                ang = idx * (360 / args.num_features)
                diff = torch.remainder(ang - bearing + 180, 360) - 180
                toward = field[diff.abs() <= 60].mean().item()
                away = field[diff.abs() >= 120].mean().item()
                asym_sum += toward - away
                asym_n += 1

        net_fx, net_fy = action[0][0].item(), action[0][1].item()
        fx = min(max(net_fx, -args.max_speed), args.max_speed)
        fy = min(max(net_fy, -args.max_speed), args.max_speed)

        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))

        previous_scan = distances.copy()

        tracker.update(player, balls)

        # Scored (true) goal
        if math.hypot(true_goal[0] - player.x, true_goal[1] - player.y) < GOAL_RADIUS:
            true_goals_reached += 1
            if args.single_goal:
                break
            true_goal = sample_goal(player, args.width, args.height)
            if goal_mode == "correct":
                fed_goal = true_goal
            elif goal_mode == "opposite":
                fed_goal = mirror_goal(true_goal, args)

        # Fed goal (only distinct from the true goal in swap modes)
        if goal_mode != "correct" and (
            math.hypot(fed_goal[0] - player.x, fed_goal[1] - player.y) < GOAL_RADIUS
        ):
            fed_goals_reached += 1
            if goal_mode == "random":
                fed_goal = sample_goal(player, args.width, args.height)
            else:  # opposite: move the true goal so the fed goal moves too
                true_goal = sample_goal(player, args.width, args.height)
                fed_goal = mirror_goal(true_goal, args)

        if render:
            import pygame

            screen.fill((255, 255, 255))
            pygame.draw.circle(
                screen,
                (0, 200, 0),
                (int(true_goal[0]), int(true_goal[1])),
                GOAL_RADIUS,
                3,
            )
            if goal_mode != "correct":
                pygame.draw.circle(
                    screen,
                    (255, 165, 0),
                    (int(fed_goal[0]), int(fed_goal[1])),
                    GOAL_RADIUS,
                    2,
                )
            for ball in balls:
                pygame.draw.circle(
                    screen, ball.color, (int(ball.x), int(ball.y)), ball.radius
                )
            pygame.display.flip()
            clock.tick(args.frequency)

    if render:
        import pygame

        pygame.quit()

    minutes = steps / 50 / 60  # simulation runs at 50 Hz
    return {
        "steps": steps,
        "true_goals": true_goals_reached,
        "fed_goals": fed_goals_reached,
        "collisions": tracker.count,
        "true_goals_per_min": true_goals_reached / minutes,
        "fed_goals_per_min": fed_goals_reached / minutes,
        "collisions_per_min": tracker.count / minutes,
        "field_asymmetry": (asym_sum / asym_n) if asym_n else float("nan"),
    }


def add_common_args(parser):
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--single_goal", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--field_every", type=int, default=25)
    parser.add_argument("--model_path", type=str, default="pretrained/goal_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    # Transformer-specific parameters
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    return parser


def summarize(results, keys):
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    model, is_goal_model = load_model(args)

    results = []
    for i in range(args.num_seeds):
        seed = args.random_seed + i
        r = run_rollout(
            model, is_goal_model, args, seed, goal_mode="correct", render=args.render
        )
        if r is None:
            sys.exit(0)
        results.append(r)
        print(
            f"Seed {seed}: steps={r['steps']}, goals={r['true_goals']}, "
            f"collisions={r['collisions']}, "
            f"goals/min={r['true_goals_per_min']:.2f}, "
            f"collisions/min={r['collisions_per_min']:.2f}, "
            f"field_asym={r['field_asymmetry']:.4f}"
        )

    avg = summarize(
        results,
        ["steps", "true_goals", "collisions", "true_goals_per_min", "collisions_per_min"],
    )
    print(
        f"\n=== {args.model_path} over {args.num_seeds} seeds ===\n"
        f"mean goals/min:      {avg['true_goals_per_min']:.2f}\n"
        f"mean collisions/min: {avg['collisions_per_min']:.2f}\n"
        f"mean steps:          {avg['steps']:.0f}"
    )
    if args.single_goal:
        print(f"mean steps to goal (single-shot): {avg['steps']:.1f}")
