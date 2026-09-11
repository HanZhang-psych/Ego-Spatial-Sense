"""
Priority-weighted, fixed-duration evaluation for the 2D Escaping Ball game.

Self-contained variant of ``evaluate.py``. Two behavioral changes:

  1. Fixed duration. The game no longer stops on the first collision; it always
     runs for ``--duration`` seconds. Survival time is therefore not the metric
     -- the metric is an accumulated *collision score*.

  2. Priority classes. Balls come in several classes, each with its own color
     and collision penalty (e.g. hitting a red ball costs more than a green
     one). On each contact the ball's penalty is added once and that ball is
     respawned far from the ego (so a single contact is counted once, and ball
     count stays constant). Lower total score = better.

Increase difficulty with ``--num_balls`` and/or the ball-speed range
(``--ball_speed_min`` / ``--ball_speed_max``).

IMPORTANT (perception): the pretrained models observe only a 720-dim LiDAR
*distance* scan (2 frames x 360 rays). That signal carries NO color/class
information, so a pretrained agent avoids every ball equally -- it cannot act on
priority. The weighted score here is thus an evaluation *metric* over existing
models, not something they were trained to optimize. Making the agent
priority-aware would require extending the observation and retraining.

Example:
    python3 evaluate_scored.py --model_path pretrained/es2.pth \
        --random_seed 42 --num_balls 20 --duration 60 --device cpu --render
"""

import sys
import pygame
import random
import math
import colorsys
import torch
import argparse


from model.es2 import Es2Model
from model.mlp import MLPModel
from model.transformer import TransformerModel


# Palette used for ball classes, ordered from highest to lowest priority by
# convention. Extend if you define more than five classes.
CLASS_COLORS = [
    (220, 20, 20),    # red
    (245, 190, 0),    # amber
    (40, 180, 60),    # green
    (30, 120, 220),   # blue
    (150, 60, 200),   # purple
]
PLAYER_COLOR = (25, 25, 25)   # dark, so it is never confused with a red ball
ARROW_COLOR = (200, 0, 200)   # magenta, high contrast against class colors


# ---------------- Utility: Draw an Arrow ----------------
def draw_arrow(
    surface,
    color,
    start,
    end,
    arrow_width=3,
    arrow_head_length=10,
    arrow_head_angle=math.pi / 6,
):
    pygame.draw.line(surface, color, start, end, arrow_width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    left_angle = angle + math.pi - arrow_head_angle
    right_angle = angle + math.pi + arrow_head_angle
    left_point = (
        end[0] + arrow_head_length * math.cos(left_angle),
        end[1] + arrow_head_length * math.sin(left_angle),
    )
    right_point = (
        end[0] + arrow_head_length * math.cos(right_angle),
        end[1] + arrow_head_length * math.sin(right_angle),
    )
    pygame.draw.polygon(surface, color, [end, left_point, right_point])


# ---------------- Ball Class ----------------
class Ball:
    def __init__(self, x, y, radius, color, speed, class_idx=0, penalty=0.0):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.radius = radius
        self.color = color
        self.speed = speed
        # Priority-class metadata (only meaningful for background balls).
        self.class_idx = class_idx
        self.penalty = penalty
        # For moving balls, choose a random initial direction.
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

    def move(self):
        self.prev_x, self.prev_y = self.x, self.y
        self.x += self.dx
        self.y += self.dy

        # Bounce off the walls.
        if self.x - self.radius < 0 or self.x + self.radius > args.width:
            self.dx *= -1
        if self.y - self.radius < 0 or self.y + self.radius > args.height:
            self.dy *= -1

    def respawn(self, player, min_distance):
        """Teleport to a random spot away from the ego, with a fresh velocity.

        Keeps the ball's class/penalty; only position and direction change.
        """
        while True:
            x = random.randint(30, args.width - 30)
            y = random.randint(30, args.height - 30)
            if math.hypot(x - player.x, y - player.y) >= min_distance:
                break
        self.x = x
        self.y = y
        self.prev_x, self.prev_y = x, y
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

    def collides_with(self, other):
        distance = math.hypot(self.x - other.x, self.y - other.y)
        return distance < self.radius + other.radius


# ---------------- Evaluate Function ----------------
def evaluate(model, args):
    """Run the fixed-duration, priority-scored simulation. Returns a result dict."""

    random.seed(args.random_seed)

    # --- Resolve the class scheme (penalties + probabilities). ---
    penalties = list(args.penalties)
    num_classes = len(penalties)
    if num_classes > len(CLASS_COLORS):
        print(
            f"Error: {num_classes} classes requested but only "
            f"{len(CLASS_COLORS)} colors defined. Add more to CLASS_COLORS."
        )
        sys.exit()

    if args.class_probs is not None:
        probs = list(args.class_probs)
        if len(probs) != num_classes:
            print("Error: --class_probs length must match --penalties length.")
            sys.exit()
    else:
        probs = [1.0 / num_classes] * num_classes

    quiet = getattr(args, "quiet", False)

    pygame.init()
    pygame.font.init()

    if args.render:
        font = pygame.font.SysFont("Arial", 22)
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("Priority-Weighted Escaping Ball (Fixed Duration)")
    else:
        screen = None

    WHITE = (255, 255, 255)
    clock = pygame.time.Clock()

    # Create the player and the background balls.
    player = Ball(args.width // 2, args.height // 2, 20, PLAYER_COLOR, 0)
    balls = []
    margin = 10
    min_distance = player.radius + 15 + margin
    for _ in range(args.num_balls):
        while True:
            x = random.randint(30, args.width - 30)
            y = random.randint(30, args.height - 30)
            if math.hypot(x - player.x, y - player.y) >= min_distance:
                break
        radius = 20
        speed = random.randint(args.ball_speed_min, args.ball_speed_max)
        cls = random.choices(range(num_classes), weights=probs, k=1)[0]
        balls.append(
            Ball(x, y, radius, CLASS_COLORS[cls], speed, class_idx=cls, penalty=penalties[cls])
        )
    balls.append(player)

    total_steps = int(args.duration * args.frequency)

    if not quiet:
        scheme = ", ".join(
            f"class{i}(penalty={penalties[i]}, p={probs[i]:.2f})" for i in range(num_classes)
        )
        print(
            f"[scored] duration={args.duration:.0f}s ({total_steps} steps), "
            f"balls={args.num_balls}, speed=[{args.ball_speed_min},{args.ball_speed_max}]"
        )
        print(f"[scored] classes: {scheme}")

    # ---------------- LiDAR Scan Function ----------------
    def lidar_scan():
        distances = []
        intersections = []
        for i in range(args.num_features):
            rad = math.radians(i)
            dx = math.cos(rad)
            dy = math.sin(rad)

            min_distance_val = float("inf")
            intersection_point = None

            if dx != 0:
                t = (0 - player.x) / dx
                if t > 0:
                    y_int = player.y + t * dy
                    if 0 <= y_int <= args.height and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (0, y_int)
            if dx != 0:
                t = (args.width - player.x) / dx
                if t > 0:
                    y_int = player.y + t * dy
                    if 0 <= y_int <= args.height and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (args.width, y_int)
            if dy != 0:
                t = (0 - player.y) / dy
                if t > 0:
                    x_int = player.x + t * dx
                    if 0 <= x_int <= args.width and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (x_int, 0)
            if dy != 0:
                t = (args.height - player.y) / dy
                if t > 0:
                    x_int = player.x + t * dx
                    if 0 <= x_int <= args.width and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (x_int, args.height)

            for ball in balls:
                if ball == player:
                    continue
                bx = ball.x - player.x
                by = ball.y - player.y
                A = dx**2 + dy**2
                B = -2 * (bx * dx + by * dy)
                C = bx**2 + by**2 - ball.radius**2
                discriminant = B**2 - 4 * A * C
                if discriminant >= 0:
                    sqrt_disc = math.sqrt(discriminant)
                    t1 = (-B - sqrt_disc) / (2 * A)
                    t2 = (-B + sqrt_disc) / (2 * A)
                    if t1 > 0 and t1 < min_distance_val:
                        min_distance_val = t1
                        intersection_point = (player.x + t1 * dx, player.y + t1 * dy)
                    if t2 > 0 and t2 < min_distance_val:
                        min_distance_val = t2
                        intersection_point = (player.x + t2 * dx, player.y + t2 * dy)
            distances.append(min_distance_val)
            intersections.append(intersection_point)

        return distances, intersections

    previous_scan = None

    # ---------------- Scoring state ----------------
    class_counts = [0] * num_classes
    weighted_score = 0.0
    total_collisions = 0

    # ---------------- Main Simulation Loop ----------------
    steps = 0
    while steps < total_steps:
        steps += 1

        if args.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

        # Update background balls (the player ball does not move on its own).
        for ball in balls:
            if ball != player:
                ball.move()

        distances, intersections = lidar_scan()

        if previous_scan is None:
            previous_scan = distances.copy()

        scan_input = (
            torch.tensor(previous_scan + distances, dtype=torch.float32)
            .unsqueeze(0)
            .to(args.device)
        )

        with torch.no_grad():
            action = model(scan_input)

        net_fx, net_fy = action[0][0].item(), action[0][1].item()
        fx = min(max(net_fx, -args.max_speed), args.max_speed)
        fy = min(max(net_fy, -args.max_speed), args.max_speed)
        net_force_magnitude = math.hypot(net_fx, net_fy)

        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))

        previous_scan = distances.copy()

        # ---- Scoring: count each contact once, then respawn the ball. ----
        for ball in balls:
            if ball != player and player.collides_with(ball):
                class_counts[ball.class_idx] += 1
                weighted_score += ball.penalty
                total_collisions += 1
                ball.respawn(player, min_distance)

        if args.render:
            screen.fill(WHITE)
            for ball in balls:
                ball.draw(screen)

            max_distance_threshold = math.hypot(args.width, args.height)
            for i, (d, intersection) in enumerate(zip(distances, intersections)):
                if intersection:
                    rad_angle_line = math.radians(i)
                    start_x = player.x + player.radius * math.cos(rad_angle_line)
                    start_y = player.y + player.radius * math.sin(rad_angle_line)
                    start_point = (start_x, start_y)
                    norm = min(d / max_distance_threshold, 1)
                    hue = math.sqrt(norm) * 0.33
                    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
                    color = (int(r * 255), int(g * 255), int(b * 255))
                    pygame.draw.line(screen, color, start_point, intersection, 1)

            if net_force_magnitude > 0:
                arrow_angle = math.degrees(math.atan2(net_fy, net_fx))
            else:
                arrow_angle = 0
            arrow_length = 40
            start_pos = (player.x, player.y)
            end_pos = (
                player.x + arrow_length * math.cos(math.radians(arrow_angle)),
                player.y + arrow_length * math.sin(math.radians(arrow_angle)),
            )
            draw_arrow(screen, ARROW_COLOR, start_pos, end_pos)

            time_left = (total_steps - steps) / args.frequency
            hud = font.render(
                f"t-{time_left:5.1f}s   score: {weighted_score:.0f}   hits: {total_collisions}",
                True,
                (0, 0, 0),
            )
            screen.blit(hud, (10, 10))
            breakdown = "  ".join(
                f"c{i}:{class_counts[i]}" for i in range(num_classes)
            )
            legend = font.render(breakdown, True, (0, 0, 0))
            screen.blit(legend, (10, 36))
            pygame.display.flip()
            clock.tick(args.frequency)
        elif not getattr(args, "no_throttle", False):
            clock.tick(args.frequency)

    result = {
        "weighted_score": weighted_score,
        "total_collisions": total_collisions,
        "class_counts": class_counts,
        "penalties": penalties,
        "duration": args.duration,
        "steps": steps,
    }

    if not quiet:
        print("-" * 60)
        for i in range(num_classes):
            print(
                f"  class{i}: penalty={penalties[i]:<4} hits={class_counts[i]:<4} "
                f"subtotal={penalties[i] * class_counts[i]:.0f}"
            )
        print(f"  total hits      : {total_collisions}")
        print(f"  WEIGHTED SCORE  : {weighted_score:.0f}  (lower is better)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment and simulation settings
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_balls", type=int, default=20)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Fixed episode length in seconds (never ends early).",
    )

    # Difficulty: background-ball speed range (inclusive, integer px/step).
    parser.add_argument("--ball_speed_min", type=int, default=1)
    parser.add_argument("--ball_speed_max", type=int, default=3)

    # Priority-class scheme.
    parser.add_argument(
        "--penalties",
        nargs="+",
        type=float,
        default=[5.0, 2.0, 1.0],
        help="Per-class collision penalties, highest priority first. The number "
        "of values sets the number of classes (max 5).",
    )
    parser.add_argument(
        "--class_probs",
        nargs="+",
        type=float,
        default=None,
        help="Spawn probability per class (same length as --penalties). "
        "Defaults to a uniform split. Need not sum to 1 (auto-normalized).",
    )

    # Visualization settings
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--no_throttle",
        action="store_true",
        help="Headless only: run as fast as possible (useful for batch sweeps).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-run prints."
    )
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)

    # Model settings
    parser.add_argument("--model_path", type=str, default="pretrained/es2.pth")
    parser.add_argument("--device", type=str, default="cuda")

    # Transformer-specific parameters
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)

    args = parser.parse_args()

    if "es2" in args.model_path:
        model = Es2Model(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)
        print("Loading ES2 model...")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.eval()

    elif "mlp" in args.model_path:
        model = MLPModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)
        print("Loading MLP model...")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.eval()

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
        print("Loading Transformer model...")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.eval()

    else:
        print("Error: undefined model type.")
        sys.exit()

    evaluate(model, args)
