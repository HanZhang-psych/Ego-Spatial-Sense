"""
Adversarial edge-case evaluation for the 2D Escaping Ball game.

This is a self-contained variant of ``evaluate.py``. It reproduces the exact
same simulation, LiDAR sensing, and model interface, and adds one extra
"long-tail" stressor that never appears in the training demonstration:

    At a randomly chosen (but *seeded*) moment, one of the balls switches into
    a pursuit mode and starts moving erratically toward the ego ball.

Design choices (see the paper's Feature #1: "Reflexive avoidance of imminent
collision threats"):

  * Reproducible, not wall-clock random. The trigger time and the pursuer's
    per-step jitter are drawn from ``--random_seed`` so that every model
    (es2 / mlp / transformer) faces the *identical* perturbation. The pre-
    trigger ball layout is drawn with the same RNG sequence as ``evaluate.py``,
    so the base scenario matches that script for the same seed.

  * Solvable by construction. The pursuer's speed is bounded (default: the
    ego's own ``--max_speed``), so a competent evader can still escape it in a
    bounded arena. A pursuer strictly faster than the ego would make collision
    unavoidable and the test would discriminate between no models.

  * "Erratic" = homing heading + bounded angular noise. Each step the pursuer
    aims its velocity at the ego and then perturbs the heading by a random
    angle in ``[-pursuit_noise, +pursuit_noise]`` radians. Speed magnitude
    stays fixed at ``--pursuer_speed`` so the bound above always holds.

Report this as its OWN condition (its own survival-time row); do not fold it
into the 10/20/30/40-ball averages from the paper.

Example:
    python3 evaluate_adversarial.py \
        --model_path pretrained/es2.pth \
        --random_seed 42 --num_balls 10 --device cpu --render
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
    def __init__(self, x, y, radius, color, speed):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.radius = radius
        self.color = color
        self.speed = speed
        # For moving balls, choose a random initial direction.
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

        # Adversarial-pursuit state (inert unless this ball is the pursuer).
        self.pursuing = False
        self.pursuit_speed = 0.0
        self.pursuit_noise = 0.0
        self.target = None  # reference to the ego ball once pursuit starts

    def move(self):
        self.prev_x, self.prev_y = self.x, self.y

        if self.pursuing and self.target is not None:
            # Erratic homing: aim at the ego, then perturb the heading.
            angle = math.atan2(self.target.y - self.y, self.target.x - self.x)
            angle += random.uniform(-self.pursuit_noise, self.pursuit_noise)
            self.dx = self.pursuit_speed * math.cos(angle)
            self.dy = self.pursuit_speed * math.sin(angle)
            self.x += self.dx
            self.y += self.dy
            # Clamp inside the arena (a pursuer should not bounce away from prey).
            self.x = max(self.radius, min(args.width - self.radius, self.x))
            self.y = max(self.radius, min(args.height - self.radius, self.y))
            return

        # Default (non-adversarial) behavior: constant velocity + wall bounce.
        self.x += self.dx
        self.y += self.dy
        if self.x - self.radius < 0 or self.x + self.radius > args.width:
            self.dx *= -1
        if self.y - self.radius < 0 or self.y + self.radius > args.height:
            self.dy *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

    def collides_with(self, other):
        distance = math.hypot(self.x - other.x, self.y - other.y)
        return distance < self.radius + other.radius


# ---------------- Evaluate Function ----------------
def evaluate(model, args):
    """
    Run the simulation. The model receives two LiDAR scans concatenated:
      - previous scan (length 360)
      - current scan (length 360)
    For the very first iteration, the previous scan is set equal to the current scan.
    The resulting input has 720 features.

    One ball becomes an adversarial pursuer at a seeded trigger time.
    """

    random.seed(args.random_seed)

    pygame.init()
    pygame.font.init()

    if args.render:
        font = pygame.font.SysFont("Arial", 24)
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("Avoid the Balls (Adversarial Edge Case)")
    else:
        screen = None

    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    ORANGE = (255, 140, 0)  # highlights the pursuer once it activates

    clock = pygame.time.Clock()

    # Create the player (red ball) and blue balls.
    # NOTE: this RNG sequence is identical to evaluate.py, so for a given seed
    # the base scenario matches that script exactly.
    player = Ball(
        args.width // 2, args.height // 2, 20, RED, 0
    )  # Player does not move on its own.
    balls = []
    margin = 10
    min_distance = (
        player.radius + 15 + margin
    )  # Ensure blue balls don't start too near the player.
    for _ in range(args.num_balls):
        while True:
            x = random.randint(30, args.width - 30)
            y = random.randint(30, args.height - 30)
            if math.hypot(x - player.x, y - player.y) >= min_distance:
                break
        radius = 20
        speed = random.randint(1, 3)
        balls.append(Ball(x, y, radius, BLUE, speed))

    # ---- Adversarial setup (all draws AFTER ball init to preserve layout) ----
    if args.num_balls < 1:
        print("Adversarial mode requires at least one ball.")
        sys.exit()

    pursuer_speed = args.pursuer_speed if args.pursuer_speed > 0 else args.max_speed

    # Which ball turns hostile.
    if 0 <= args.pursuer_index < args.num_balls:
        pursuer_idx = args.pursuer_index
    else:
        pursuer_idx = random.randrange(args.num_balls)
    pursuer = balls[pursuer_idx]

    # When it turns hostile (in steps), drawn from a seeded time window.
    trig_lo = int(args.trigger_min * args.frequency)
    trig_hi = int(args.trigger_max * args.frequency)
    trig_hi = max(trig_hi, trig_lo)
    trigger_step = random.randint(trig_lo, trig_hi)

    quiet = getattr(args, "quiet", False)
    if not quiet:
        print(
            f"[adversarial] ball #{pursuer_idx} will pursue at "
            f"t={trigger_step / args.frequency:.2f}s (step {trigger_step}), "
            f"speed={pursuer_speed:.2f} (ego max={args.max_speed:.2f}), "
            f"heading noise=±{args.pursuit_noise:.2f} rad"
        )

    balls.append(player)

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

            # Check intersections with walls.
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

            # Check intersections with the other balls.
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

    # Variable to hold the previous scan.
    previous_scan = None

    # ---------------- Main Simulation Loop ----------------
    steps = 0
    pursuit_started = False

    while True:
        steps += 1

        if steps >= args.max_steps:
            if not quiet:
                print(f"Maximum steps reached: {args.max_steps}")
            return steps

        if args.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return steps

        # Activate the pursuer at the seeded trigger time.
        if not pursuit_started and steps >= trigger_step:
            pursuit_started = True
            pursuer.pursuing = True
            pursuer.pursuit_speed = pursuer_speed
            pursuer.pursuit_noise = args.pursuit_noise
            pursuer.target = player
            pursuer.color = ORANGE
            if not quiet:
                print(f"[adversarial] pursuit ACTIVATED at step {steps}")

        # Update the balls (the player ball does not move on its own)
        for ball in balls:
            if ball != player:
                ball.move()

        # Perform LiDAR scan.
        distances, intersections = lidar_scan()

        # Initialize previous_scan on the very first iteration.
        if previous_scan is None:
            previous_scan = distances.copy()

        # Create model input: previous scan (360) + current scan (360) -> 720.
        scan_input = (
            torch.tensor(previous_scan + distances, dtype=torch.float32)
            .unsqueeze(0)
            .to(args.device)
        )

        with torch.no_grad():
            action = model(scan_input)

        net_fx, net_fy = action[0][0].item(), action[0][1].item()

        # Bound fx and fy to max speed.
        fx = min(max(net_fx, -args.max_speed), args.max_speed)
        fy = min(max(net_fy, -args.max_speed), args.max_speed)

        net_force_magnitude = math.hypot(net_fx, net_fy)

        player.x += int(fx)
        player.y += int(fy)

        # Keep the player within bounds.
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))

        # Update previous_scan for the next iteration.
        previous_scan = distances.copy()

        # Check for collisions between the player and the other balls.
        for ball in balls:
            if ball != player and player.collides_with(ball):
                hit_by_pursuer = ball is pursuer and pursuit_started
                if not quiet:
                    tag = " (by the pursuer)" if hit_by_pursuer else ""
                    print(f"Collision{tag}! Total steps", steps)
                return steps

        if args.render:
            screen.fill(WHITE)

            # Draw all balls.
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
                    hue = math.sqrt(norm) * 0.33  # Adjust for color variation.
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
            draw_arrow(screen, RED, start_pos, end_pos)

            status = "PURSUIT" if pursuit_started else "calm"
            text_surface = font.render(
                f"t={steps / args.frequency:.1f}s  "
                f"Angle: {arrow_angle:.1f}deg  Mag: {net_force_magnitude:.2f}  [{status}]",
                True,
                (0, 0, 0),
            )
            screen.blit(text_surface, (10, 10))
            pygame.display.flip()
            clock.tick(args.frequency)
        elif not getattr(args, "no_throttle", False):
            # In headless mode, just control simulation speed.
            # (--no_throttle runs as fast as possible, e.g. for batch sweeps.)
            clock.tick(args.frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment and simulation settings
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=15000)

    # Adversarial edge-case settings
    parser.add_argument(
        "--pursuer_speed",
        type=float,
        default=0.0,
        help="Speed of the pursuing ball once active. <=0 defaults to --max_speed "
        "(keeps the task solvable; a faster pursuer makes collision unavoidable).",
    )
    parser.add_argument(
        "--pursuit_noise",
        type=float,
        default=0.6,
        help="Max heading perturbation in radians for the 'erratic' pursuit.",
    )
    parser.add_argument(
        "--trigger_min",
        type=float,
        default=5.0,
        help="Earliest time (s) the pursuit can start.",
    )
    parser.add_argument(
        "--trigger_max",
        type=float,
        default=30.0,
        help="Latest time (s) the pursuit can start.",
    )
    parser.add_argument(
        "--pursuer_index",
        type=int,
        default=-1,
        help="Index of the ball that becomes the pursuer. <0 picks one at random "
        "(seeded).",
    )

    # Visualization settings
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--no_throttle",
        action="store_true",
        help="Headless only: run as fast as possible instead of capping at "
        "--frequency steps/sec (useful for batch sweeps).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress prints (used by the batch runner).",
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

    steps = evaluate(model, args)

    print(f"Survived {steps} steps ({steps / args.frequency:.2f} s)")
