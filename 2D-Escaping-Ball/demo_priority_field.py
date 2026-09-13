"""Live COMBINED priority-map demo - GATED construction of record.

The agent plays the continuous goal task.  Every LiDAR ray is drawn from
the agent to its intersection point (exactly as in evaluate.py), but
colored by the COMBINED priority field the planner reads on that ray -
one map, with the precedence rule of record: after each goal a
goal-free ANTICIPATION WINDOW (default 50 steps) where the map is
obstacle + beta * history-of-the-previous-goal and the agent drifts;
then the next goal appears and OVERRIDES - history is gated off and
the map is obstacle + goal:

  red = negative (push / avoid)   ->   green = positive (pull / approach)

Trials are fixation-start, as in evaluate_priming.py: reaching a goal
teleports the agent to center before the anticipation window, so the
drift toward the remembered location is visible movement.  Half the
goals repeat the previous location (the drift pays off), half spawn
elsewhere.  Green ring = current goal, purple dot = the previous goal
(the one-back trace), red arrow = the planner's movement.  --beta sets
the history strength (default 0.2, the priming demonstration's
human-plausible regime; the checkpoint of record has beta_H = 0), and
--window the anticipation length.  Esc/Q quits.

Run: python demo_priority_field.py  (requires pretrained/goal_es2.pth -
train with train_goal_es2.py)."""

import argparse
import colorsys
import math
import random

import pygame
import torch

from model.goal_es2 import GoalEs2Model
from environment import (GOAL_RADIUS, CollisionTracker, lidar_scan,
                         make_world, sample_goal)

W = H = 800
WHITE, RED, GREEN, PURPLE = (255,255,255), (255,0,0), (17,136,68), (120,60,160)

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="pretrained/goal_es2.pth")
ap.add_argument("--num_balls", type=int, default=10)
ap.add_argument("--beta", type=float, default=0.2,
                help="history strength beta_H (0 = model of record)")
ap.add_argument("--window", type=int, default=50,
                help="goal-free anticipation steps between goals")
ap.add_argument("--fps", type=int, default=50)
args = ap.parse_args()

model = GoalEs2Model(num_features=360)
model.load_state_dict(torch.load(args.model_path, map_location="cpu",
                                 weights_only=True))
model.eval()

def draw_arrow(surface, color, start, end, arrow_width=4, head=12):
    pygame.draw.line(surface, color, start, end, arrow_width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (end[0] - head * math.cos(ang - 0.4), end[1] - head * math.sin(ang - 0.4))
    right = (end[0] - head * math.cos(ang + 0.4), end[1] - head * math.sin(ang + 0.4))
    pygame.draw.polygon(surface, color, [end, left, right])


random.seed(0)
player, balls = make_world(args.num_balls, W, H)
tracker = CollisionTracker()
prev, goal, prev_goal = None, None, None
goal_used = free_used = goals = steps = 0

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("the combined priority map, live")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (
                e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
            running = False

    if goal is not None and (goal_used >= 300 or
            math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS):
        if goal_used < 300:
            goals += 1
        prev_goal = goal                 # the one-back trace of record
        goal = None                      # enter the anticipation window
        free_used = 0
        # fixation start, as in the priming trials: teleport to center so
        # the anticipation drift is visible (standing at the reached goal,
        # drifting toward it would mean standing still)
        player.x, player.y = W // 2, H // 2
        tracker.in_contact.clear()
        prev = None
    if goal is None:
        if prev_goal is None or free_used >= args.window:
            goal = sample_goal(player, W, H, min_dist=250)
            goal_used = 0
            if prev_goal is not None and random.random() < 0.5:
                goal = prev_goal         # REPEAT trial: goal at the trace
        else:
            free_used += 1

    for b in balls:
        if b is not player:
            b.move()
    d, inter = lidar_scan(player, balls, W, H, 360)
    if prev is None:
        prev = d.copy()
    if goal is not None:                 # goal visible: history gated OFF
        gvec, hvec = (goal[0] - player.x, goal[1] - player.y), (0.0, 0.0)
    elif prev_goal is not None:          # anticipation: history is the map
        gvec, hvec = (0.0, 0.0), (prev_goal[0] - player.x,
                                  prev_goal[1] - player.y)
    else:
        gvec, hvec = (0.0, 0.0), (0.0, 0.0)
    obs = torch.tensor(prev + d + list(gvec),
                       dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        f_obs, f_goal = model.compute_fields(obs)
        hv = torch.tensor(hvec, dtype=torch.float32).unsqueeze(0)
        f_hist = model.geometric_field(hv, model.goal_gain)
        field = (f_obs + f_goal + args.beta * f_hist)[0].numpy()
        action = model.sense_action_layers(
            torch.tensor(field, dtype=torch.float32).unsqueeze(0))
    fx = min(max(action[0, 0].item(), -10), 10)
    fy = min(max(action[0, 1].item(), -10), 10)
    player.x = max(player.radius, min(W - player.radius, player.x + int(fx)))
    player.y = max(player.radius, min(H - player.radius, player.y + int(fy)))
    prev = d.copy()
    steps += 1
    goal_used += 1
    tracker.update(player, balls)

    # ---- draw: evaluate.py style, rays colored by the combined field ----
    screen.fill(WHITE)
    for b in balls:
        if b is not player:
            pygame.draw.circle(screen, (60, 90, 210), (int(b.x), int(b.y)),
                               b.radius)
    vmax = max(float(abs(field).max()), 1e-6)
    for i, pt in enumerate(inter):
        if pt:
            a = math.radians(i)
            start = (player.x + player.radius * math.cos(a),
                     player.y + player.radius * math.sin(a))
            n = float(field[i]) / vmax            # [-1, 1]
            hue = 0.165 * (1 + n)                 # red (push) -> green (pull)
            r, g, bl = colorsys.hsv_to_rgb(hue, 1, 1)
            pygame.draw.line(screen, (int(r*255), int(g*255), int(bl*255)),
                             start, pt, 1)
    if goal is not None:
        pygame.draw.circle(screen, GREEN, (int(goal[0]), int(goal[1])),
                           GOAL_RADIUS, 3)
    if prev_goal is not None:
        pygame.draw.circle(screen, PURPLE,
                           (int(prev_goal[0]), int(prev_goal[1])), 6)
    pygame.draw.circle(screen, RED, (player.x, player.y), player.radius)
    if math.hypot(fx, fy) > 0:
        draw_arrow(screen, RED, (player.x, player.y),
                   (player.x + fx * 6, player.y + fy * 6))
    phase = ("GOAL VISIBLE: history gated off" if goal is not None
             else f"ANTICIPATION {free_used}/{args.window}: history drives the map")
    screen.blit(font.render(
        f"{phase}   beta = {args.beta:g}   goals {goals}   "
        f"collisions {tracker.count}", True, (0, 0, 0)), (10, 10))
    pygame.display.flip()
    clock.tick(args.fps)

pygame.quit()
