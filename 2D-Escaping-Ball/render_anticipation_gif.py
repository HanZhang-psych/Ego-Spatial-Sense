"""Render the selection-history learning GIF (fixation-start structure).

Simulates the frozen goal_es2 agent with the runtime spread trace
(anticipation_experiment.py mechanics) and renders two segments — early
trials (trace empty) and late trials (trace learned) — with title cards.
Each trial: reset to the center cross, an anticipation period with no
goal (the red trail shows the excursion driven by the trace alone; with
beta=0.15 it settles near the frequent region's edge — see the
edge-equilibrium diagnostic in RESULTS_reach_avoid.md), then goal onset
and pursuit.

Frames are written to --frames_dir; assemble with e.g.:
    ffmpeg -framerate 25 -i frames/f%05d.png \
      -vf "scale=560:560:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
      anticipation_learning.gif
"""

import argparse
import math
import os
import random

import pygame
import torch

from anticipation_experiment import QUAD, sample_goal_biased, spread_goal_field
from evaluate_reach_avoid import load_model
from reach_avoid_common import GOAL_RADIUS, CollisionTracker, lidar_scan, make_world

CENTER = (400, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/goal_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--bias_p", type=float, default=0.7)
    parser.add_argument("--min_spawn_dist", type=float, default=150)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--antic_steps", type=int, default=80)
    parser.add_argument("--early_trials", type=int, nargs=2, default=[0, 4])
    parser.add_argument("--late_trials", type=int, nargs=2, default=[30, 34])
    parser.add_argument("--frame_every", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--frames_dir", type=str, default="frames_anticipation")
    args = parser.parse_args()

    model, _ = load_model(args)
    os.makedirs(args.frames_dir, exist_ok=True)

    seg_a = range(*args.early_trials)
    seg_b = range(*args.late_trials)
    last = max(seg_b)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    pygame.init()
    surf = pygame.Surface((args.width, args.height))
    font = pygame.font.SysFont("Arial", 22)
    big = pygame.font.SysFont("Arial", 30)
    player, balls = make_world(10, args.width, args.height)
    tracker = CollisionTracker()
    state = {"prev": None, "frame": 0}
    trace_points = []
    trail = []

    def draw(goal, l1, l2, capture, step_i):
        if not capture or step_i % args.frame_every != 0:
            return
        surf.fill((255, 255, 255))
        pygame.draw.rect(surf, (255, 249, 224),
                         (QUAD[0], QUAD[1], QUAD[2] - QUAD[0], QUAD[3] - QUAD[1]))
        pygame.draw.rect(surf, (200, 180, 120),
                         (QUAD[0], QUAD[1], QUAD[2] - QUAD[0], QUAD[3] - QUAD[1]), 2)
        pygame.draw.line(surf, (0, 0, 0), (CENTER[0] - 12, CENTER[1]),
                         (CENTER[0] + 12, CENTER[1]), 2)
        pygame.draw.line(surf, (0, 0, 0), (CENTER[0], CENTER[1] - 12),
                         (CENTER[0], CENTER[1] + 12), 2)
        ov = pygame.Surface((args.width, args.height), pygame.SRCALPHA)
        for age, p in enumerate(trace_points):
            w = (1 - args.eta) ** age
            pygame.draw.circle(ov, (0, 160, 60, int(30 + 150 * w)),
                               (int(p[0]), int(p[1])), 7)
        if len(trail) > 1:
            pygame.draw.lines(ov, (220, 60, 60, 180), False, trail, 3)
        surf.blit(ov, (0, 0))
        for b in balls:
            if b is not player:
                pygame.draw.circle(surf, b.color, (int(b.x), int(b.y)), b.radius)
        if goal:
            pygame.draw.circle(surf, (0, 200, 0), (int(goal[0]), int(goal[1])),
                               GOAL_RADIUS, 4)
        pygame.draw.circle(surf, (255, 0, 0), (int(player.x), int(player.y)),
                           player.radius)
        surf.blit(font.render(l1, True, (0, 0, 0)), (10, 8))
        surf.blit(font.render(l2, True, (60, 60, 60)), (10, 34))
        surf.blit(font.render(
            "cross = start (reset each trial)   shaded = frequent-goal region   "
            "green dots = learned trace", True, (120, 100, 40)), (10, 772))
        pygame.image.save(surf, f"{args.frames_dir}/f{state['frame']:05d}.png")
        state["frame"] += 1

    def title_card(lines, nframes=55):
        surf.fill((255, 255, 255))
        for i, ln in enumerate(lines):
            t = big.render(ln, True, (0, 0, 0))
            surf.blit(t, (args.width // 2 - t.get_width() // 2, 330 + i * 44))
        for _ in range(nframes):
            pygame.image.save(surf, f"{args.frames_dir}/f{state['frame']:05d}.png")
            state["frame"] += 1

    def sim_step(goal):
        for b in balls:
            if b is not player:
                b.move()
        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if state["prev"] is None:
            state["prev"] = d.copy()
        gdx, gdy = (goal[0] - player.x, goal[1] - player.y) if goal else (0.0, 0.0)
        obs = torch.tensor(state["prev"] + d + [gdx, gdy],
                           dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            of, gf = model.compute_fields(obs)
            if trace_points:
                ws = [(1 - args.eta) ** a for a in range(len(trace_points))]
                tot = sum(ws)
                gf_h = spread_goal_field(model, trace_points,
                                         [w / tot for w in ws], player.x, player.y)
                field = of + gf + args.beta * gf_h
            else:
                field = of + gf
            a = model.sense_action_layers(field)
        fx = min(max(a[0][0].item(), -10), 10)
        fy = min(max(a[0][1].item(), -10), 10)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        state["prev"] = d.copy()
        tracker.update(player, balls)

    title_card(["Early trials (1-4)",
                "Each trial: reset to the cross, wait with NO goal,",
                "then a goal appears. Trace empty - no bias yet"])
    for gi in range(last + 1):
        cap = gi in seg_a or gi in seg_b
        if gi == min(seg_b):
            title_card(["After 30 trials",
                        "While waiting, the agent shifts toward the learned",
                        "region and holds position near its edge -",
                        "so goals appearing there are already close"])
        player.x, player.y = CENTER
        state["prev"] = None
        tracker.in_contact.clear()
        trail.clear()
        for fs in range(args.antic_steps):
            sim_step(None)
            trail.append((int(player.x), int(player.y)))
            draw(None, f"trial {gi + 1} - ANTICIPATION (no goal exists)",
                 "red trail = shift from start, driven by memory alone", cap, fs)
        goal = sample_goal_biased(player, args, biased=True)
        trace_points.insert(0, (goal[0], goal[1]))
        del trace_points[60:]
        steps = 0
        while steps < 800:
            sim_step(goal)
            steps += 1
            draw(goal, f"trial {gi + 1} - goal onset, pursuing",
                 f"steps: {steps}", cap, steps)
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                break
    print(f"{state['frame']} frames written to {args.frames_dir}/")


if __name__ == "__main__":
    main()
