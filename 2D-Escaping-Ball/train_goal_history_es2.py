"""Train GoalHistoryEs2Model on ordered goal-history demonstrations."""

import argparse
import csv
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model.goal_history_es2 import GoalHistoryEs2Model


def load_tensors(path, num_features, device):
    df = pd.read_csv(path).reset_index(drop=True)
    scan_cols = [f"scan_{i}" for i in range(num_features)]
    scans = torch.tensor(df[scan_cols].values, dtype=torch.float32, device=device)
    goal = torch.tensor(df[["goal_dx", "goal_dy"]].values, dtype=torch.float32, device=device)
    player = torch.tensor(df[["player_x", "player_y"]].values, dtype=torch.float32, device=device)
    goal_xy = torch.tensor(df[["goal_x", "goal_y"]].values, dtype=torch.float32, device=device)
    spawn = torch.tensor(df["goal_spawn"].values.astype(bool), device=device)
    target = torch.tensor(df[["fx", "fy"]].values, dtype=torch.float32, device=device)
    episode = df["episode"].values
    pairs = [i for i in range(len(df) - 1) if episode[i] == episode[i + 1]]
    return df, scans, goal, player, goal_xy, spawn, target, torch.tensor(pairs, device=device)


def build_history_vectors(df, player, goal_xy, spawn, eta, width, height):
    out = []
    center = torch.tensor([width / 2, height / 2], dtype=player.dtype, device=player.device)
    trace = center
    last_ep = None
    for i, ep in enumerate(df["episode"].values):
        if ep != last_ep:
            trace = center
            last_ep = ep
        if bool(spawn[i].item()):
            trace = (1 - eta) * trace + eta * goal_xy[i]
        out.append(trace - player[i])
    return torch.stack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/data_goal_history.csv")
    parser.add_argument("--model_path", type=str, default="pretrained/goal_history_es2.pth")
    parser.add_argument("--log_path", type=str, default="loss_goal_history_es2.csv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--disable_history",
        action="store_true",
        help="freeze beta_H at 0 so the history field never enters the sum",
    )
    args = parser.parse_args()

    df, scans, goal, player, goal_xy, spawn, target, pairs = load_tensors(
        args.data_path, args.num_features, args.device
    )
    model = GoalHistoryEs2Model(
        num_features=args.num_features,
        num_actions=args.num_actions,
        sensing_range=args.sensing_range,
    ).to(args.device)
    if os.path.exists(args.model_path):
        print(f"Loading model parameters from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    if args.disable_history:
        with torch.no_grad():
            model.beta_H.zero_()
        model.beta_H.requires_grad = False
        print("history DISABLED (beta_H frozen at 0)")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    losses = []
    best = float("inf")

    for epoch in range(args.num_epochs):
        hist = build_history_vectors(
            df, player, goal_xy, spawn, model.eta_H, args.width, args.height
        )
        i0 = pairs
        i1 = pairs + 1
        inputs = torch.cat([scans[i0], scans[i1], goal[i1], hist[i1]], dim=1)
        pred = model(inputs)
        loss = criterion(pred, target[i1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if loss.item() < best:
            best = loss.item()
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_path)
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            print(
                f"epoch {epoch}: loss={loss.item():.5f} "
                f"eta_H={model.eta_H.item():.4f} beta_H={model.beta_H.item():+.4f}"
            )

    with open(args.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss"])
        writer.writerows([[x] for x in losses])
    print(f"best loss {best:.5f}; model saved to {args.model_path}")


if __name__ == "__main__":
    main()
