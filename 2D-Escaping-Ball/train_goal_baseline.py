"""Train the goal-conditioned MLP or Transformer baseline on the same
reach-avoid demonstrations as train_goal_es2.py (no ES2 k-alternation)."""

import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset.dataloader_goal import get_goal_data
from model.goal_mlp import GoalMLPModel
from model.goal_transformer import GoalTransformerModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=["mlp", "transformer"], required=True)
    parser.add_argument("--data_path", type=str, default="dataset/data_goal.csv")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=150)

    # Transformer-specific parameters (match evaluate.py defaults)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)

    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = f"pretrained/goal_{args.model}.pth"
    if args.log_path is None:
        args.log_path = f"loss_goal_{args.model}.csv"
    print(f"Using device: {args.device}")

    scan_columns = [f"scan_{i}" for i in range(args.num_features)]
    data_loader = get_goal_data(
        file_path=args.data_path,
        scan_columns=scan_columns,
        goal_columns=["goal_dx", "goal_dy"],
        target_columns=["fx", "fy"],
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.model == "mlp":
        model = GoalMLPModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)
    else:
        model = GoalTransformerModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            sensing_range=args.sensing_range,
        ).to(args.device)

    if os.path.exists(args.model_path):
        print(f"Loading model parameters from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    else:
        print("No pre-trained model found. Starting training from scratch.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_loss = float("inf")
    epoch_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        with tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch"
        ) as progress_bar:
            for inputs, targets in progress_bar:
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.5f}")
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"New best loss: {best_loss:.4f}. Model saved to {args.model_path}")

    with open(args.log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss"])
        writer.writerows([[loss] for loss in epoch_losses])
    print(f"Losses saved to {args.log_path}")


if __name__ == "__main__":
    main()
