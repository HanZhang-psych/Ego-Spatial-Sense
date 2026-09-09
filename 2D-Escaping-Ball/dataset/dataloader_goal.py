import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class GoalSenseDataset(Dataset):
    """
    Dataset for goal-directed reach-avoid demonstrations.
    Each sample concatenates:
      - the previous frame's LiDAR scan
      - the current frame's LiDAR scan
      - the current frame's relative goal vector (goal_dx, goal_dy)
    The target is the current frame's (fx, fy).
    Pairs that cross an episode boundary are excluded.
    """

    def __init__(self, data, scan_columns, goal_columns, target_columns, device):
        data = data.reset_index(drop=True)
        self.device = device
        self.scan_columns = scan_columns
        self.goal_columns = goal_columns
        self.target_columns = target_columns

        # Valid indices: consecutive frame pairs within the same episode.
        episodes = data["episode"].values
        self.pairs = [
            i for i in range(len(data) - 1) if episodes[i] == episodes[i + 1]
        ]
        self.data = data

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i = self.pairs[idx]
        first_sample = self.data.iloc[i]
        next_sample = self.data.iloc[i + 1]

        inputs = torch.tensor(
            list(first_sample[self.scan_columns].values)
            + list(next_sample[self.scan_columns].values)
            + list(next_sample[self.goal_columns].values),
            dtype=torch.float32,
        ).to(self.device)

        target = torch.tensor(
            [next_sample[col] for col in self.target_columns], dtype=torch.float32
        ).to(self.device)

        return inputs, target


def get_goal_data(
    file_path,
    scan_columns,
    goal_columns,
    target_columns,
    batch_size=64,
    device="cpu",
):
    data = pd.read_csv(file_path)
    dataset = GoalSenseDataset(data, scan_columns, goal_columns, target_columns, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
