import math

import torch
import torch.nn as nn

from model.es2 import SpatialSenseBlock


class GoalEs2Model(nn.Module):
    """Goal-conditioned ES2: obstacle field + goal field + history field.

    Input: previous LiDAR scan | current LiDAR scan | current goal dx/dy |
    history trace dx/dy.  The trainer constructs the history trace by
    replaying goal-spawn positions in world coordinates with a learned
    leaky rate eta_H, then subtracting the current player position.

    The LiDAR path is treated as the obstacle-perception module.  The
    current goal and the world-space history trace are converted into direct
    geometric fields: cosine alignment with each ray, scaled by learned
    distance gain.  The history field has an explicit strength parameter so
    it can bias the visible goal without overwhelming it.
    """

    def __init__(self, num_features=360, num_actions=2, sensing_range=800.0):
        super().__init__()
        self.num_features = num_features
        self.sensing_range = sensing_range

        self.spatial_sense_block = SpatialSenseBlock(
            sensing_range=sensing_range, num_features=num_features
        )
        self.goal_gain = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.history_gain = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.beta_H = nn.Parameter(torch.tensor(0.1))
        self.raw_eta_H = nn.Parameter(torch.tensor(-2.9444))  # sigmoid ~= 0.05

        self.sense_action_layers = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_actions),
        )

        angles = torch.tensor(
            [math.radians(i * (360 / num_features)) for i in range(num_features)]
        )
        self.register_buffer("ray_cos", torch.cos(angles))
        self.register_buffer("ray_sin", torch.sin(angles))
        self._initialize_parameters()

    @property
    def eta_H(self):
        return torch.sigmoid(self.raw_eta_H)

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def geometric_field(self, vec, gain_net):
        """Write one relative position vector into a geometric ray-wise field."""
        dist = torch.norm(vec, dim=1, keepdim=True)
        unit = vec / (dist + 1e-6)
        alignment = unit[:, 0:1] * self.ray_cos + unit[:, 1:2] * self.ray_sin
        gain = gain_net(dist / self.sensing_range)
        return gain * alignment

    def compute_fields(self, input):
        scan_input = input[:, : 2 * self.num_features]
        goal = input[:, 2 * self.num_features : 2 * self.num_features + 2]
        hist = input[:, 2 * self.num_features + 2 :]

        obstacle_field = self.spatial_sense_block(scan_input)
        goal_field = self.geometric_field(goal, self.goal_gain)
        history_field = self.geometric_field(hist, self.history_gain)
        return obstacle_field, goal_field, history_field

    def forward(self, input):
        obstacle_field, goal_field, history_field = self.compute_fields(input)
        field = obstacle_field + goal_field + self.beta_H * history_field
        return self.sense_action_layers(field)
