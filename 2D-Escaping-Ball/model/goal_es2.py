import math

import torch
import torch.nn as nn

from model.es2 import SpatialSenseBlock


class GoalEs2Model(nn.Module):
    """Goal-conditioned ES2 model for the reach-avoid task.

    Input: (batch, 2 * num_features + 2) =
        previous LiDAR scan | current LiDAR scan | relative goal (dx, dy) in pixels.

    The obstacle-driven spatial sense field is computed exactly as in Es2Model.
    The goal enters as a second, top-down field: a per-ray alignment pattern
    cos(theta_i - goal_bearing) scaled by a learned gain conditioned on the
    normalized goal distance.  The two fields are summed into one attention
    field F(q | s_e) — asymmetric toward the goal sector — which the action
    layers map to (fx, fy).
    """

    def __init__(self, num_features=360, num_actions=2, sensing_range=800.0):
        super(GoalEs2Model, self).__init__()

        self.num_features = num_features
        self.sensing_range = sensing_range

        self.spatial_sense_block = SpatialSenseBlock(
            sensing_range=sensing_range, num_features=num_features
        )

        # Learned gain on the goal field, conditioned on goal distance.
        self.goal_gain = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

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

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_fields(self, input):
        """Return (obstacle_field, goal_field), each (batch, num_features)."""
        scan_input = input[:, : 2 * self.num_features]
        goal = input[:, 2 * self.num_features :]

        obstacle_field = self.spatial_sense_block(scan_input)

        goal_dist = torch.norm(goal, dim=1, keepdim=True)
        goal_unit = goal / (goal_dist + 1e-6)
        # Per-ray alignment with the goal bearing: cos(theta_i - goal_bearing)
        alignment = goal_unit[:, 0:1] * self.ray_cos + goal_unit[:, 1:2] * self.ray_sin
        gain = self.goal_gain(goal_dist / self.sensing_range)
        goal_field = gain * alignment

        return obstacle_field, goal_field

    def forward(self, input):
        obstacle_field, goal_field = self.compute_fields(input)
        spatial_sense = obstacle_field + goal_field
        action = self.sense_action_layers(spatial_sense)
        return action
