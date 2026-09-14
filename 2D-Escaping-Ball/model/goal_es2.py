import math

import torch
import torch.nn as nn

GRID = 8  # the spatial memory is a GRID x GRID leaky map over the arena


class AdditiveSenseBlock(nn.Module):
    """The obstacle sense of record: a proximity channel and a looming
    channel, each with one learned scalar gain, summed.

    Each LiDAR distance first passes through a sigmoid with learned per-ray
    sensitivity `k` (the psychophysics of closeness: near = strong, far =
    weak, saturating at both ends).  The obstacle field is then a linear
    combination of two channels built from the previous and current
    closeness:

        obstacle(dir) = g_C * closeness_now
                      + g_L * (closeness_now - closeness_prev) * closeness_now

    - `g_C * closeness_now` is the PROXIMITY channel: how close something is
      right now (static standoff).
    - `g_L * (closeness_now - closeness_prev) * closeness_now` is the
      LOOMING channel: the change in closeness (approach) gated by current
      closeness, so approach only counts where something is already near.

    This replaces the earlier looming-only block (model/es2.py
    SpatialSenseBlock, obstacle = proj(closeness_now - closeness_prev) *
    closeness_now, a learned nonlinear projection of the change signal).
    The two channels are co-equal evidence sources combined additively - the
    same shape as the visual-search model's F = g_C*color + g_F*shape -
    rather than a single gated product, and each gain is one interpretable
    scalar instead of an MLP.  Behavioral cloning on the expert
    demonstrations settles on g_C ~ 1.0, g_L ~ 0 (proximity carries the
    task; the change signal is nearly redundant when proximity is
    available), while matching or slightly beating the looming-only agent on
    throughput and safety across the demonstrated and stress regimes.

    The input is the full two-scan vector (prev | current) so the dataloader
    and evaluation harness are unchanged.
    """

    def __init__(self, sensing_range: float, num_features: int = 360):
        super().__init__()
        k_init_value = self.inverse_sigmoid(sensing_range, torch.tensor([0.01]))
        self.k = nn.Parameter(torch.full((num_features,), k_init_value.item()))
        self.g_C = nn.Parameter(torch.tensor(1.0))   # proximity gain
        self.g_L = nn.Parameter(torch.tensor(1.0))   # looming gain

    def inverse_sigmoid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (-1 / x) * torch.log(y / (2 - y))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

    def forward(self, input):
        feature_size = input.size(1) // 2
        sample = input[:, :feature_size]
        sample_next = input[:, feature_size:]

        closeness_prev = self.sigmoid(sample)
        closeness_now = self.sigmoid(sample_next)
        looming = (closeness_now - closeness_prev) * closeness_now

        return self.g_C * closeness_now + self.g_L * looming


class GoalEs2Model(nn.Module):
    """Goal-conditioned ES2: obstacle field + goal field + beta_H * history field.

    Network input: previous LiDAR scan | current LiDAR scan | goal dx/dy.

    The obstacle field is the additive proximity + looming sense of
    AdditiveSenseBlock (obstacle = g_C * closeness_now
    + g_L * (closeness_now - closeness_prev) * closeness_now).

    The history field comes from the model's own spatial memory - a leaky
    accumulator over an 8x8 grid of the arena in WORLD coordinates
    (allocentric storage).  At each goal spawn:

        W *= (1 - eta_H);  W[cell(goal)] += eta_H        (update_memory)

    The readout is egocentric and reuses the trained goal machinery: every
    cell acts as a faint goal weighted by its accumulated mass
    (history_field), and one scalar beta_H sets how strongly the whole
    memory pulls (forward).  eta_H = how fast memory updates, W = what is
    remembered, beta_H = how much it biases behavior; a one-back "previous
    goal" memory is the eta_H = 1 degenerate case.

    The memory is rollout state, not a learned weight: it is absent from
    the state dict, reset_memory() clears it, and behavioral cloning never
    touches it - the policy trains on (scans, goal) alone, so beta_H and
    eta_H can be swept or fitted afterwards without retraining the network
    (evaluate_statistical_learning.py fits both from demonstrations).

    Precedence rule of record: a visible goal OVERRIDES history - pass
    history_field to forward() only during goal-free periods.
    """

    def __init__(self, num_features=360, num_actions=2, sensing_range=800.0,
                 eta_H=0.15):
        super().__init__()
        self.num_features = num_features
        self.sensing_range = sensing_range

        self.spatial_sense_block = AdditiveSenseBlock(
            sensing_range=sensing_range, num_features=num_features
        )
        self.goal_gain = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.beta_H = nn.Parameter(torch.tensor(0.0))
        self.eta_H = eta_H          # leaky rate: plain state, not a weight
        self.memory = None          # set by reset_memory()

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

    # ---- the leaky spatial memory ----

    def reset_memory(self, width, height):
        """Clear the grid and anchor it to an arena of the given size."""
        cw, ch = width / GRID, height / GRID
        self._arena = (width, height)
        self._centers = torch.tensor(
            [[(i + 0.5) * cw, (j + 0.5) * ch]
             for j in range(GRID) for i in range(GRID)],
            dtype=torch.float32,
        )
        self.memory = torch.zeros(GRID * GRID)

    def update_memory(self, goal):
        """Leaky accumulation at a goal spawn: decay everything, boost
        the goal's cell."""
        width, height = self._arena
        ci = min(int(goal[0] // (width / GRID)), GRID - 1)
        cj = min(int(goal[1] // (height / GRID)), GRID - 1)
        self.memory = (1 - self.eta_H) * self.memory
        self.memory[cj * GRID + ci] += self.eta_H

    def history_field(self, pos):
        """Egocentric readout of the memory from world position pos:
        each cell is a faint goal weighted by its mass.  Returns
        (1, num_features); forward() scales it by beta_H."""
        vec = self._centers - torch.tensor(pos, dtype=torch.float32)
        with torch.no_grad():
            G = self.geometric_field(vec, self.goal_gain)
        return (self.memory.unsqueeze(1) * G).sum(0, keepdim=True)

    # ---- the fields ----

    def geometric_field(self, vec, gain_net):
        """Write relative position vectors into geometric ray-wise fields."""
        dist = torch.norm(vec, dim=1, keepdim=True)
        unit = vec / (dist + 1e-6)
        alignment = unit[:, 0:1] * self.ray_cos + unit[:, 1:2] * self.ray_sin
        gain = gain_net(dist / self.sensing_range)
        return gain * alignment

    def compute_fields(self, input):
        scan_input = input[:, : 2 * self.num_features]
        goal = input[:, 2 * self.num_features :]

        obstacle_field = self.spatial_sense_block(scan_input)
        goal_field = self.geometric_field(goal, self.goal_gain)
        return obstacle_field, goal_field

    def forward(self, input, history_field=None):
        obstacle_field, goal_field = self.compute_fields(input)
        field = obstacle_field + goal_field
        if history_field is not None:
            field = field + self.beta_H * history_field
        return self.sense_action_layers(field)
