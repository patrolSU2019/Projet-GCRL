import torch
from bbrl import Agent

from .utiles import build_mlp


class GCQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, goal_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + goal_dim] + list(hidden_layers) + [action_dim],
            activation=torch.nn.ReLU(),
        )

    def forward(self, t, choose_action=True, **kwargs):
        achieved_goal = self.get(("env/env_obs", t))

        # desired_goal Tensor of shape (N, goal_dim)
        desired_goal = self.get(("env/desired_goal", t))

        # Concatenate the observation and the desired goal
        obs = torch.cat([achieved_goal, desired_goal], dim=-1)

        q_values = self.model(obs).squeeze(-1)
        self.set(("q_values", t), q_values)

        if choose_action:
            action = q_values.argmax(-1)
            self.set(("action", t), action)
