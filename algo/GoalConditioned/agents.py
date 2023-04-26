from copy import deepcopy
from os import truncate
from bbrl import Agent
from bbrl.workspace import Workspace
from sympy import N
import torch


class GoalAgent(Agent):
    def __init__(self, n_env, goal_dim, goal_range, goal_type="int"):
        super().__init__()
        self.n_env = n_env
        self.goal_dim = goal_dim
        self.goal_range = goal_range
        self.goal = None
        self.goal_type = goal_type

    def _custom_rand_int(self, size, ranges):
        result = torch.empty(size).type(torch.int64)
        for j in range(size[1]):
            lower, upper = ranges[j][0], ranges[j][1]
            result[:, j] = torch.randint(lower, upper, (1, size[0])).type(torch.int64)
        return result

    def _custom_rand_float(self, size, ranges):
        result = torch.empty(size)
        for j in range(size[1]):
            lower, upper = ranges[j][0], ranges[j][1]
            result[:, j] = torch.rand(size[0]) * (upper - lower) + lower
        return result

    def forward(self, t, **kwargs):
        if t == 0:
            match self.goal_type:
                case "int":
                    goal = self._custom_rand_int(
                        (self.n_env, self.goal_dim), self.goal_range
                    )
                case "float":
                    goal = self._custom_rand_float(
                        (self.n_env, self.goal_dim), self.goal_range
                    )
                case _:
                    raise ValueError("Invalid goal type")
            self.set(("env/desired_goal", t), goal)
            self.goal = goal
        else:
            assert self.goal is not None, "Goal not set"
            self.set(("env/desired_goal", t), self.goal)


class RewardAgent(Agent):
    def __init__(self, compute_reward, compute_terminated):
        super().__init__()
        self.compute_reward = compute_reward
        self.compute_terminated = compute_terminated

    def forward(self, t, **kwargs):
        if t == 0:
            n_env = self.get(("env/env_obs", t)).shape[0]
            self.set(("env/return", t), torch.zeros(n_env).float())
            return
        achieved_goal = self.get(("env/env_obs", t))
        desired_goal = self.get(("env/desired_goal", t))
        assert self.compute_reward is not None, "Reward function not set"
        reward = self.compute_reward(achieved_goal, desired_goal)
        self.set(("env/reward", t - 1), reward)
        self.set(("env/reward", t), reward)
        _return = self.get(("env/return", t - 1))
        _return += reward
        self.set(("env/return", t), _return)

        # terminated = self.compute_terminated(achieved_goal, desired_goal)
        # done = self.get(("env/done", t))
        # self.set(("env/truncated", t), done)
        # done = terminated | done
        # self.set(("env/done", t), done)


class HerAgent(Agent):
    def __init__(self):
        super().__init__()
        self._pos = 0

    def forward(self, **kwargs):
        pass


class HerFinal(HerAgent):
    def __init__(self):
        super().__init__()
        self.goal = None

    def forward(self, t, trajectory, **kwargs):
        if t == 0:
            assert self.workspace is not None, "Workspace not set"
            self.goal = trajectory["env/env_obs"][-1][:, [0]]
        self.set(("env/desired_goal", t), self.goal)
        self.set(("env/env_obs", t), trajectory.get("env/env_obs", t))
        self.set(("env/done", t), trajectory.get("env/done", t))
        self.set(("action", t), trajectory.get("action", t))
