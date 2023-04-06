import numpy as np
import torch
from bbrl.workspace import Workspace


class her:
    def __init__(self, reward_fun, replay_k):
        self.replay_k = replay_k
        self.reward_fun = reward_fun

    def sample(self, workspace):
        raise NotImplementedError


class FutureHer(her):
    def __init__(self, reward_fun, replay_k):
        super().__init__(reward_fun, replay_k)

    def sample(self, transitions: Workspace):
        her_transitions = Workspace()
        i = 0
        for t in range(transitions.time_size()):
            for k in range(self.replay_k):
                future = np.random.randint(t, transitions.time_size())
                desired_goal = transitions.get("env/desired_goal", t + future)
                state = transitions.get("env/observation", t)
                action = transitions.get("action", t)
                next_state = transitions.get("env/observation", t + 1)
                achieved_goal = transitions.get("env/achieved_goal", t)
                done = achieved_goal == desired_goal
                reward = self.reward_fun(achieved_goal, desired_goal)
                her_transitions.set("env/observation", i, state)
                her_transitions.set("env/observation", i + 1, next_state)
                her_transitions.set("env/desired_goal", i, desired_goal)
                her_transitions.set("env/achieved_goal", i, achieved_goal)
                her_transitions.set("env/desired_goal", i + 1, desired_goal)
                her_transitions.set(
                    "env/achieved_goal",
                    i + 1,
                    transitions.get("env/achieved_goal", t + 1),
                )
                her_transitions.set("action", i, action)
                her_transitions.set("action", i + 1, transitions.get("action", t + 1))
                her_transitions.set("reward", i, torch.Tensor([0.0]))
                her_transitions.set("reward", i + 1, reward)
                her_transitions.set("done", i, torch.Tensor([False]))
                her_transitions.set("done", i + 1, torch.Tensor([done]))
                i += 2

        return her_transitions
