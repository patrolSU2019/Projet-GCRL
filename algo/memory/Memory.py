# Replay buffer for RL

import numpy as np


class ReplayBuffer:
    """A simple FIFO experience replay buffer."""

    def __init__(self, buffer_size):
        self.states = np.zeros(buffer_size, dtype=np.int32)
        self.next_states = np.zeros(buffer_size, dtype=np.int32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.goals = np.zeros(buffer_size, dtype=np.int32)
        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size

    def store(self, state, act, rew, next_state, goal):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.goals[self.ptr] = goal
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def store_tarjectorys(self, trajectorys):
        for state, action, reward, next_state, goal in trajectorys:
            self.store(state, action, reward, next_state, goal)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states=self.states[idxs],
            next_states=self.next_states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            goals=self.goals[idxs],
        )
        return batch
