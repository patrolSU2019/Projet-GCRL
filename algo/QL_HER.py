import matplotlib.pyplot as plt
from .memory.Memory import ReplayBuffer
import numpy as np
from bbrl_gym.envs.maze_mdp import MazeMDPEnv
from tqdm import tqdm
import gym
import bbrl_gym


class QLearingHER:
    """
    tabular q-learning with experience replay
    """

    def __init__(
        self,
        env: MazeMDPEnv,
        buffer_size=500,
        batch_size=32,
        epsilon=0.1,
        learning_rate=0.5,
        max_steps=40,
        max_episodes=500,
        her_k=4,
    ):
        self.env = env
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.n
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = env.gamma
        self.max_steps = max_steps
        env.set_timeout(max_steps)
        self.max_episodes = max_episodes
        self.q_table = np.zeros((self.state_dim, self.state_dim, self.action_dim))
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.her_k = her_k

    def select_action(self, state, goal):
        if np.random.rand() <= self.epsilon or np.max(self.q_table[state, goal]) == 0:
            return np.random.choice(self.action_dim)
        q_values = self.q_table[state, goal]
        return np.argmax(q_values)

    def update(self, batch):
        for state, action, reward, next_state, goal in zip(
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["goals"],
        ):
            delta = (
                reward
                + self.discount_factor * np.max(self.q_table[next_state, goal])
                - self.q_table[state, goal, action]
            )
            self.q_table[state, goal, action] = (
                self.q_table[state, goal, action] + self.learning_rate * delta
            )

    def train(self):
        time_list = []
        q_list = []
        for i in range(10):
            with tqdm(
                total=int(self.max_episodes / 10), desc="Iteration %d" % i
            ) as pbar:
                for i_episode in range(int(self.max_episodes / 10)):
                    state = self.env.reset(uniform=True)
                    transitions = []
                    her_transitions = []
                    disered_goal = np.random.randint(self.state_dim - 1)
                    self.env.change_last_states([disered_goal])
                    done = False
                    while not done:
                        action = self.select_action(state, disered_goal)
                        next_state, reward, done, _ = self.env.step(action)
                        transitions.append(
                            (state, action, reward, next_state, disered_goal)
                        )
                        state = next_state
                    self.replay_buffer.store_tarjectorys(transitions)
                    time_list.append(self.env.mdp.timestep)
                    for t in range(len(transitions)):
                        for k in range(self.her_k):
                            future = np.random.randint(t, len(transitions))
                            achieved_goal = transitions[future][4]
                            state = transitions[t][0]
                            action = transitions[t][1]
                            next_state = transitions[t][3]
                            self.env.change_last_states([achieved_goal])
                            reward = self.env.mdp.r[state, action]
                            her_transitions.append(
                                (state, action, reward, next_state, achieved_goal)
                            )
                    self.replay_buffer.store_tarjectorys(her_transitions)
                    if self.replay_buffer.size > self.batch_size:
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        self.update(batch)
                    q_list.append(np.linalg.norm(self.q_table))
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": "%d"
                                % (self.max_episodes / 10 * i + i_episode + 1),
                                "q_norm": "%.3f" % np.mean(q_list[-10:]),
                            }
                        )
                    pbar.update(1)
        self.env.close()
        return q_list, time_list

    def render_policy(self):
        for i in range(self.state_dim - 1):
            self.env.change_last_states([i])
            self.env.draw_v_pi(
                self.q_table[:, i, :], np.argmax(self.q_table[:, i, :], axis=1)
            )


def main():
    env = gym.make("MazeMDP-v0", kwargs={"width": 3, "height": 3, "ratio": 0.2})
    env.reset()
    env.init_draw("The maze")
    algo = QLearingHER(env, max_episodes=1500)
    q_list, time_list = algo.train()
    # plot the return curve

    plt.plot(range(len(q_list)), q_list)
    plt.xlabel("Episodes")
    plt.ylabel("Q-Norm")
    plt.show()


if __name__ == "__main__":
    main()
