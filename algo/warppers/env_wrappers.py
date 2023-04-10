import random
import numpy as np
import gym
from gym import spaces
from torch import Tensor


class MazeMDPContinuousGoalWrapper(gym.Wrapper):
    """
    Specific wrapper to turn the Tabular MazeMDP into a continuous state version
    """

    def __init__(self, env):
        super(MazeMDPContinuousGoalWrapper, self).__init__(env)
        # Building a new continuous observation space from the coordinates of each state
        high = np.array(
            [
                env.coord_x.max() + 1,
                env.coord_y.max() + 1,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                env.coord_x.min(),
                env.coord_y.min(),
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low, high),
                "achieved_goal": spaces.Discrete(env.nb_states),
                "desired_goal": spaces.Discrete(env.nb_states),
            }
        )

    def is_continuous_state(self):
        # By contrast with the wrapped environment where the state space is discrete
        return True

    def reset(self):
        obs = self.env.reset()
        x = self.env.coord_x[obs]
        y = self.env.coord_y[obs]
        xc = x + random.random()
        yc = y + random.random()
        continuous_obs = [xc, yc]
        disired_goal = [self.env.mdp.terminal_states[0]]
        achieved_goal = [self.env.mdp.current_state]

        return {
            "observation": continuous_obs,
            "achieved_goal": achieved_goal,
            "desired_goal": disired_goal,
        }

    def step(self, action):
        # Turn the discrete state into a pair of continuous coordinates
        # Take the coordinates of the state and add a random number to x and y to
        # sample anywhere in the [1, 1] cell...
        next_state, reward, done, info = self.env.step(action)
        x = self.env.coord_x[next_state]
        y = self.env.coord_y[next_state]
        xc = x + random.random()
        yc = y + random.random()
        next_continuous = [xc, yc]
        desired_goal = [self.env.mdp.terminal_states[0]]
        achieved_goal = [self.env.mdp.current_state]
        return (
            {
                "observation": next_continuous,
                "achieved_goal": achieved_goal,
                "desired_goal": desired_goal,
            },
            reward,
            done,
            info,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes it dependent on a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, terminated, truncated, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        if achieved_goal == desired_goal:
            return 1.0
        else:
            return 0.0

    def change_goal(self, goal):
        if isinstance(goal, Tensor):
            goal = goal.tolist()
        # print("Changing goal to ", goal)
        self.change_last_states(goal)

    def int_goal_to_continuous(self, goal):
        x = self.env.coord_x[goal]
        y = self.env.coord_y[goal]
        return [x, y]
