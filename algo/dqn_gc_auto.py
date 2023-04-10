import copy
import os
import sys
import hydra
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import gym

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace,
# or until a given condition is reached
from bbrl.agents import Agents, RemoteAgent, TemporalAgent, PrintAgent

# AutoResetGymAgent is an agent able to execute a batch of gym environments
# with auto-resetting. These agents produce multiple variables in the workspace:
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’,
# ... When called at timestep t=0, then the environments are automatically reset.
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent

# Not present in the A2C version...
from bbrl.utils.logger import TFLogger
from bbrl.utils.replay_buffer import ReplayBuffer

import bbrl_gym

from warppers import MazeMDPContinuousGoalWrapper

from bbrl_examples.models.loggers import Logger, RewardLogger
from bbrl_examples.models.plotters import Plotter
from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class AuroResetGoalEnvAgent(AutoResetGymAgent):
    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        seed=None,
        action_string="action",
        goal_string="desired_goal",
        output="env/",
    ):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args=make_env_args,
            n_envs=n_envs,
            seed=seed,
            action_string=action_string,
            output=output,
        )
        self.goal_string = goal_string

    def forward(self, t=0, save_render=False, render=False, **kwargs):
        """
        Perform one step by reading the `action`
        """

        observations = []
        rewards = []
        for k, env in enumerate(self.envs):
            if not self.is_running[k] or t == 0:
                goals = self.get((self.goal_string, t))
                assert goals.size()[0] == self.n_envs, "Incompatible number of envs"
                env.change_goal(goals[k])
                observations.append(self._reset(k, save_render, render))

                if t > 0:
                    rew = self.previous_reward[k]
                    rewards.append(rew)
            else:
                assert t > 0
                action = self.get((self.input, t - 1))
                assert action.size()[0] == self.n_envs, "Incompatible number of envs"
                full_obs, reward = self._step(k, action[k], save_render, render)
                self.previous_reward[k] = reward
                observations.append(full_obs)
                rewards.append(reward)

        if t > 0:
            self.set_reward(rewards, t - 1)
            self.set_reward(rewards, t)
        self.set_obs(observations, t)


class GoalAgent(Agent):
    def __init__(self, n_env, goal_dim, goal_range, goal_type="float"):
        super().__init__()
        self.n_env = n_env
        self.goal_dim = goal_dim
        self.goal_range = goal_range

        self.goal_type = goal_type

    def _custom_rand_float(self, size, ranges):
        """Generates a tensor of random numbers with different ranges for each position"""
        result = torch.empty(size)
        for j in range(size[1]):
            lower, upper = ranges[j][0], ranges[j][1]
            result[:, j] = torch.rand(size[0]).mul_(upper - lower).add_(lower)

        return result

    def _custom_rand_int(self, size, ranges):
        result = torch.empty(size).type(torch.int64)
        for j in range(size[1]):
            lower, upper = ranges[j][0], ranges[j][1]
            result[:, j] = torch.randint(lower, upper, (1, size[0])).type(torch.int64)
        return result

    def forward(self, t, **kwargs):
        if self.goal_type == "float":
            goal = self._custom_rand_float((self.n_env, self.goal_dim), self.goal_range)
        elif self.goal_type == "int":
            goal = self._custom_rand_int((self.n_env, self.goal_dim), self.goal_range)
        else:
            raise ValueError("Unknown goal type")
        self.set(("desired_goal", t), goal)


class RewardAgent(Agent):
    def __init__(self, reward_scale):
        super().__init__()
        self.reward_scale = reward_scale

    def forward(self, t, **kwargs):
        if t == 0:
            return
        # desired_goal Tensor of shape (N, goal_dim)
        desired_goal = self.get(("env/desired_goal", t))

        # achieved_goal Tensor of shape (N, goal_dim)
        achieved_goal = self.get(("env/achieved_goal", t))

        reward = (achieved_goal == desired_goal).all(dim=-1).float()

        self.set(("env/reward", t), reward * self.reward_scale)


class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, goal_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + goal_dim] + list(hidden_layers) + [action_dim],
            activation=nn.ReLU(),
        )

    def forward(self, t, choose_action=True, **kwargs):
        # observation Tensor of shape (N, obs_dim)
        obs = self.get(("env/observation", t))

        # desired_goal Tensor of shape (N, goal_dim)
        desired_goal = self.get(("env/desired_goal", t))
        achieved_goal = self.get(("env/achieved_goal", t))
        # done = (desired_goal == achieved_goal).squeeze()
        # if done.ndimension() == 0:
        #     done = done.view(
        #         1,
        #     )
        # self.set(("env/done", t), done)

        # Concatenate the observation and the desired goal
        obs = torch.cat([obs, desired_goal], dim=-1)

        q_values = self.model(obs).squeeze(-1)
        self.set(("q_values", t), q_values)

        if choose_action:
            action = q_values.argmax(-1)
            self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        q_values = self.model(obs).squeeze(-1)
        if stochastic:
            probs = torch.softmax(q_values, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = q_values.argmax(-1)
        return action

    def predict_value(self, obs, action):
        q_values = self.model(obs).squeeze(-1)
        return q_values[action[0].int()]


class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t, **kwargs):
        q_values = self.get(("q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)


def make_gym_env(env_name, env_kwargs):
    return MazeMDPContinuousGoalWrapper(gym.make(env_name, kwargs=env_kwargs))


def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = DiscreteQAgent(
        obs_size, cfg.algorithm.architecture.hidden_size, act_size, cfg.goal.goal_dim
    )
    target_critic = copy.deepcopy(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon_init)
    q_agent = TemporalAgent(critic)
    p_agent = PrintAgent()
    target_q_agent = TemporalAgent(target_critic)
    goal_setter = GoalAgent(
        cfg.algorithm.n_envs, cfg.goal.goal_dim, cfg.goal.goal_range, cfg.goal.goal_type
    )
    eval_goal_setter = GoalAgent(
        cfg.algorithm.nb_evals,
        cfg.goal.goal_dim,
        cfg.goal.goal_range,
        cfg.goal.goal_type,
    )
    reward_calculator = RewardAgent(cfg.goal.reward_scale)
    tr_agent = Agents(goal_setter, train_env_agent, critic, explorer)
    ev_agent = Agents(eval_goal_setter, eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, q_agent, target_q_agent


# Configure the optimizer
def setup_optimizers(cfg, q_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = q_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values, action):
    """_summary_

    Args:
        cfg (_type_): _description_
        reward (torch.Tensor): A (T x B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (T x B) tensor containing 0 if the episode is completed at time $t$
        q_values (torch.Tensor): a (T x B x A) tensor containing Q values
        action (torch.LongTensor): a (T) long tensor containing the chosen action

    Returns:
        torch.Scalar: The DQN loss
    """

    # We compute the max of Q-values over all actions
    max_q = target_q_values[1].max(-1)[0].detach()

    # To get the max of Q(s_{t+1}, a), we take max_q[1:]
    # The same about must_bootstrap.
    target = reward[:-1] + cfg.algorithm.discount_factor * max_q * must_bootstrap.int()

    # To get Q(s,a), we use torch.gather along the 3rd dimension (the action)
    act = action[0].unsqueeze(-1)
    qvals = torch.gather(q_values[0], dim=1, index=act).squeeze()

    # Compute the temporal difference (use must_boostrap as to mask out finished episodes)
    td = target - qvals

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # print(critic_loss)
    return critic_loss


def run_dqn(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    shortest_timestep = 10e9

    # 2) Create the environment agent
    train_env_agent = AuroResetGoalEnvAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = AuroResetGoalEnvAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )

    # 3) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, target_q_agent = create_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, q_agent)
    nb_steps = 0
    tmp_steps = 0
    tmp_steps2 = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace, t=1, n_steps=cfg.algorithm.n_steps, stochastic=True
            )
        else:
            train_agent(
                train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True
            )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        rb.put(transition_workspace)

        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
        q_agent(rb_workspace, t=0, n_steps=2, choose_action=False)

        q_values, done, truncated, reward, action = rb_workspace[
            "q_values", "env/done", "env/truncated", "env/reward", "action"
        ]

        with torch.no_grad():
            target_q_agent(rb_workspace, t=0, n_steps=2, stochastic=True)

        target_q_values = rb_workspace["q_values"]
        # assert torch.equal(q_values, target_q_values), "values differ"

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        # Compute critic loss
        critic_loss = compute_critic_loss(
            cfg, reward, must_bootstrap, q_values, target_q_values, action
        )

        # Store the loss for tensorboard display
        logger.add_log("critic_loss", critic_loss, nb_steps)

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            q_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        if nb_steps - tmp_steps2 > cfg.algorithm.target_critic_update:
            tmp_steps2 = nb_steps
            target_q_agent.agent = copy.deepcopy(q_agent.agent)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace, t=0, stop_variable="env/done", choose_action=True
            )

            timestep = eval_workspace["env/timestep"][-1]
            mean = timestep.float().mean()
            logger.add_log("timestep", mean, nb_steps)
            reward_logger.add(nb_steps, mean)
            print(f"nb_steps: {nb_steps}, timesteps to reach goal: {mean}")
            if cfg.save_best and mean < shortest_timestep:
                shortest_timestep = mean
                directory = "./dqn_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "dqn_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                if cfg.plot_agents:
                    policy = eval_agent.agent.agents[1]
                    plot_policy(
                        policy,
                        eval_env_agent,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        shortest_timestep,
                        stochastic=False,
                    )
                    plot_critic(
                        policy,
                        eval_env_agent,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        shortest_timestep,
                    )


@hydra.main(config_path="./configs/", config_name="dqn_her_maze.yaml")
def main(cfg: DictConfig):
    # train_env_agent = NoAutoResetGymAgent(
    #     get_class(cfg.gym_env),
    #     get_arguments(cfg.gym_env),
    #     cfg.algorithm.n_envs,
    #     cfg.algorithm.seed,
    # )

    # eval_env_agent = NoAutoResetGymAgent(
    #     get_class(cfg.gym_env),
    #     get_arguments(cfg.gym_env),
    #     cfg.algorithm.nb_evals,
    #     cfg.algorithm.seed,
    # )

    # train_agent, eval_agent, q_agent, target_q_agent = create_dqn_agent(
    #     cfg, train_env_agent, eval_env_agent
    # )

    # train_workspace = Workspace()

    # train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True)

    # t = train_workspace
    # print(t)

    logdir = "./plot/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    reward_logger = RewardLogger(
        logdir + "dqn_rb_target.steps", logdir + "dqn_rb_target.rwd"
    )
    for seed in range(cfg.algorithm.nb_seeds):
        cfg.algorithm.seed = seed
        torch.manual_seed(cfg.algorithm.seed)
        run_dqn(cfg, reward_logger)
        if seed < cfg.algorithm.nb_seeds - 1:
            reward_logger.new_episode()
    reward_logger.save()
    plotter = Plotter(logdir + "dqn_rb_target.steps", logdir + "dqn_rb_target.rwd")
    plotter.plot_reward("qdn rb and target", cfg.gym_env.env_name)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
