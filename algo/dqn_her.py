import copy
import os
import sys
import hydra
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import gym

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace,
# or until a given condition is reached
from bbrl.agents import Agents, RemoteAgent, TemporalAgent

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

from warppers import MazeMDPContinuousHerWrapper


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + 1] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, choose_action=True, **kwargs):
        # observation Tensor of shape (1, 2), squeeze(0) -> (2,)
        obs = self.get(("env/observation", t)).squeeze(0)
        # desired_goal Tensor of shape (1, )
        desired_goal = self.get(("env/desired_goal", t))
        obs = torch.cat([obs, desired_goal], dim=0)
        # obs (3,) -> (1, 3)
        obs = obs.unsqueeze(0)
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


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self, epoch, critic_loss, entropy_loss, actor_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("entropy_loss", entropy_loss, epoch)
        self.add_log("actor_loss", actor_loss, epoch)

    def log_reward_losses(self, rewards, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)


def make_gym_env(env_name, env_kwargs):
    return MazeMDPContinuousHerWrapper(gym.make(env_name, kwargs=env_kwargs))


def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = DiscreteQAgent(obs_size, cfg.algorithm.architecture.hidden_size, act_size)
    target_critic = copy.deepcopy(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon_init)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
    tr_agent = Agents(train_env_agent, critic, explorer)
    ev_agent = Agents(eval_env_agent, critic)

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
    # Compute temporal difference
    max_q = target_q_values[1].max(-1)[0].detach()
    target = reward[:-1] + cfg.algorithm.discount_factor * max_q * must_bootstrap.int()
    act = action[0].unsqueeze(-1)
    qvals = torch.gather(q_values[0], dim=1, index=act).squeeze()
    td = target - qvals
    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss


def her_sample(cfg, transitions: Workspace):
    # TODO : this is not correct yet
    her_transitions = Workspace()
    i = 0
    for t in range(transitions.time_size()):
        for k in range(cfg.HER_K):
            future = np.random.randint(t, transitions.time_size())
            desired_goal = transitions.get("env/desired_goal", t + future)
            state = transitions.get("env/observation", t)
            action = transitions.get("action", t)
            next_state = transitions.get("env/observation", t + 1)
            achieved_goal = transitions.get("env/achieved_goal", t)
            done = achieved_goal == desired_goal
            reward = torch.FloatTensor([1.0]) if done else torch.FloatTensor([0.0])
            her_transitions.set("env/observation", i, state)
            her_transitions.set("env/observation", i + 1, next_state)
            her_transitions.set("env/desired_goal", i, desired_goal)
            her_transitions.set("env/achieved_goal", i, achieved_goal)
            her_transitions.set("env/desired_goal", i + 1, desired_goal)
            her_transitions.set(
                "env/achieved_goal", i + 1, transitions.get("env/achieved_goal", t + 1)
            )
            her_transitions.set("action", i, action)
            her_transitions.set("action", i + 1, transitions.get("action", t + 1))
            her_transitions.set("reward", i, torch.Tensor([0.0]))
            her_transitions.set("reward", i + 1, reward)
            her_transitions.set("done", i, torch.Tensor([False]))
            her_transitions.set("done", i + 1, torch.Tensor([done]))
            i += 2
    return her_transitions


def run_dqn(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
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
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"epoch: {epoch}, reward: {mean}")
            reward_logger.add(nb_steps, mean)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
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
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        policy,
                        eval_env_agent,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )


@hydra.main(config_path="./configs/", config_name="dqn_her_maze.yaml")
def main(cfg: DictConfig):
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )

    train_agent, eval_agent, q_agent, target_q_agent = create_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )

    train_workspace = Workspace()

    train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True)

    print(train_workspace.get_transitions().get("env/done", 0))
    print(train_workspace.get_transitions().get("env/done", 1))


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
