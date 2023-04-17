import sys
import os

import copy

import torch
import gym
import hydra

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymb import NoAutoResetGymAgent

from bbrl_examples.models.exploration_agents import EGreedyActionSelector
from bbrl_examples.models.loggers import Logger, RewardLogger
from bbrl_examples.models.plotters import Plotter

from GoalConditioned import GoalAgent, RewardAgent, HerFinal
from critics import GCQAgent


def make_gym_env(env_name):
    return gym.make(env_name)


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


def compute_reward(achieved_goal, desired_goal):
    # achieved_goal (batch, 4)
    # desired_goal (batch, 2)
    achieved_goal = achieved_goal[:, [0, 2]]
    dist = torch.abs(achieved_goal - desired_goal)
    epsilon = torch.tensor([1, 0.1])
    reward = torch.all(dist < epsilon, dim=1).float()
    return reward


def compute_terminated(achieved_goal, desired_goal):
    # achieved_goal (batch, 4)
    # desired_goal (batch, 2)
    achieved_goal = achieved_goal[:, [0, 2]]
    dist = torch.abs(achieved_goal - desired_goal)

    epsilon = torch.tensor([1, 0.1])
    terminated = torch.all(dist < epsilon, dim=1).bool()
    return terminated


def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    goal_dim = cfg.goal.goal_dim
    critic = GCQAgent(
        obs_size, cfg.algorithm.architecture.hidden_size, act_size, goal_dim
    )
    target_critic = copy.deepcopy(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon_init)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
    goal_agent = GoalAgent(
        cfg.algorithm.n_envs, goal_dim, cfg.goal.goal_range, cfg.goal.goal_type
    )
    reward_agent = RewardAgent(
        compute_reward=compute_reward, compute_terminated=compute_terminated
    )
    tr_agent = Agents(goal_agent, train_env_agent, critic, explorer, reward_agent)
    goal_agent = GoalAgent(
        cfg.algorithm.nb_evals, goal_dim, cfg.goal.goal_range, cfg.goal.goal_type
    )
    ev_agent = Agents(goal_agent, eval_env_agent, critic, reward_agent)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    her_agent = TemporalAgent(Agents(HerFinal(), reward_agent))
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, q_agent, target_q_agent, her_agent


def run_dqn(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent = NoAutoResetGymAgent(
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
    train_agent, eval_agent, q_agent, target_q_agent, her_agent = create_dqn_agent(
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
    nb_measures = 0

    # 7) Training loop
    while nb_measures < cfg.algorithm.nb_measures:
        train_workspace = Workspace()  # Used for training
        train_agent(
            train_workspace,
            t=0,
            stop_variable="env/done",
            stochastic=True,
        )

        her_workspace = Workspace()

        her_agent(
            her_workspace,
            t=0,
            n_steps=train_workspace.time_size(),
            trajectory=train_workspace,
        )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        her_transition_workspace = her_workspace.get_transitions()
        rb.put(transition_workspace)
        rb.put(her_transition_workspace)

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
            nb_measures += 1
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                choose_action=True,
                render=True,
            )
            print(eval_workspace["env/desired_goal"][-1])
            print(eval_workspace["env/env_obs"][-1])
            time_step = eval_workspace["env/timestep"][-1].float()
            mean = time_step.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, timesetp: {mean}")
            reward_logger.add(nb_steps, mean)


def main_loop(cfg):
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


@hydra.main(config_path="./configs/", config_name="dqn_gc_cartpole.yaml")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    main_loop(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
