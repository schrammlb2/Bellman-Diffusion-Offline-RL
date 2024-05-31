import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP, NormedMLP
from offlinerlkit.modules import Actor, Critic
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.buffer import NextActionBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import ReBRACPolicy

import pyrallis
from dataclasses import dataclass


"""
suggested hypers
alpha=2.5 for all D4RL-Gym tasks
"""


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--algo-name", type=str, default="rebrac")
#     parser.add_argument("--task", type=str, default="hopper-medium-v2")
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
#     # parser.add_argument("--hidden-dims", type=int, nargs='*', default=[512, 512])
#     # parser.add_argument("--actor-lr", type=float, default=3e-4)
#     # parser.add_argument("--critic-lr", type=float, default=3e-4)
#     parser.add_argument("--actor-lr", type=float, default=0.001)
#     parser.add_argument("--critic-lr", type=float, default=0.001)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--tau", type=float, default=0.005)
#     parser.add_argument("--exploration-noise", type=float, default=0.1)
#     parser.add_argument("--policy-noise", type=float, default=0.2)
#     parser.add_argument("--noise-clip", type=float, default=0.5)
#     parser.add_argument("--update-actor-freq", type=int, default=2)
#     # parser.add_argument("--actor-action-reg-weight", type=float, default=0.4)
#     # parser.add_argument("--critic-action-reg-weight", type=float, default=0.4)
#     parser.add_argument("--actor-action-reg-weight", type=float, default=0.01)
#     parser.add_argument("--critic-action-reg-weight", type=float, default=0.01)
#     parser.add_argument("--epoch", type=int, default=1000)
#     parser.add_argument("--step-per-epoch", type=int, default=1000)
#     parser.add_argument("--eval_episodes", type=int, default=10)
#     # parser.add_argument("--batch-size", type=int, default=256)
#     parser.add_argument("--batch-size", type=int, default=1024)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--no-q", action='store_true')

#     return parser.parse_args()

@dataclass
class Config:
    # wandb params
    project: str = "CORL"
    group: str = "rebrac"
    name: str = "rebrac"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    train_seed: int = 0
    eval_seed: int = 42
    relative_state_bc_coef: float = 1.0

    # def __post_init__(self):
    #     self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"

def get_next_actions(dataset):
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                            dataset["next_observations"][i]
                            ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        for key in dataset.keys():
            trajs[i]
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                        terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    
    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset


@pyrallis.wrap()
def train(config: Config):
    # create env and dataset
    task = config.dataset_name
    algo_name = config.name + "no_q"
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [config.hidden_dim]*config.critic_n_hiddens

    env = gym.make(task)
    dataset = qlearning_dataset(env)
    if 'antmaze' in task:
        dataset["rewards"] -= 1.0
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    max_action = env.action_space.high[0]
    

    # create buffer
    buffer = NextActionBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device=device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    # seed
    random.seed(config.train_seed)
    np.random.seed(config.train_seed)
    torch.manual_seed(config.train_seed)
    torch.cuda.manual_seed_all(config.train_seed)
    torch.backends.cudnn.deterministic = True
    env.seed(config.train_seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
    critic1_backbone = NormedMLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims)
    critic2_backbone = NormedMLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims)
    actor = Actor(actor_backbone, action_dim, max_action=max_action, device=device)

    critic1 = Critic(critic1_backbone, device)
    critic2 = Critic(critic2_backbone, device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.critic_learning_rate)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config.critic_learning_rate)

    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # create policy
    policy = ReBRACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=config.tau,
        gamma=config.gamma,
        max_action=max_action,
        exploration_noise=GaussianNoise(sigma=0.1),
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        update_actor_freq=config.policy_freq,        
        actor_action_reg_weight=config.actor_bc_coef, 
        critic_action_reg_weight=config.critic_bc_coef,
        scaler=scaler,
        no_q=True
    )

    # log
    # log_dirs = make_log_dirs(task, algo_name, config.train_seed, vars(args))
    log_dirs = make_log_dirs(task, algo_name, config.train_seed, vars(config))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(config))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=config.num_epochs,
        step_per_epoch=config.num_updates_on_epoch,
        batch_size=config.batch_size,
        eval_episodes=config.eval_episodes
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()