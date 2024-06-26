import argparse
import random

import gym
import d4rl

import numpy as np
import torch
import ray
from ray import tune


from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor, ActorProb, Critic, TanhDiagGaussian, DiffusionModel
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import SOMRegularizedSACPolicy
from offlinerlkit.policy import RenyiRegSACPolicy



# from torch.backends.cudnn import benchmark as cudnn_benchmark

"""
suggested hypers
alpha=2.5 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="renyi")
    parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[512, 512])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    # parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--action_reg_weight", type=float, default=0.1)
    parser.add_argument("--state_reg_weight", type=float, default=10.)
    parser.add_argument("--num_diffusion_iters", type=int, default=10)
    # parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def trainable(config):
    import d4rl
    # set config
    global args
    args_for_exp = vars(args)
    for k, v in config.items():
        args_for_exp[k] = v
    args_for_exp = argparse.Namespace(**args_for_exp)
    print(args_for_exp.task)
    # create env and dataset
    env = gym.make(args_for_exp.task)
    dataset = qlearning_dataset(env)
    if 'antmaze' in args_for_exp.task:
        dataset["rewards"] -= 1.0
    args_for_exp.obs_shape = env.observation_space.shape
    args_for_exp.action_dim = np.prod(env.action_space.shape)
    args_for_exp.max_action = env.action_space.high[0]

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args_for_exp.obs_shape,
        obs_dtype=np.float32,
        action_dim=args_for_exp.action_dim,
        action_dtype=np.float32,
        device=args_for_exp.device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    # seed
    random.seed(args_for_exp.seed)
    np.random.seed(args_for_exp.seed)
    torch.manual_seed(args_for_exp.seed)
    torch.cuda.manual_seed_all(args_for_exp.seed)
    # torch.backends.cudnn.deterministic = True
    env.seed(args_for_exp.seed)

    # create policy model
    # h=256
    # h=512
    actor_backbone = MLP(input_dim=np.prod(args_for_exp.obs_shape), hidden_dims=args_for_exp.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args_for_exp.obs_shape)+args_for_exp.action_dim, hidden_dims=args_for_exp.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args_for_exp.obs_shape)+args_for_exp.action_dim, hidden_dims=args_for_exp.hidden_dims)
    diffusion_backbone = MLP(input_dim=2*np.prod(args_for_exp.obs_shape)+args_for_exp.action_dim + 1, hidden_dims=args_for_exp.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args_for_exp.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args_for_exp.max_action
    )
    actor = ActorProb(actor_backbone, dist, args_for_exp.device)

    critic1 = Critic(critic1_backbone, args_for_exp.device)
    critic2 = Critic(critic2_backbone, args_for_exp.device)
    diffusion_model = DiffusionModel(diffusion_backbone, obs_dim=np.prod(args_for_exp.obs_shape), device=args_for_exp.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args_for_exp.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args_for_exp.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args_for_exp.critic_lr)
    diffusion_optim = torch.optim.Adam(diffusion_model.parameters(), lr=args_for_exp.critic_lr)


    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # create policy
    policy = RenyiRegSACPolicy(
        actor,
        critic1,
        critic2,
        diffusion_model,
        actor_optim,
        critic1_optim,
        critic2_optim,
        diffusion_optim,
        tau=args_for_exp.tau,
        gamma=args_for_exp.gamma,
        max_action=args_for_exp.max_action,
        exploration_noise=GaussianNoise(sigma=args_for_exp.exploration_noise),
        policy_noise=args_for_exp.policy_noise,
        noise_clip=args_for_exp.noise_clip,
        update_actor_freq=args_for_exp.update_actor_freq,
        action_reg_weight=args_for_exp.action_reg_weight,
        state_reg_weight=args_for_exp.state_reg_weight,
        num_diffusion_iters=args_for_exp.num_diffusion_iters,
        scaler=scaler
    )

    # log
    log_dirs = make_log_dirs(args_for_exp.task, args_for_exp.algo_name, args_for_exp.seed, vars(args_for_exp))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args_for_exp))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args_for_exp.epoch,
        step_per_epoch=args_for_exp.step_per_epoch,
        batch_size=args_for_exp.batch_size,
        eval_episodes=args_for_exp.eval_episodes, 
        # report=True,
        report=False,
    )

    # train
    result = policy_trainer.train()
    tune.report(**result)
    return result


if __name__ == "__main__":
    import ray
    from ray import train, tune
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    import os
    ray.init()

    # args = get_args()

    # config = {
    #     "action_reg_weight": tune.qloguniform(0.001, 1, 0.001),
    #     "state_reg_weight": tune.qloguniform(0.01, 50, 0.01),
    #     "num_diffusion_iters": tune.lograndint(3, 100),
    # }

    # algo = OptunaSearch()
    # # algo.restore_from_dir(
    # #   os.path.join("~/my_results", "my-experiment-1")
    # # )
    # # algo = ConcurrencyLimiter(algo, max_concurrent=4)
    # # algo = ConcurrencyLimiter(algo)
    # num_samples = 100
    # trainable_with_resources = tune.with_resources(trainable, {"gpu": 0.25})
    # # args.device = "cpu"
    # # trainable_with_resources = tune.with_resources(trainable, {"cpu": 1})
    # algo = ConcurrencyLimiter(algo, max_concurrent=4)
    # tuner = tune.Tuner(
    #     trainable_with_resources,
    #     tune_config=tune.TuneConfig(
    #         metric="reward",
    #         mode="max",
    #         search_alg=algo,
    #         num_samples=num_samples,
    #     ),
    #     run_config=ray.train.RunConfig(
    #         name="my-experiment-1",
    #         storage_path="~/my_results",
    #     ),
    #     param_space=config
    # )
    # results = tuner.fit()
    # # trainable({})


    args = get_args()
    config = {
        # "action_reg_weight": tune.grid_search([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]),
        "action_reg_weight": tune.grid_search([0.001, 0.01, 0.1, 1]),
        "state_reg_weight": tune.grid_search([0.01, 0.1, 1, 10]),
        "f_divergence_exponent": tune.grid_search([0.1, 1, 10])
    }

    analysis = tune.run(
        trainable,
        name="tune_renyi",
        config=config,
        resources_per_trial={
            "gpu": 0.5
        }
    )