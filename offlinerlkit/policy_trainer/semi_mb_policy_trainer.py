import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer, StateBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

from ray import train

# model-free policy trainer
class SemiMBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        report: bool = False
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.report = report

        self._rollout_freq = 10
        self._rollout_batch_size = batch_size
        self._real_ratio = 1/2

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):
            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar: 
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.buffer.sample(self._rollout_batch_size)["observations"]
                    rollout_transitions = self.policy.sample(init_obss)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    # self.logger.log(
                    #     "num rollout transitions: {}".\
                    #         format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    # )
                    # for _key, _value in rollout_info.items():
                    #     self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size//2)# * self._real_ratio)
                fake_sample_size = int(self._batch_size//2)#
                real_batch = self.buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}

                self.policy.train()
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

            if self.report:
                # train.report({"reward": ep_reward_mean})
                train.report({"reward": norm_ep_rew_mean})

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
