import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.policy import CQLPolicy
from copy import deepcopy

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from offlinerlkit.dynamics import ObsDiffusionModel

class MBStoredSemiMBCOMBO2Policy(CQLPolicy):
    """
    Conservative Offline Model-Based Policy Optimization <Ref: https://arxiv.org/abs/2102.08363>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        diffusion_model: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        diffusion_model_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        uniform_rollout: bool = False,
        rho_s: str = "mix", 
        rollout_length = 5,
        use_rollout_length = False
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            cql_weight=cql_weight,
            temperature=temperature,
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            lagrange_threshold=lagrange_threshold,
            cql_alpha_lr=cql_alpha_lr,
            num_repeart_actions=num_repeart_actions
        )

        self._uniform_rollout = uniform_rollout
        self._rho_s = rho_s

        self.rollout_length = rollout_length

        if use_rollout_length:
            diffusion_gamma = (self.rollout_length - 1)/self.rollout_length
        else: 
            diffusion_gamma = gamma

        self.diffusion_model = ObsDiffusionModel(
            diffusion_model, 
            diffusion_model_optim, 
            diffusion_gamma,
            tau,
            num_diffusion_iters=10
        )



    def train(self) -> None:
        super().train()
        self.diffusion_model.train()

    def eval(self) -> None:
        super().eval()
        self.diffusion_model.eval()

    def _sync_weight(self) -> None:
        super()._sync_weight()
        self.diffusion_model._sync_weight()


    def sample(self, obss):
        return self.diffusion_model.sample(obss, lambda x: self.actforward(x)[0])

    def update_diffusion(self, batch):
        return self.diffusion_model.update_diffusion(batch, lambda x: self.actforward(x)[0])

    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in fake_batch.keys()}

        obss = mix_batch["observations"]

        real_obss, real_actions, real_next_obss, real_rewards, real_terminals = (
            real_batch["observations"], real_batch["actions"], \
            real_batch["next_observations"], real_batch["rewards"], real_batch["terminals"]
        )
        batch_size = real_obss.shape[0]


        sampled_states, diffusion_loss = self.update_diffusion(real_batch)
        
        # update actor
        a, log_probs = self.actforward(real_obss)
        q1a, q2a = self.critic1(real_obss, a), self.critic2(real_obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_real_next_obss = real_next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, real_next_obss.shape[-1])
                tmp_real_next_actions, _ = self.actforward(tmp_real_next_obss)
                tmp_real_next_q1 = self.critic1_old(tmp_real_next_obss, tmp_real_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_real_next_q2 = self.critic2_old(tmp_real_next_obss, tmp_real_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                real_next_q = torch.min(tmp_real_next_q1, tmp_real_next_q2)
        else:
            with torch.no_grad():
                real_next_actions, next_log_probs = self.actforward(real_next_obss)
                real_next_q = torch.min(
                    self.critic1_old(real_next_obss, real_next_actions),
                    self.critic2_old(real_next_obss, real_next_actions)
                )
                if not self._deterministic_backup:
                    real_next_q -= self._alpha * real_next_log_probs

        target_q = real_rewards + self._gamma * (1 - real_terminals) * real_next_q
        q1, q2 = self.critic1(real_obss, real_actions), self.critic2(real_obss, real_actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        if self._rho_s == "model":
            obss, actions, next_obss = fake_batch["observations"], \
                fake_batch["actions"], fake_batch["next_observations"]
            
        batch_size = len(real_obss)
        vlist1, vlist2 = [], []
        for obss in [real_batch['observations'], fake_batch['observations']]:
        # for obss in [real_batch['observations'], sampled_states]:
            random_actions = torch.FloatTensor(
                batch_size * self._num_repeat_actions, real_actions.shape[-1]
            ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
            # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
            tmp_obss = obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])
            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
            random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)
            vlist1.append(obs_pi_value1)
            vlist2.append(obs_pi_value2)
            vlist1.append(random_value1)
            vlist2.append(random_value2)


        tmp_obss = real_batch["observations"].unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, real_batch["observations"].shape[-1])
        tmp_next_obss = real_batch["next_observations"].unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, real_batch["next_observations"].shape[-1])

        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        vlist1.append(next_obs_pi_value1)
        vlist2.append(next_obs_pi_value2)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        # cat_q1 = torch.cat([obs_pi_value1, random_value1], 1)
        # cat_q2 = torch.cat([obs_pi_value2, random_value2], 1)        
        cat_q1 = torch.cat(vlist1, 1)
        cat_q2 = torch.cat(vlist2, 1)
        # Samples from the original dataset
        q1, q2 = self.critic1(real_obss, real_actions), self.critic2(real_obss, real_actions)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result