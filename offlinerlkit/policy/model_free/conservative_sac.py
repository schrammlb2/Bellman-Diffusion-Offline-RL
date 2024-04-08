import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.policy import SACPolicy


class ConservativeSACPolicy(SACPolicy):
    """
    Conservative Q-Learning <Ref: https://arxiv.org/abs/2006.04779>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
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
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        self._policy_noise = 0.2
        self._alpha = 2.5
        self.alpha_optim = None

        self.actor_old = deepcopy(actor)
        self.actor_old.eval()

    def dist_actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)

        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return dist, squashed_action, log_prob


    def _sync_weight(self) -> None:
        super()._sync_weight()
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions = self.actor_old(next_obss).rsample()[0]
            # next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) #- self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        dist, a, log_probs = self.dist_actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        kl = dist.log_prob(actions)
        q = torch.min(q1a, q2a)
        # actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()

        lmbda = self._alpha / q.abs().mean().detach()
        # lmbda = 0 / q.abs().mean().detach()
        # actor_loss = - lmbda*torch.min(q1a, q2a).mean() + kl.mean()
        q = self.critic1(obss, a)
        lmbda = self._alpha / q.abs().mean().detach()
        actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }


        return result

    # def learn(self, batch: Dict) -> Dict[str, float]:
    #     obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
    #         batch["next_observations"], batch["rewards"], batch["terminals"]

    #     # update critic
    #     q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
    #     with torch.no_grad():
    #         next_actions, next_log_probs = self.actforward(next_obss)
    #         next_q = torch.min(
    #             self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
    #         ) #- self._alpha * next_log_probs
    #         target_q = rewards + self._gamma * (1 - terminals) * next_q

    #     critic1_loss = ((q1 - target_q).pow(2)).mean()
    #     self.critic1_optim.zero_grad()
    #     critic1_loss.backward()
    #     self.critic1_optim.step()

    #     critic2_loss = ((q2 - target_q).pow(2)).mean()
    #     self.critic2_optim.zero_grad()
    #     critic2_loss.backward()
    #     self.critic2_optim.step()

    #     # update actor
    #     dist, a, log_probs = self.dist_actforward(obss)
    #     q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

    #     kl = dist.log_prob(actions)
    #     # actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
    #     actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * kl.mean()

    #     if self._is_auto_alpha:
    #         # log_probs = log_probs.detach() + self._target_entropy
    #         log_probs = kl.detach() + self._target_entropy
    #         alpha_loss = -(self._log_alpha * log_probs).mean()

    #     self._sync_weight()

    #     result = {
    #         "loss/actor": actor_loss.item(),
    #         "loss/critic1": critic1_loss.item(),
    #         "loss/critic2": critic2_loss.item(),
    #     }


    #     return result


    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            lmbda = self._alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }