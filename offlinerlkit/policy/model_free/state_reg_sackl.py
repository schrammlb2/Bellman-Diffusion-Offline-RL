import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinerlkit.policy import SACPolicy


class StateRegSACKLPolicy(SACPolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_reg_weight: float=0.1,
        tau: float = 0.005,
        gamma: float  = 0.99,
        # state_reg_weight: float = 10.,
        max_action: float = 1.0,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        self._max_action = max_action
        self.action_reg_weight = action_reg_weight
        # self.state_reg_weight = state_reg_weight

    def kl(self, dist, actions):
        bound=0.999 #For numerical stability
        atanh_actions = torch.atanh(
            torch.clamp(actions, 
            min=-self._max_action*bound, max=self._max_action*bound)
        )
        atanh_a = dist.mode()[1]
        # tanh_a = torch.tanh(atanh_a)
        target_sigma = 0.2
        sigma = dist.scale#*target_sigma
        # sigma = torch.tensor(0.2)
        # kl = (((atanh_a - atanh_actions).pow(2)*sigma**(-2)) + 2*torch.log(sigma)).sum(-1)
        kl = (
            ((target_sigma**2 + (atanh_a - atanh_actions).pow(2))*sigma**(-2)) 
            + 2*torch.log(sigma/target_sigma)
            - 1
        ).sum(-1)
        return kl


    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        valid_next_actions = batch["valid_next_actions"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_dist = self.actor(obss)
            next_squashed_actions, next_raw_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_squashed_actions, next_raw_actions)
            next_q = torch.min(
                self.critic1_old(next_obss, next_squashed_actions), self.critic2_old(next_obss, next_squashed_actions)
            ) 
            next_entropy_bonus = - self._alpha * next_log_probs
            next_kl_penalty = self.action_reg_weight*valid_next_actions*self.kl(next_dist, batch["next_actions"])
            # target_q = rewards + self._gamma * (1 - terminals) * ( next_q - next_kl_penalty + next_entropy_bonus)
            target_q = rewards + self._gamma * (1 - terminals) * ( next_q )# + next_entropy_bonus)

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        dist = self.actor(obss)
        squashed_a, a = dist.rsample()
        log_probs = dist.log_prob(squashed_a, a)
        q1a, q2a = self.critic1(obss, squashed_a), self.critic2(obss, squashed_a)

        kl_penalty = self.kl(dist, actions)
        entropy_bonus = - self._alpha * log_probs
        
        q = torch.min(q1a, q2a).mean()
        
        lmbda = 1 / torch.min(q1a.abs(), q2a.abs()).mean().detach()

        actor_loss = - (
            lmbda*q 
            + 0*entropy_bonus.mean() 
            - self.action_reg_weight*kl_penalty.mean()
        )
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/q_mean": actor_loss.item(),
            # "loss/kl_penalty": kl_penalty.mean().item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "metric/sigma": dist.scale.mean().item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

