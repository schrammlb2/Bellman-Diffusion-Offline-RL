import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler


class SACKL2Policy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        actor_action_reg_weight: int = 0.4, 
        critic_action_reg_weight: int = 0.4, 
        scaler: StandardScaler = None
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
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq
        )

        self.scaler = scaler
        self.actor_action_reg_weight = actor_action_reg_weight
        self.critic_action_reg_weight = critic_action_reg_weight
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        with torch.no_grad():
            dist = self.actor(obs)
        # if deterministic: 
        #     return dist.mode()[0].cpu().numpy()
        # else:
        #     return dist.rsample()[0].cpu().numpy()
        action = dist.mode()[0].cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action

    def log_prob(self, dist, actions):
        bound=0.999 #For numerical stability
        atanh_actions = torch.atanh(
            torch.clamp(actions, 
            min=-self._max_action*bound, max=self._max_action*bound)
        )
        target_sigma = 0.2
        log_prob = target_sigma**(2)*(
            (((atanh_a - atanh_actions).pow(2))*sigma**(-2)) 
            + 2*torch.log(sigma/target_sigma) - 1
        ).sum(-1)


    def kl(self, dist, actions):
        eps = 1e-3 #For numerical stability
        atanh_actions = torch.atanh(
            torch.clamp(actions, 
            min=-self._max_action*(1-eps), max=self._max_action*(1-eps))
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
        kl = (
            ((sigma**2 + (atanh_a - atanh_actions).pow(2))*target_sigma**(-2)) 
            - 2*torch.log(sigma/target_sigma)
            - 1
        ).sum(-1)
        tanh_correction = - torch.log(self._max_action*(1 - actions.pow(2)) + eps).sum(-1)
        # tanh_correction = -torch.log(self._max_action*(1 - dist.mode()[0].pow(2)) + eps).sum(-1)
        tanh_correction += torch.log(self._max_action*(1 - dist.mode()[0].pow(2)) + eps).sum(-1)

        # kl = target_sigma**(2)*dist.log_prob(actions)
        # return (dist.mode()[0] - actions).pow(2).sum(-1)
        # return target_sigma**(2)*(kl + tanh_correction)
        return target_sigma**(2)*(kl)
        # return (dist.mode()[0] - actions).pow(2).sum(-1) + (dist.scale - 0.2).pow(2).sum(-1)
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        valid_next_actions = batch["valid_next_actions"]
        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            # next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            # next_actions = (self.actor_old(next_obss).mode()[0] + noise).clamp(-self._max_action, self._max_action)
            # next_actions = self.actor_old(next_obss).mode()[0]
            next_dist = self.actor_old(next_obss)
            # next_actions = next_dist.rsample()[0]
            next_actions = (self.actor_old(next_obss).mode()[0] + noise).clamp(-self._max_action, self._max_action)
            # next_actions = next_dist.mode()[0]
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            # bc_penalty = (
            #     self.critic_action_reg_weight*valid_next_actions*(
            #         (next_actions - batch["next_actions"])**2).sum(-1)
            # )
            bc_penalty = self.critic_action_reg_weight*valid_next_actions*self.kl(next_dist, batch["next_actions"])
            target_q = rewards + self._gamma * (1 - terminals) * ( next_q - bc_penalty )
        
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
            # a = self.actor(obss)
            dist = self.actor(obss)
            a = dist.mode()[0]
            # a = self.actor(obss).rsample()[0]
            q = self.critic1(obss, a)
            lmbda = 1 / q.abs().mean().detach()
            # actor_loss = -lmbda * q.mean() + self.actor_action_reg_weight*((a - actions).pow(2)).mean()
            # bc_penalty = self.actor_action_reg_weight*((a - actions).pow(2).sum(-1)).mean()
            bc_penalty = self.actor_action_reg_weight*self.kl(dist, actions).mean()
            # bc_penalty = self.actor_action_reg_weight*((a - actions).pow(2).sum(-1)).mean()
            actor_loss = -lmbda * q.mean() + bc_penalty
                #ReBRAC sums over action dim
                #Try using ReBRAC coefficients and with the sum, see if this lets you match ReBRAC perf. 
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._last_bc_penalty = bc_penalty.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/bc"   : self._last_bc_penalty,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }