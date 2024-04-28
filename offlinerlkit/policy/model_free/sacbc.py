import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler


class SACBCPolicy(TD3Policy):
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
        alpha: float = 2.5,
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

        self._alpha = alpha
        self.scaler = scaler
    
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
        # with torch.no_grad():
        #     action = self.actor(obs).mode()[1].cpu().numpy()
        # if not deterministic:
        #     action = action + self.exploration_noise(action.shape)
        #     action = np.clip(action, -self._max_action, self._max_action)
        if deterministic:
            with torch.no_grad():
                action = self.actor(obs).mode()[0].cpu().numpy()
        if not deterministic:
            with torch.no_grad():
                action = self.actor(obs).rsample()[0].cpu().numpy()
                # mode = self.actor(obs).mode()[1]
                # noise = (torch.randn_like(mode) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
                # action = torch.tanh((mode + noise))
        return action
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            # next_actions = (self.actor_old(next_obss).mode()[1] + noise).clamp(-self._max_action, self._max_action)
            # next_actions = (self.actor_old(next_obss).mode()[0] + noise).clamp(-self._max_action, self._max_action)
            # dist = self.actor_old(next_obss)
            # if np.random.rand() < 0.0001:
            #     import ipdb
            #     ipdb.set_trace()
            # next_actions = self.actor_old(next_obss).rsample()[0]
            # next_actions = batch["next_actions"]
            # next_actions = torch.tanh((self.actor_old(next_obss).mode()[1] + noise))
            # next_actions = self.actor_old(next_obss).mode()[0]
            next_actions = self.actor_old(next_obss).rsample()[0]
            # next_actions = (self.actor_old(next_obss).rsample()[1] + noise).clamp(-self._max_action, self._max_action)
            # next_actions = (self.actor_old(next_obss).rsample()[1]).clamp(-self._max_action, self._max_action)
            # next_actions = self.actor_old(next_obss).rsample()[0]
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()


        dist = self.actor(obss)
        bound = 0.999
        atanh_actions = torch.atanh(
            torch.clamp(actions, 
            min=-self._max_action*bound, max=self._max_action*bound)
        )
        a = dist.mode()[1].detach()
        # a = actions
        tanh_a = torch.tanh(a)
        sigma = dist.scale
        kl = (
            ((a - atanh_actions).pow(2)*sigma**(-2)).mean() + 
            2*torch.log(sigma).mean() - torch.log(torch.tensor(2*torch.pi))
        )*self._policy_noise**2
        # q1_prev, q2_prev = self.critic1(obss, a), self.critic2(obss, a)
        # critic1_loss += (1/self._alpha*kl.detach()*q1_prev).mean()
        # critic2_loss += (1/self._alpha*kl.detach()*q2_prev).mean()
        critic1_loss += (1/self._alpha*kl.detach()*q1).mean()
        critic2_loss += (1/self._alpha*kl.detach()*q2).mean()
            # critic1_loss += (1/self._alpha*kl.detach()*q1_prev).mean()
            # critic2_loss += (1/self._alpha*kl.detach()*q2_prev).mean()



        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:

            # a = self.actor(obss).mode()[1]
            dist = self.actor(obss)
            # a = dist.mode()[0]
            a = dist.rsample()[1]
            tanh_a = torch.tanh(a)
            sigma = dist.scale
            q = self.critic1(obss, tanh_a)
            lmbda = self._alpha / q.abs().mean().detach()
            # log_pol_noise = torch.log(self._policy_noise)
            c = self._policy_noise
            bound = 0.999
            atanh_actions = torch.atanh(
                torch.clamp(actions, 
                min=-self._max_action*bound, max=self._max_action*bound)
            )
            kl = ((a - atanh_actions).pow(2)*sigma**(-2)).mean() + 2*torch.log(sigma).mean()
            actor_loss = -lmbda * q.mean() + kl*self._policy_noise**2
            # actor_loss = -lmbda * q.mean() + log_probs.mean()  

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