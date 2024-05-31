import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import ReBRACPolicy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler


from copy import deepcopy

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler


class ReBRACSOMPolicy(ReBRACPolicy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        diffusion_model: nn.Module,
        data_diffusion_model: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        diffusion_model_optim: torch.optim.Optimizer,
        data_diffusion_model_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        actor_action_reg_weight: int = 0.1, 
        critic_action_reg_weight: int = 0.1, 
        relative_state_reg_weight = 1,
        scaler: StandardScaler = None, 
        no_q = False,
        num_diffusion_iters: int = 10,
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
            update_actor_freq=update_actor_freq,
            actor_action_reg_weight=actor_action_reg_weight,
            critic_action_reg_weight=critic_action_reg_weight,
            scaler=scaler,
            no_q = no_q
        )

        self.diffusion_model = diffusion_model
        self.diffusion_model_old = deepcopy(diffusion_model)
        self.diffusion_model_optim = diffusion_model_optim
        self.data_diffusion_model = data_diffusion_model
        self.data_diffusion_model_old = deepcopy(data_diffusion_model)
        self.data_diffusion_model_optim = data_diffusion_model_optim

        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type="sample",
        )
        self.relative_state_reg_weight = relative_state_reg_weight

    
    def train(self) -> None:
        super().train()
        self.diffusion_model.train()
        self.data_diffusion_model.train()


    def eval(self) -> None:
        super().eval()
        self.diffusion_model.eval()
        self.data_diffusion_model.eval()

    def _sync_weight(self) -> None:
        super()._sync_weight()
        for o, n in zip(self.data_diffusion_model_old.parameters(), self.data_diffusion_model.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.diffusion_model_old.parameters(), self.diffusion_model.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    

    def diff_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        **kwargs,
    ):
        # predict_epsilon = deprecate("predict_epsilon", "0.12.0", message, take_from=kwargs)
        # predict_epsilon = True
        # if predict_epsilon is not None:
        #     new_config = dict(self.noise_scheduler.config)
        #     new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
        #     # self.noise_scheduler._internal_dict = FrozenDict(new_config)

        t = timestep

        if model_output.shape[1] == sample.shape[1] * 2 and self.noise_scheduler.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[t - 1] if t > 0 else self.noise_scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.noise_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.noise_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.noise_scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.noise_scheduler.betas[t]) / beta_prod_t
        current_sample_coeff = self.noise_scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            if device.type == "mps":
                # randn does not work reproducibly on mps
                variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                variance_noise = variance_noise.to(device)
            else:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                )
            if self.noise_scheduler.variance_type == "fixed_small_log":
                variance = self.noise_scheduler._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            else:
                variance = (self.noise_scheduler._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample #+ variance

        if not return_dict:
            return (pred_prev_sample,)

        # return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
        return pred_prev_sample


    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.noise_scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def predict(self, model, noise, obs, a):
        map_i = lambda t: -1 + 2*t/self.num_diffusion_iters

        x = noise
        x_list = []
        diff_steps = torch.ones((obs.shape[0],1))#, device=self.device)
        for i in range(self.num_diffusion_iters-1, 0, -1):
            epsilon_prediction = model(
                x=x, obs=obs, 
                step=map_i(diff_steps*i),
                actions=a)
            x = self.diff_step(
                model_output=epsilon_prediction,
                sample=x, 
                timestep=i-1
            )
            x_list.append(x)
        
        return x#, torch.stack(x_list, 0)
 
    def get_eta(self, timesteps):
        t = timesteps.cpu()
        alpha = self.noise_scheduler.alphas[t].to(timesteps.device)
        alpha_prod = self.noise_scheduler.alphas_cumprod[t].to(timesteps.device)
        beta = 1-alpha
        eta = (beta**2)*alpha*(1-alpha_prod)
        eta_norm = eta/eta.mean()
        return eta_norm

    def map_i(self, t):
        return -1 + 2*t/self.num_diffusion_iters

    def update_diffusion(self, batch, pred_next_actions):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        with torch.no_grad():
            batch_size = rewards.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 

        diffusion_list = []
        with torch.no_grad():
            next_diff_obss = self.predict(
                    model=self.diffusion_model_old,
                    noise=obss_noise, obs=next_obss, a=pred_next_actions
            ).detach()
            noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 
            next_noised_obss = self.add_noise(next_diff_obss, obss_noise, diff_steps)            
            next_target = self.diffusion_model_old(
                x=next_noised_obss, obs=next_obss, 
                step=self.map_i(diff_steps),
                actions=pred_next_actions).detach()

        current_prediction = self.diffusion_model(
            x=noised_obss, obs=obss, 
            step=self.map_i(diff_steps),
            actions=actions)
        next_prediction = self.diffusion_model(
            x=next_noised_obss, obs=obss, 
            step=self.map_i(diff_steps),
            actions=actions)

        diffusion_loss = (
            (1-self._gamma)*(current_prediction - next_obss)**2 + 
            (self._gamma  )*(next_prediction -  next_target)**2
        ).mean()
        diffusion_list.append(diffusion_loss)
        diffusion_loss = torch.stack(diffusion_list).mean()


        self.diffusion_model_optim.zero_grad()
        diffusion_loss.backward()
        self.diffusion_model_optim.step()


        data_prediction = self.data_diffusion_model(
            x=noised_obss, step=self.map_i(diff_steps)
        )
        data_diffusion_loss = ((data_prediction - obss)**2).mean()
        self.data_diffusion_model_optim.zero_grad()
        data_diffusion_loss.backward()
        self.data_diffusion_model_optim.step()

        return next_diff_obss, diffusion_loss

    def state_bc(self, obss, a=None):
        if type(a) == type(None):
            a = self.actor(obss)
        batch_size = obss.shape[0]
        diff_steps = torch.randint(0, 
            self.num_diffusion_iters, 
            (batch_size, 1)
        ).long().to(obss.device)
        eta = self.get_eta(diff_steps)

        obss_noise = torch.randn(obss.shape, device=obss.device)
        pred_obss = self.predict(
                model=self.diffusion_model_old,
                noise=obss_noise, obs=obss, a=a
        ).detach()
        noised_pred_obss  = self.add_noise(pred_obss, obss_noise, diff_steps)

        obs_prediction = self.diffusion_model_old(
            x=noised_pred_obss, obs=obss, 
            step=self.map_i(diff_steps),
            actions=a
        )
        data_obs_prediction = self.data_diffusion_model(
            x=noised_pred_obss, step=self.map_i(diff_steps)
        )
        state_bc_penalty = self.relative_state_reg_weight*self.actor_action_reg_weight*(
            eta*((data_obs_prediction - obs_prediction)**2).sum(-1)
        ).mean()
        return state_bc_penalty

    def rand_like(self, x):
        # Generate tensor with same shape as x, and same mean and std for the last axis
        device = x.device
        rands = torch.rand(x.shape, device=device)
        means = x.mean(0)
        return (x - means)/x.std(0)
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        valid_next_actions = batch["valid_next_actions"]
        batch_size = rewards.shape[0]
        
        # update critic
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)

        if self.use_q:
            q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
            with torch.no_grad():
                next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
                bc_penalty = (
                    self.critic_action_reg_weight*valid_next_actions*(
                        (next_actions - batch["next_actions"])**2).sum(-1)
                )
                target_q = rewards + self._gamma * (1 - terminals) * ( next_q - bc_penalty )
            
            critic1_loss = ((q1 - target_q).pow(2)).mean()
            critic2_loss = ((q2 - target_q).pow(2)).mean()

            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            self.critic1_optim.step()

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()
        else: 
            critic1_loss = torch.tensor(0)
            critic2_loss = torch.tensor(0)

        sampled_states, diffusion_loss = self.update_diffusion(batch, next_actions)


        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            bc_penalty = self.actor_action_reg_weight*((a - actions).pow(2).sum(-1)).mean()
            
            # Generate random samples with same mean and variance as real samples
            # And train the policy to go back to the real samples
            rand_obss = self.rand_like(obss)
            rand_a = self.actor(rand_obss)

            #Mix the batches
            mixed_obss = torch.cat([obss, rand_obss], dim=0)
            mixed_a = torch.cat([a, rand_a], dim=0)
            state_bc_penalty = self.state_bc(mixed_obss, a=mixed_a)

            if self.use_q:
                q = self.critic1(obss, a)
                lmbda = 1 / q.abs().mean().detach()
                # actor_loss = -lmbda * q.mean() + self.actor_action_reg_weight*((a - actions).pow(2)).mean()
                actor_loss = -lmbda * q.mean() + bc_penalty + state_bc_penalty
            else: 
                actor_loss = bc_penalty + state_bc_penalty
                #ReBRAC sums over action dim
                #Try using ReBRAC coefficients and with the sum, see if this lets you match ReBRAC perf. 

            # actor_loss = bc_penalty

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._last_bc_penalty = bc_penalty.item()
            self._last_state_bc_penalty = state_bc_penalty.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/bc"   : self._last_bc_penalty,
            "loss/state_bc"   : self._last_state_bc_penalty,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }