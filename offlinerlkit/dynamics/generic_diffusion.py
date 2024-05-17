import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from copy import deepcopy

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler


class GenericDiffusionModel:
    def __init__(
        self, 
        diffusion_model, 
        diffusion_model_optim,
        gamma,
        tau,
        num_diffusion_iters
    ):
        self.var_dim = diffusion_model.last.out_features

        self.gamma = gamma
        self._tau = tau

        self.diffusion_model = diffusion_model
        self.diffusion_model_old = deepcopy(diffusion_model)
        self.diffusion_model_optim = diffusion_model_optim

        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type="sample",
        )
    
    def map_i(self, t):
        return -1 + 2*t/self.num_diffusion_iters

    def eval(self):
    	self.diffusion_model.eval()

    def train(self):
    	self.diffusion_model.train()

    def _sync_weight(self):
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
    
    # def predict(self, model, noise, obs, a):
    def _predict(self, model, condition):
        map_i = lambda t: -1 + 2*t/self.num_diffusion_iters

        assert len(condition.shape) == 2
        batch_size = condition.shape[0]
        x = torch.randn((batch_size, self.var_dim), device=condition.device)
        x_list = []
        diff_steps = torch.ones((batch_size,1))#, device=self.device)
        for i in range(self.num_diffusion_iters-1, 0, -1):
            epsilon_prediction = model(
                x=x, condition=condition,
                step=self.map_i(diff_steps*i))
            x = self.diff_step(
                model_output=epsilon_prediction,
                sample=x, 
                timestep=i-1
            )
            x_list.append(x)
        
        return x#, torch.stack(x_list, 0)


    def get_eta(self, timesteps):
        timesteps = timesteps.cpu()
        alpha = self.q_noise_scheduler.alphas[timesteps].to(self.device)
        alpha_prod = self.q_noise_scheduler.alphas_cumprod[timesteps].to(self.device)
        beta = 1-alpha
        eta = (beta**2)*alpha*(1-alpha_prod)
        eta_norm = eta/eta.mean()
        return eta_norm


    # def sample(self, obss, sample_actions):
    #     with torch.no_grad():
    #         a = sample_actions(obss)
    #         batch_size = obss.shape[0]
    #         diff_steps = torch.randint(0, 
    #             self.num_diffusion_iters, 
    #             (batch_size, 1)
    #         ).long().to(obss.device)
    #         obss_noise = torch.randn(obss.shape, device=obss.device)
    #         noised_obss = self.add_noise(obss, obss_noise, diff_steps) 
    #         future_obss = self._predict(
    #                 model=self.diffusion_model_old,
    #                 noise=obss_noise, obs=obss, a=a
    #         ).detach()
    #     samples = {
    #         "obss": future_obss.cpu()
    #     }
    #     return samples

    def sample(self, obss, sample_actions):
        with torch.no_grad():
            a = sample_actions(obss)
            batch_size = obss.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(obss, obss_noise, diff_steps)
            condition = torch.cat([obss, a], dim=-1) 
            future_obss = self._predict(
                    model=self.diffusion_model_old,
                    condition=condition
            ).detach()
        samples = {
            "obss": future_obss.cpu()
        }
        return samples



class ObsDiffusionModel(GenericDiffusionModel):    
    def __init__(
        self, 
        diffusion_model, 
        diffusion_model_optim,
        gamma,
        tau,
        num_diffusion_iters,
        # var_dim,
        # cond_dim, 
    ):
        super().__init__(diffusion_model, 
            diffusion_model_optim,
            gamma,
            tau,
            num_diffusion_iters
        )

    def sample(self, obss, sample_actions):
        with torch.no_grad():
            a = sample_actions(obss)
            batch_size = obss.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(obss, obss_noise, diff_steps)
            condition = torch.cat([obss, a], dim=-1) 
            future_obss = self._predict(
                    model=self.diffusion_model_old,
                    condition=condition
            ).detach()
        samples = {
            "obss": future_obss.cpu()
        }
        return samples


    def update_diffusion(self, batch, sample_actions):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        # valid_next_actions = batch["valid_next_actions"]
        # next_actions = batch["next_actions"]
        pred_next_actions = sample_actions(obss)

        diffusion_list = []
        with torch.no_grad():
            batch_size = rewards.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 

            condition = torch.cat([next_obss, pred_next_actions], dim=-1) 
            pred_obss = self._predict(
                    model=self.diffusion_model_old,
                    condition=condition
            ).detach()
            noised_next_obss = self.add_noise(next_obss, obss_noise, diff_steps) 
            noised_pred_obss = self.add_noise(pred_obss, obss_noise, diff_steps)            
            next_target = self.diffusion_model_old(
                x=noised_pred_obss, 
                condition=torch.cat([next_obss, pred_next_actions], dim=-1),
                step=self.map_i(diff_steps),).detach()

        current_prediction = self.diffusion_model(
            x=noised_next_obss, 
            condition=torch.cat([next_obss, pred_next_actions], dim=-1),
            step=self.map_i(diff_steps)
        )
        # next_prediction = current_prediction
        next_prediction = self.diffusion_model(
            x=noised_pred_obss , 
            condition=torch.cat([next_obss, pred_next_actions], dim=-1),
            step=self.map_i(diff_steps)
        )

        # diffusion_loss = (
        #     (1-self._gamma)*(current_prediction - next_obss)**2 + 
        #     (self._gamma  )*(next_prediction - next_target)**2
        # ).mean()
        diffusion_loss = (
            (1-self.gamma)*(current_prediction - next_obss)**2 + 
            (  self.gamma)*(next_prediction  - next_target)**2
        ).mean()
        diffusion_list.append(diffusion_loss)
        diffusion_loss = torch.stack(diffusion_list).mean()


        self.diffusion_model_optim.zero_grad()
        diffusion_loss.backward()
        self.diffusion_model_optim.step()

        return pred_obss, diffusion_loss



class GammaDiffusionModel(GenericDiffusionModel):    
    def __init__(
        self, 
        diffusion_model, 
        diffusion_model_optim,
        gamma,
        tau,
        num_diffusion_iters,
        # var_dim,
        # cond_dim, 
    ):
        super().__init__(diffusion_model, 
            diffusion_model_optim,
            gamma,
            tau,
            num_diffusion_iters
        )

    def sample(self, obss, sample_actions):
        with torch.no_grad():
            a = sample_actions(obss)
            batch_size = obss.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(obss, obss_noise, diff_steps)
            condition = torch.cat([obss, a], dim=-1) 
            future_obss = self._predict(
                    model=self.diffusion_model_old,
                    condition=condition
            ).detach()
        samples = {
            "obss": future_obss.cpu()
        }
        return samples


    def update_diffusion(self, batch, sample_actions):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        # valid_next_actions = batch["valid_next_actions"]
        # next_actions = batch["next_actions"]
        pred_next_actions = sample_actions(obss)

        diffusion_list = []
        with torch.no_grad():
            batch_size = rewards.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 

            condition = torch.cat([next_obss, pred_next_actions], dim=-1) 
            pred_obss = self._predict(
                    model=self.diffusion_model_old,
                    condition=condition
            ).detach()
            noised_next_obss = self.add_noise(next_obss, obss_noise, diff_steps) 
            noised_pred_obss = self.add_noise(pred_obss, obss_noise, diff_steps)            
            next_target = self.diffusion_model_old(
                x=noised_pred_obss, 
                condition=torch.cat([next_obss, pred_next_actions], dim=-1),
                step=self.map_i(diff_steps),).detach()

        current_prediction = self.diffusion_model(
            x=noised_next_obss, 
            condition=torch.cat([next_obss, pred_next_actions], dim=-1),
            step=self.map_i(diff_steps)
        )
        # next_prediction = current_prediction
        next_prediction = self.diffusion_model(
            x=noised_pred_obss , 
            condition=torch.cat([next_obss, pred_next_actions], dim=-1),
            step=self.map_i(diff_steps)
        )

        diffusion_loss = (
            (1-self.gamma)*(current_prediction - next_obss)**2 + 
            (  self.gamma)*(next_prediction  - next_target)**2
        ).mean()
        diffusion_list.append(diffusion_loss)
        diffusion_loss = torch.stack(diffusion_list).mean()


        self.diffusion_model_optim.zero_grad()
        diffusion_loss.backward()
        self.diffusion_model_optim.step()

        return pred_obss, diffusion_loss