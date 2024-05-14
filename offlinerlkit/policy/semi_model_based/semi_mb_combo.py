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

class SemiMBCOMBOPolicy(CQLPolicy):
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
        rho_s: str = "mix"
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
        for _ in range(10):
            print("Test")

        self._uniform_rollout = uniform_rollout
        self._rho_s = rho_s


        self.diffusion_model = diffusion_model
        self.diffusion_model_old = deepcopy(diffusion_model)
        self.diffusion_model_optim = diffusion_model_optim

        self.num_diffusion_iters = 10
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type="sample",
        )


    
    def map_i(self, t):
        return -1 + 2*t/self.num_diffusion_iters

    def train(self) -> None:
        super().train()
        self.diffusion_model.train()

    def eval(self) -> None:
        super().eval()
        self.diffusion_model.eval()

    def _sync_weight(self) -> None:
        super()._sync_weight()
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
                step=self.map_i(diff_steps*i),
                actions=a)
            x = self.diff_step(
                model_output=epsilon_prediction,
                sample=x, 
                timestep=i-1
            )
            x_list.append(x)
        
        return x#, torch.stack(x_list, 0)

    def sample(self, obss):
        with torch.no_grad():
            a, log_probs = self.actforward(obss)
            batch_size = obss.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(obss, obss_noise, diff_steps) 
            future_obss = self.predict(
                    model=self.diffusion_model_old,
                    noise=obss_noise, obs=obss, a=a
            ).detach()

        samples = {
            "obss": future_obss.cpu()
        }
        return samples


    # def update_diffusion(self, batch):
    #     obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
    #         batch["next_observations"], batch["rewards"], batch["terminals"]
    #     # valid_next_actions = batch["valid_next_actions"]
    #     # next_actions = batch["next_actions"]
    #     pred_next_actions, log_probs = self.actforward(obss)

    #     diffusion_list = []
    #     with torch.no_grad():
    #         batch_size = rewards.shape[0]
    #         diff_steps = torch.randint(0, 
    #             self.num_diffusion_iters, 
    #             (batch_size, 1)
    #         ).long().to(obss.device)
    #         obss_noise = torch.randn(obss.shape, device=obss.device)
    #         noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 


    #         next_diff_obss = self.predict(
    #                 model=self.diffusion_model_old,
    #                 noise=obss_noise, obs=next_obss, a=pred_next_actions
    #         ).detach()
    #         noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 
    #         next_noised_obss = self.add_noise(next_diff_obss, obss_noise, diff_steps)            
    #         next_target = self.diffusion_model_old(
    #             x=next_noised_obss, obs=next_obss, 
    #             step=self.map_i(diff_steps),
    #             actions=pred_next_actions).detach()

    #     current_prediction = self.diffusion_model(
    #         x=noised_obss, obs=obss, 
    #         step=self.map_i(diff_steps),
    #         actions=actions)
    #     next_prediction = self.diffusion_model(
    #         x=next_noised_obss, obs=obss, 
    #         step=self.map_i(diff_steps),
    #         actions=actions)

    #     diffusion_loss = (
    #         (1-self._gamma)*(current_prediction - next_obss)**2 + 
    #         (self._gamma  )*(next_prediction - next_target)**2
    #     ).mean()
    #     diffusion_list.append(diffusion_loss)
    #     diffusion_loss = torch.stack(diffusion_list).mean()


    #     self.diffusion_model_optim.zero_grad()
    #     diffusion_loss.backward()
    #     self.diffusion_model_optim.step()

    #     return next_diff_obss, diffusion_loss'

    
    def update_diffusion(self, batch):
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        # valid_next_actions = batch["valid_next_actions"]
        # next_actions = batch["next_actions"]
        pred_next_actions, log_probs = self.actforward(obss)

        diffusion_list = []
        with torch.no_grad():
            batch_size = rewards.shape[0]
            diff_steps = torch.randint(0, 
                self.num_diffusion_iters, 
                (batch_size, 1)
            ).long().to(obss.device)
            obss_noise = torch.randn(obss.shape, device=obss.device)
            noised_obss = self.add_noise(next_obss, obss_noise, diff_steps) 


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
            (self._gamma  )*(next_prediction - next_target)**2
        ).mean()
        diffusion_list.append(diffusion_loss)
        diffusion_loss = torch.stack(diffusion_list).mean()


        self.diffusion_model_optim.zero_grad()
        diffusion_loss.backward()
        self.diffusion_model_optim.step()

        return next_diff_obss, diffusion_loss

    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in fake_batch.keys()}

        # obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
        #     mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        # batch_size = obss.shape[0]
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
        for obss in [real_batch['observations']]:#, fake_batch['observations']]:
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
        cat_q1 = torch.cat([obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, random_value2], 1)
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