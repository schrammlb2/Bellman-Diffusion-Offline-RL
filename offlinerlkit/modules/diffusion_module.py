import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


class DiffusionModel(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        obs_dim: int, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = obs_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        obs: Union[np.ndarray, torch.Tensor],
        step: Union[np.ndarray, torch.Tensor], 
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        step = torch.as_tensor(step, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([x, obs, step, actions], dim=1)
        logits = self.backbone(obs)
        output = self.last(logits)
        return output



class UnconditionalDiffusionModel(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        output_dim: int, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, output_dim).to(device)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        step: Union[np.ndarray, torch.Tensor], 
    ) -> torch.Tensor:
        step = torch.as_tensor(step, device=self.device, dtype=torch.float32)
        x = torch.cat([x, step], dim=1)
        logits = self.backbone(x)
        output = self.last(logits)
        return output


class RewardDiffusionModel(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        obs_dim: int, 
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = obs_dim
        self.state_layer = nn.Linear(latent_dim, output_dim).to(device)
        self.reward_layer = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        obs: Union[np.ndarray, torch.Tensor],
        step: Union[np.ndarray, torch.Tensor], 
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        step = torch.as_tensor(step, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([x, obs, step, actions], dim=1)
        logits = self.backbone(obs)
        state_output = self.state_layer(logits)
        reward_output = self.reward_layer(logits)
        return state_output, reward_output
