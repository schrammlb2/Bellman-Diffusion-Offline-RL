import numpy as np
import torch
import torch.nn as nn
from torch_utils import persistence

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

# @persistence.persistent_class
class MPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        # if self.training:
        #     with torch.no_grad():
        #         self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

class MPSiLU(nn.Module): 
    def __init__(self): 
        super(MPSiLU, self).__init__() 
  
    def forward(self, x, beta=1): 
        return torch.nn.functional.silu(x) / 0.596


# class MPMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dims: Union[List[int], Tuple[int]],
#         output_dim: Optional[int] = None,
#         activation: nn.Module = MPSiLU,
#         dropout_rate: Optional[float] = None
#     ) -> None:
#         super().__init__()
#         hidden_dims = [input_dim] + list(hidden_dims)
#         model = []
#         # activation = nn.Module.SELU
#         # activation = MPSiLU
#         for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
#             model += [MPLinear(in_dim, out_dim), activation()]

#         self.output_dim = hidden_dims[-1]
#         if output_dim is not None:
#             model += [nn.Linear(hidden_dims[-1], output_dim)]
#             self.output_dim = output_dim
#         self.model = nn.Sequential(*model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)

#     def norm_weights(self):
#         for layer in self.model:
#             if hasattr(layer, "norm_weights"): 
#                 layer.norm_weights()

# @persistence.persistent_class
class MPMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        # activation = nn.Module.SELU
        activation = nn.SiLU
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [MPLinear(in_dim, out_dim), activation()]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [MPLinear(hidden_dims[-1], output_dim), activation()]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def norm_weights(self):
        for layer in self.model:
            if hasattr(layer, "norm_weights"): 
                layer.norm_weights()


# @persistence.persistent_class
class MPDenseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim = None,
        activation: nn.Module = MPSiLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        # h = h[0]
        model = []
        self.activation = activation
        # activation = nn.Module.SELU
        # activation = MPSiLU
        cumulative = input_dim
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            # model += [MPLinear(cumulative, out_dim)]
            model += [MPMLP(input_dim=cumulative, hidden_dims=[out_dim], output_dim=out_dim)]
            cumulative += out_dim

        self.cumulative = cumulative
        if output_dim is not None:
            self.final_layer = nn.Linear(cumulative, output_dim)
            self.output_dim = output_dim
            self.last_layer = True
        else: 
            self.final_layer = None
            self.output_dim = cumulative
            self.last_layer = False

        

        self.model = nn.ModuleList(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            out = layer(x)
            x = torch.cat([x, out], dim=-1)
        if self.last_layer:
            return self.final_layer(x)
        else: 
            return x

    def norm_weights(self):
        for layer in self.model:
            if hasattr(layer, "norm_weights"): 
                layer.norm_weights()