import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets.VecNorm import VecNorm

class MLP(nn.Module):
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
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    def norm_weights(self):
        pass



class NormedMLP(nn.Module):
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
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation(), nn.LayerNorm(out_dim)]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def norm_weights(self):
        pass

class VecNormMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.Tanh,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            # model += [nn.Linear(in_dim, out_dim), activation(), nn.LayerNorm(out_dim), VecNorm()]
            # model += [nn.Linear(in_dim, out_dim), activation(), VecNorm()]
            model += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        # model += [nn.LayerNorm(out_dim), VecNorm()]
        model += [VecNorm()]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    def norm_weights(self):
        pass

class DenseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim = None,
        activation: nn.Module = nn.ReLU,
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
        for _ in range(3):
            # model += [MPLinear(cumulative, out_dim)]
            out_dim = hidden_dims[-1]
            model += [NormedMLP(input_dim=cumulative, hidden_dims=hidden_dims)]
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
