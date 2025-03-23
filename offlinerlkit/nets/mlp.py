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
            # model += [nn.Linear(in_dim, out_dim), activation(), nn.LayerNorm(out_dim)]
            model += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), activation()]
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



class ScaleLayer(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, inpt):
        return scale*inpt

class AccordianBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[List[int], Tuple[int]],
        activation: nn.Module = nn.Tanh,
        layer_num: int = 1
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        scale=False
        if scale: 
            model = []
            # model += [nn.LayerNorm(input_dim), ScaleLayer(layer_num**(-0.5)), nn.Linear(input_dim, hidden_dim),  activation()]
            # model += [nn.LayerNorm(hidden_dim), ScaleLayer(layer_num**(-0.5)), nn.Linear(hidden_dim, hidden_dim),  activation()]
            model += [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),  activation()]
            model += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),  activation()]
            model += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim),  activation()]  
            model += [nn.LayerNorm(input_dim), nn.Linear(input_dim, input_dim), ScaleLayer(layer_num**(-0.5))]          
        else: 
            model = []
            model += [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),  activation()]
            model += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),  activation()]
            model += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim),  activation()]
            model += [nn.LayerNorm(input_dim), nn.Linear(input_dim, input_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)

    def norm_weights(self):
        pass

class AccordionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        wide_hidden_dim: int = 2048,
        narrow_hidden_dim: int = 256,
        num_blocks: int=3,
        activation: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        model = [nn.Linear(input_dim, wide_hidden_dim), activation()]
        for l in range(num_blocks): 
            model += [AccordianBlock(wide_hidden_dim, narrow_hidden_dim, activation)]

        model += [nn.LayerNorm(wide_hidden_dim), activation(), nn.LayerNorm(wide_hidden_dim), 
            nn.Linear(wide_hidden_dim, output_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
