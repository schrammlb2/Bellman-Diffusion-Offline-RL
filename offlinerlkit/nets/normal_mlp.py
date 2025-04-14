import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets.VecNorm import VecNorm


# class ExtremelyNormalMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: Union[List[int], Tuple[int]],
#         activation: nn.Module = nn.Tanh,
#         layer_num: int = 1
#     ) -> None:
#         super().__init__()
#         self.layer_num = layer_num
#         model = []
#         model += [spectral_norm(nn.Linear(input_dim, hidden_dim)), ScaleLayer((input_dim/hidden_dim)**0.5), activation()]
#         model += [nn.LayerNorm(hidden_dim), spectral_norm(nn.Linear(hidden_dim, hidden_dim)),  activation()]
#         model += [nn.LayerNorm(hidden_dim), spectral_norm(nn.Linear(hidden_dim, input_dim)), ScaleLayer((hidden_dim/input_dim)**0.5)]  
#         # model += [activation(), nn.LayerNorm(input_dim), spectral_norm(nn.Linear(input_dim, input_dim))]  

#         self.model = nn.Sequential(*model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)

#     def norm_weights(self):
#         pass

# class SpecNormLinear(nn.Module):

def spec_norm_weights(linear):
	sv = torch.linalg.matrix_norm(linear.weight, ord=2)
	with torch.no_grad():
		weights = linear.weight.clone().detach()
		out_dim = linear.weight.shape[0]
		in_dim = linear.weight.shape[1]
		linear.weight = torch.nn.Parameter(weights*((out_dim/in_dim)**0.5)/sv)
	return linear

# def SpecNormLinear(in_dim, out_dim):
# 	linear = nn.Linear(in_dim, out_dim)
# 	# eigs = torch.linalg.eigvals(linear.weight)
# 	# sv = torch.max(eigs)
# 	# sv = torch.linalg.matrix_norm(linear.weight, ord=2)
# 	# with torch.no_grad():
# 	# 	weights = linear.weight.clone().detach()
# 	# 	linear.weight = torch.nn.Parameter(weights*((out_dim/in_dim)**0.5)/sv)
# 	linear = spec_norm_weights(linear)
# 	# linear = nn.utils.spectral_norm(linear, n_power_iterations=5)
# 	# linear = nn.utils.weight_norm(linear)
# 	return linear


class SpecNormLinear(nn.Module):
	def __init__(self, in_dim, out_dim, renorm = True, renorm_time=1e3):
		super().__init__()
		linear = nn.Linear(in_dim, out_dim)
		self.linear = spec_norm_weights(linear)
		self.renorm = renorm
		self.renorm_time = renorm_time
		self.count = 1

	def forward(self, x):
		self.count += 1
		if self.renorm and self.count % self.renorm_time == 0:
			self.linear = spec_norm_weights(self.linear)
		return self.linear(x)



class ExtremelyNormalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.Tanh,
        layer_num: int = 1
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        model = []
        model += [SpecNormLinear(input_dim, hidden_dims[0])]
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):        	
        	# model += [activation(), nn.LayerNorm(in_dim, elementwise_affine=False, bias=False), SpecNormLinear(in_dim, out_dim)]
        	model += [
            	activation(), 
            	nn.LayerNorm(in_dim, elementwise_affine=False, bias=False),
            	SpecNormLinear(in_dim, out_dim), 
            	# nn.LayerNorm(out_dim),
            ]
        	# model += [activation(), SpecNormLinear(in_dim, out_dim)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            # model += [activation(), nn.LayerNorm(in_dim, elementwise_affine=False, bias=False), nn.Linear(hidden_dims[-1], output_dim)]
            model += [
            	activation(), 
            	nn.LayerNorm(in_dim, elementwise_affine=False, bias=False), 
            	nn.Linear(hidden_dims[-1], output_dim), 
            	# SpecNormLinear(hidden_dims[-1], output_dim), 
            ]
            # model += [activation(), nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def norm_weights(self):
        pass

class ExtremelyNormalBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[List[int], Tuple[int]],
        activation: nn.Module = nn.Tanh,
        layer_num: int = 1
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        scale=True
        model = []
        model += [nn.LayerNorm(input_dim, elementwise_affine=False, bias=False), SpecNormLinear((input_dim, hidden_dim)), activation()]
        model += [nn.LayerNorm(hidden_dim, elementwise_affine=False, bias=False), SpecNormLinear((input_dim, hidden_dim)),  activation()]
        model += [nn.LayerNorm(hidden_dim, elementwise_affine=False, bias=False), SpecNormLinear((input_dim, hidden_dim))]  
        # model += [activation(), nn.LayerNorm(input_dim), spectral_norm(nn.Linear(input_dim, input_dim))]  
        if scale:     
        	model += [ScaleLayer(layer_num**(-0.5))]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def norm_weights(self):
        pass

class ExtremelyNormalResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[List[int], Tuple[int]],
        activation: nn.Module = nn.Tanh,
        layer_num: int = 1
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.model = ExtremelyNormalBlock(input_dim, hidden_dim, activation, layer_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x*(self.layer_num/(self.layer_num + 1))**0.5 + self.model(x)
        return x + self.model(x)

class ExtremelyNormalResnet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        residual_hidden_dim: int = 256,
        block_hidden_dim: int = 512,
        num_blocks: int=3,
        activation: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        model = [SpecNormLinear(input_dim, residual_hidden_dim), activation()]
        for l in range(num_blocks): 
            model += [ExtremelyNormalResidual(residual_hidden_dim, block_hidden_dim, activation)]

        model += [nn.LayerNorm(wide_hidden_dim), activation(), nn.LayerNorm(wide_hidden_dim, elementwise_affine=False, bias=False), 
            nn.Linear(wide_hidden_dim, output_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# class ExtremelyNormalNetwork():
	# Kinds of normalization and regularization
	# > Max update initialization
	# > SpectralNorm on the weights
	# > Spectral regularization (trying to maximize minimum spectral value to prevent rank collapse)
	# > sqrt(fan_in/fan_out) scaling to get feature learning
	# > Layer Norm on post-activation layer to ensure normal statistics
	# > > Post, because if we start with normalized vec, and use matrix with spec norm, have upper bound on norm of pre-activation layer
	# > Deep residual architecture
	# > 1/sqrt(L) scaling on the output of the blocks 