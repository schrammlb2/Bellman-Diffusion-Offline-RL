import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

class VecNorm(nn.Module):
    def __init__(self, p=2, dim=-1) -> None:
        super().__init__()

    def forward(self, input):
        return F.normalize(input)
