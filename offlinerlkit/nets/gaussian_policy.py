

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GaussianPolicy(nn.Module):
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
            self.mean = nn.Linear(hidden_dims[-1], output_dim)
            self.log_std = nn.Linear(hidden_dims[-1], output_dim)
            self.output_dim = output_dim
        self.trunk = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.trunk(x)
        mean, log_std = self.mean(out), self.log_std(out)-1        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor) -> torch.Tensor:        
        out = self.trunk(x)
        mean, log_std = self.mean(out), self.log_std(out)-1        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        pi_distribution = Normal(mu, torch.exp(log_std))
        return torch.tanh()