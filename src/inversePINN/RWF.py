import torch
import torch.nn as nn


class FactorizedLinear(nn.Module):
    """
    Random-Weight Factorization from Wang et al. 2023, Algorithm 2
    Implements  W = diag(exp(s)) · V  with  s ~ N(μ, σ).
    """

    def __init__(self, in_dimension: int, out_dimension: int, mu: float = 0.5, sigma: float = 0.1, bias: bool = True):
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.randn(out_dimension) * sigma + mu
        )  # (a) Initialize each scale factor as s(l) ∼ N (μ, σI)
        self.V = nn.Parameter(torch.empty(out_dimension, in_dimension))
        nn.init.xavier_normal_(self.V, gain=nn.init.calculate_gain("tanh"))
        self.bias = nn.Parameter(torch.zeros(out_dimension)) if bias else None

    def forward(self, x: torch.Tensor):
        W = (
            torch.exp(self.log_scale).unsqueeze(1) * self.V
        )  # (b) Construct the factorized weight matrices as W(l) = diag(exp(s(l))) · V(l)
        return torch.nn.functional.linear(x, W, self.bias)
