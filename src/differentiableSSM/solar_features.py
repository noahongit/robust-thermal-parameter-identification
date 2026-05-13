import math
from typing import Tuple
import torch
import torch.nn as nn


class SolarShares(nn.Module):
    def __init__(self, init_shares=(0.25, 0.25, 0.25, 0.25), are_learnable: bool = True):
        super().__init__()
        if init_shares is None:
            logits = torch.zeros(4, dtype=torch.float32)
        else:
            s = torch.as_tensor(init_shares, dtype=torch.float32)
            s = s.clamp_min(1e-6)
            s = s / s.sum()
            logits = torch.log(s)
            logits = logits - logits.mean()
        if are_learnable:
            self.logits = nn.Parameter(logits)
        else:
            self.logits = nn.Parameter(logits).requires_grad_(False)

    def weights(self) -> torch.Tensor:
        """Return directional weights (North, East, South, West) that sum to 1."""
        return torch.softmax(self.logits, dim=0)  # [4]

    def forward(self, irradiance_per_direction: torch.Tensor) -> torch.Tensor:
        """
        Args:
            irradiance_per_direction:
                [N,4]  or [B,N,4] tensor, order [N, E, S, W], units W/m^2
        Returns:
            effective irradiance:
                [N]   if input [N,4]
                [B,N] if input [B,N,4]
        """
        w = self.weights()  # [4]

        if irradiance_per_direction.dim() == 2:  # [N,4]
            return (irradiance_per_direction * w.view(1, 4)).sum(dim=-1)

        elif irradiance_per_direction.dim() == 3:  # [B,N,4]
            return (irradiance_per_direction * w.view(1, 1, 4)).sum(dim=-1)

        else:
            raise ValueError("irradiance_per_direction must be [N,4] or [B,N,4]")

    def get_solar_shares(self) -> dict[str, float]:
        w = self.weights()
        return {
            "North": round(w[0].item(), 2),
            "East": round(w[1].item(), 2),
            "South": round(w[2].item(), 2),
            "West": round(w[3].item(), 2),
        }
