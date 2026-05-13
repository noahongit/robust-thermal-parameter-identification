import math
from typing import Tuple
import torch
import torch.nn as nn

from fourierFeatures import FourierTimeEncoder
from RWF import FactorizedLinear


class DNN(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        n_hidden_layers: int,
        n_hidden_neurons: int,
        use_RWF: bool = True,
        use_RFF: bool = True,
        n_frequencies: int = 5,
        encoding_method: str = "harmonic",
        learn_init_mass_temp: bool = False,
    ):
        super().__init__()
        self.learn_init_mass_temp = learn_init_mass_temp
        if self.learn_init_mass_temp:
            self.init_theta_mass = nn.Parameter(torch.tensor(0.0))

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        if use_RFF:
            self.encoder = FourierTimeEncoder(n_frequencies=n_frequencies, encoding_method=encoding_method)
            n_additional_features = 2 * n_frequencies
            self.input_dimension += n_additional_features  # input dims + 2* n_freqs

        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = n_hidden_layers

        Dense = FactorizedLinear if use_RWF else nn.Linear

        self.input_layer = Dense(self.input_dimension, self.n_hidden_neurons)
        self.hidden_layers = nn.ModuleList(
            [Dense(self.n_hidden_neurons, self.n_hidden_neurons) for _ in range(self.n_hidden_layers - 1)]
        )
        self.output_layer = Dense(self.n_hidden_neurons, self.output_dimension)

        self.activation = nn.Tanh()
        self.init_xavier()

    def update_encoding_duration(self, tau: torch.Tensor, duration_days: float, verbose: bool = False):
        """Helper to pass dataset duration down to the encoder"""
        if hasattr(self, "encoder"):
            self.encoder.update_frequencies(duration_days)
            if verbose:
                self.encoder.plot_transformation(tau=tau)

    def forward(self, x: torch.Tensor):
        t = x[:, 0:1]  # shape: (N, 1)
        if hasattr(self, "encoder"):
            x = self.encoder(x)  # (N, input_dim) > (N, input dims + 2* n_freqs)
        x = self.activation(self.input_layer(x))
        for idx, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
        f = self.output_layer(x)

        # hard-enforce both initial conditions
        initial_state = torch.stack([self.init_theta_in, self.init_theta_mass])  # (2,) #type: ignore
        gate = 1.0 - torch.exp(-(t + 1.0) / 0.001)
        T_pred = initial_state + gate * f  # (N,2)
        return T_pred

    def init_xavier(self):
        def init_weights(m):
            if isinstance(m, nn.Linear) and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain("tanh")
                # torch.nn.init.xavier_uniform_(m.weight, gain=g)
                torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    @torch.no_grad()
    def _set_scales(self, theta_in: torch.Tensor) -> None:
        self.init_theta_in = theta_in[0].detach().clone()
        if not self.learn_init_mass_temp:
            self.init_theta_mass = theta_in[0].detach().clone()


class SolarShares(nn.Module):
    def __init__(self, init_shares=(0.25, 0.25, 0.25, 0.25)):
        super().__init__()
        if init_shares is None:
            logits = torch.zeros(4, dtype=torch.float32)
        else:
            s = torch.as_tensor(init_shares, dtype=torch.float32)
            s = s.clamp_min(1e-6)
            s = s / s.sum()
            logits = torch.log(s)
            logits = logits - logits.mean()
        self.logits = nn.Parameter(logits)

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
