import math

from matplotlib import pyplot as plt
import torch
from torch import nn

torch.manual_seed(128)


class FourierTimeEncoder(nn.Module):
    """
    Encode the first column (tau) with Fourier features.
    Supports 'random' (Gaussian) or 'harmonic' (Physics-Informed) frequencies.

    Input  :  shape (N, input_dim_NN)
    Output :  shape (N, input_dim_NN + 2*n_freq) = [ sin(2π f_i τ) | cos(2π f_i τ) | feature_NN ]
    """

    def __init__(self, n_frequencies: int = 5, sigma: float = 4.0, encoding_method: str = "harmonic"):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.encoding_method = encoding_method

        if self.encoding_method == "harmonic":
            self.update_frequencies(duration_days=14.0)
        else:
            # B ~ N(0, Iσ²)  shape (m,1) because we encode only tau
            B = torch.randn(n_frequencies, 1, dtype=torch.float32) * sigma
            self.register_buffer("B", B)

    def update_frequencies(self, duration_days: float) -> None:
        """
        Recalculates B matrix so that frequencies match the 24h cycle
        for the specific duration of the dataset.
        """
        if self.encoding_method != "harmonic":
            return

        base_freq = duration_days / 2.0

        # Create harmonics: 1x, 2x, 3x... (24h, 12h, 8h...)
        indices = torch.arange(1, self.n_frequencies + 1, dtype=torch.float32)

        # Shape (n_freq, 1)
        B = (base_freq * indices).unsqueeze(1)

        # Updates the buffer
        self.register_buffer("B", B)
        # print(f"Updated Fourier Features for {duration_days:.1f} days.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N,1)
        """
        tau = x[:, 0:1]  # (N,1)

        projection = 2 * math.pi * tau @ self.B.T  # (N,1)x(m,1).T > (N,m)

        sin_, cos_ = torch.sin(projection), torch.cos(projection)  # (N,m)

        parts = [sin_, cos_, tau]
        return torch.cat(parts, dim=-1)
