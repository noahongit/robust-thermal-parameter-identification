import torch
import torch.nn as nn


class DimlessRC(nn.Module):
    def __init__(
        self,
        Cmass0: float = -5.0,
        Cin0: float = 1.0,
        Rim0: float = 1.0,
        Ria0: float = 1.0,
        alpha0: float = 1.0,
    ):
        super().__init__()
        self.raw_Cm = nn.Parameter(torch.tensor(Cmass0, dtype=torch.float32))  # .requires_grad_(False)
        self.log_Cin = nn.Parameter(torch.log(torch.tensor(Cin0, dtype=torch.float32)))  # .requires_grad_(False)
        self.log_Rim = nn.Parameter(torch.log(torch.tensor(Rim0, dtype=torch.float32)))  # .requires_grad_(False)
        self.log_Ria = nn.Parameter(torch.log(torch.tensor(Ria0, dtype=torch.float32)))  # .requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha0, dtype=torch.float32)))  # .requires_grad_(False)

        self.relative_Pihc_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)).requires_grad_(False)
        self.relative_Pis_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)).requires_grad_(False)

        # some init, scales are set during fit
        self.t0 = torch.zeros(())
        self.P0hc = torch.zeros(())
        self.P0s = torch.zeros(())
        self.dT = torch.zeros(())

        # bounds
        self.log_min_Cm = torch.log(torch.tensor(1e0))
        self.log_max_Cm = torch.log(torch.tensor(1e10))

    # ---- physical parameters --------------------------------------------
    def C_mass(self):
        coefficient = torch.sigmoid(self.raw_Cm)
        log_Cmass = self.log_min_Cm + coefficient * (self.log_max_Cm - self.log_min_Cm)
        return torch.exp(log_Cmass)

    def C_in(self):
        return torch.exp(self.log_Cin)

    def R_im(self):
        return torch.exp(self.log_Rim)

    def R_ia(self):
        return torch.exp(self.log_Ria)

    def alpha(self):
        return torch.exp(self.log_alpha)

    def relative_Pihc(self):
        return self.relative_Pihc_raw  # torch.sigmoid(self.relative_Pihc_raw)

    def relative_Pis(self):
        return self.relative_Pis_raw  # torch.sigmoid(self.relative_Pis_raw)

    # ---- physical parameters --------------------------------------------

    # ---- k-coefficients (dimension-less) --------------------------------
    def k_im_mass(self):
        return self.t0 / (self.R_im() * self.C_mass())  # scales (theta_in_pred - theta_thermal_mass_pred)

    def k_im_in(self):
        return self.t0 / (self.R_im() * self.C_in())  # scales (theta_thermal_mass_pred - theta_in_pred)

    def k_ia_in(self):
        return self.t0 / (self.R_ia() * self.C_in())  # scales (theta_amb - theta_in_pred)

    def k_Phc_in(self):
        return self.t0 * self.P0hc / (self.C_in() * self.dT)  # scales Pihc

    def k_Phc_mass(self):
        return self.t0 * self.P0hc / (self.C_mass() * self.dT)  # scales Pihc

    def k_Ps_in(self):
        return self.t0 * self.P0s * self.alpha() / (self.C_in() * self.dT)  # scales Pis

    def k_Ps_mass(self):
        return self.t0 * self.P0s * self.alpha() / (self.C_mass() * self.dT)  # scales Pis

    # ---- k-coefficients (dimension-less) --------------------------------

    def get_phyiscal_RCparams(self) -> dict[str, float]:
        return {
            "C_mass": round(self.C_mass().item(), 2),
            "C_in": round(self.C_in().item(), 2),
            "R_im": round(self.R_im().item(), 6),
            "R_ia": round(self.R_ia().item(), 6),
            "alpha": round(self.alpha().item(), 4),
        }

    @torch.no_grad()
    def _set_scales(self, t0: torch.Tensor, P0hc: torch.Tensor, P0s: torch.Tensor, dT: torch.Tensor) -> None:
        dtype = self.log_Cin.dtype
        self.t0 = torch.as_tensor(t0, dtype=dtype)
        self.P0hc = torch.as_tensor(P0hc, dtype=dtype)
        self.P0s = torch.as_tensor(P0s, dtype=dtype)
        self.dT = torch.as_tensor(dT, dtype=dtype)
