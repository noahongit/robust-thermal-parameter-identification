from typing import Any, Dict, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from solar_features import SolarShares
from utils import _create_log_dir, plot_all_weights_evolution, plot_weights_evolution

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "lines.linewidth": 1.5,
        "figure.figsize": (10, 6),
        "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage{bm} \usepackage{siunitx}",
    }
)


class DifferentiableSSM(nn.Module):
    def __init__(self, tag: str | None = None):
        super().__init__()
        self.log_dir = _create_log_dir(model=self, tag=tag)

        self.C_ref = 1e6
        self.R_ref = 1e-1

        self.log_Ria = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_Rim = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_Cin = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_Cmass = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.raw_rel_Phc_air = torch.tensor(1.0)
        self.raw_rel_Ps_air = torch.tensor(1.0)

        self.solar_shares = SolarShares(are_learnable=False)

        self.train_loss_history = []
        self.evaluation_loss_history = []
        self.RCparams_history = {name: [] for name in self.write_learned_RCparams()}
        self.register_buffer("C", torch.tensor([[1.0, 0.0]]))

    def _positivity_transform(self, z):
        return torch.exp(z)

    def C_mass(self):
        return self.C_ref * torch.exp(self.log_Cmass)

    def C_in(self):
        return self.C_ref * torch.exp(self.log_Cin)

    def R_im(self):
        return self.R_ref * torch.exp(self.log_Rim)

    def R_ia(self):
        return self.R_ref * torch.exp(self.log_Ria)

    def alpha(self):
        return torch.exp(self.log_alpha)

    def k_im_mass(self) -> torch.Tensor:
        return self.t0 / (self.R_im() * self.C_mass())  # scales (theta_in_pred - theta_thermal_mass_pred)

    def k_im_in(self) -> torch.Tensor:
        return self.t0 / (self.R_im() * self.C_in())  # scales (theta_thermal_mass_pred - theta_in_pred)

    def k_ia_in(self):
        return self.t0 / (self.R_ia() * self.C_in())  # scales (theta_amb - theta_in_pred)

    def k_Phc_in(self) -> torch.Tensor:
        return self.t0 * self.P0hc / (self.C_in() * self.dT)  # scales Pihc

    def k_Phc_mass(self) -> torch.Tensor:
        return self.t0 * self.P0hc / (self.C_mass() * self.dT)  # scales Pihc

    def k_Ps_in(self) -> torch.Tensor:
        return self.t0 * self.P0s * self.alpha() / (self.C_in() * self.dT)  # scales Pis

    def k_Ps_mass(self) -> torch.Tensor:
        return self.t0 * self.P0s * self.alpha() / (self.C_mass() * self.dT)  # scales Pis

    def get_physical_params(self):
        R_ia = self.R_ia()
        R_im = self.R_im()
        C_in = self.C_in()
        C_m = self.C_mass()
        alpha = self.alpha()
        rel_Phc_air = self.raw_rel_Phc_air
        rel_Ps_air = self.raw_rel_Ps_air
        return R_ia, R_im, C_in, C_m, alpha, rel_Phc_air, rel_Ps_air

    def _window_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.mean()
        reduce_dims = tuple(range(1, x.dim()))
        return x.mean(dim=reduce_dims, keepdim=True)

    def create_dimless_physics_input(
        self,
        t: torch.Tensor,
        Tin: torch.Tensor,
        Tamb: torch.Tensor,
        Ph: torch.Tensor,
        Pc: torch.Tensor,
        Ps: torch.Tensor,
        is_eval: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Create dimensionless inputs while preserving the incoming tensor shapes.

        Supported shapes:
          t: [N] or [B, N]
          Tin, Tamb, Ph, Pc: [B, N]
          Ps: [B, N] or [B, N, 4]
        """
        if is_eval and not all(hasattr(self, name) for name in ["dT", "Tref", "P0hc", "P0s", "t0"]):
            raise RuntimeError("Evaluation batch prepared before non-dimensional scales were initialized.")

        if not is_eval:
            self.dT = torch.tensor(2.0)
            self.Tref = self._window_mean(Tin).squeeze()
            self.P0hc = torch.tensor(1000.0)
            self.P0s = torch.tensor(100.0)
            self.t0 = torch.tensor(1e5)

        if t.dim() == 1:
            tau = (t - t[0]) / self.t0
            dtau = (t[1] - t[0]) / self.t0
        elif t.dim() == 2:
            tau = (t - t[:, :1]) / self.t0
            dtau = (t[:, 1] - t[:, 0]).median() / self.t0
        else:
            raise ValueError("Wrong shape, see prepare_batch")

        # create dimless features
        theta_in_true = (Tin - self.Tref) / self.dT
        theta_in0 = theta_in_true[..., 0]
        theta_amb = (Tamb - self.Tref) / self.dT
        Pi_h = Ph / self.P0hc
        Pi_c = Pc / self.P0hc
        Pi_net = Pi_h - Pi_c
        Pi_s = Ps / self.P0s

        return tau, dtau, Pi_net, Pi_h, Pi_c, Pi_s, theta_in_true, theta_amb, theta_in0

    def continuous_matrices(self, nondim: bool = True):
        R_ia, R_im, C_in, C_m, alpha, relative_Pihc_air, relative_Pis_air = self.get_physical_params()

        if not nondim:
            a11 = -(1.0 / (R_ia * C_in) + 1.0 / (R_im * C_in))
            a12 = 1.0 / (R_im * C_in)
            a21 = 1.0 / (R_im * C_m)
            a22 = -(1.0 / (R_im * C_m))
            A = torch.stack([torch.stack([a11, a12]), torch.stack([a21, a22])])

            b11 = relative_Pihc_air / C_in
            b12 = (relative_Pis_air * alpha) / C_in
            b13 = 1.0 / (R_ia * C_in)

            b21 = (1.0 - relative_Pihc_air) / C_m
            b22 = (1.0 - relative_Pis_air) * alpha / C_m
            b23 = torch.tensor(0.0, dtype=A.dtype, device=A.device)
            B = torch.stack([torch.stack([b11, b12, b13]), torch.stack([b21, b22, b23])])
            return A, B

        k_im_mass = self.k_im_mass()
        k_im_in = self.k_im_in()
        k_ia_in = self.k_ia_in()
        k_Phc_in = self.k_Phc_in()
        k_Phc_mass = self.k_Phc_mass()
        k_Ps_in = self.k_Ps_in()
        k_Ps_mass = self.k_Ps_mass()

        A_nondim = torch.stack(
            [
                torch.stack([-(k_ia_in + k_im_in), k_im_in]),
                torch.stack([k_im_mass, -k_im_mass]),
            ]
        )

        B_nondim = torch.stack(
            [
                torch.stack([relative_Pihc_air * k_Phc_in, relative_Pis_air * k_Ps_in, k_ia_in]),
                torch.stack(
                    [
                        (1.0 - relative_Pihc_air) * k_Phc_mass,
                        (1.0 - relative_Pis_air) * k_Ps_mass,
                        torch.tensor(0.0, dtype=A_nondim.dtype, device=A_nondim.device),
                    ]
                ),
            ]
        )

        return A_nondim, B_nondim

    @staticmethod
    def _robust_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Solves AX = B using Least Squares.
        This is robust to near-singular A (e.g. extremely high Resistance).
        """
        return torch.linalg.lstsq(A, B).solution

    # ---------- SOLVE-BASED ZOH ----------
    def discretize_ZOH(self, A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact ZOH (constant input per step):
          Ad = expm(A*dt)
          G0 = solve(A, Ad - I)
          Bd = G0 @ B
        """
        n = A.shape[0]
        I = torch.eye(n, dtype=A.dtype, device=A.device)
        Ad = torch.matrix_exp(A * dt)
        # G0 = torch.linalg.solve(A, Ad - I)  # A*G0 = Ad - I
        G0 = self._robust_solve(A, Ad - I)
        Bd = G0 @ B
        return Ad, Bd

    @staticmethod
    def _ensure_correct_shape(x: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        if x.dim() == 1:
            return False, x.unsqueeze(0)  # [1,N]
        if x.dim() == 2:
            return True, x  # [B,N]
        raise ValueError("Inputs must be [N] or [B,N].")

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        hold: str = "zoh",
    ):
        # unpack dimless
        theta_in_true = batch["theta_in_true"]
        theta_amb = batch["theta_amb"]
        theta_in0 = batch["theta_in0"]
        Pi_net = batch["Pi_net"]
        Pi_s_directional = batch["Pi_s_directional"]
        dtau = batch["dtau"]

        A, Bm = self.continuous_matrices()
        device, dtype = A.device, A.dtype

        B_dim, N = theta_in_true.shape

        if Pi_s_directional is None:
            raise ValueError("Batch missing Ps_directional while use_solar_features=True")

        Ps_eff_stacked = self.solar_shares(Pi_s_directional)

        # form u [B,N,3]
        u = torch.stack([Pi_net, Ps_eff_stacked, theta_amb], dim=-1)

        # initial state
        X_prev = torch.stack([theta_in0, theta_in0], dim=1)  # [B,2]

        # precompute kernels
        mode = hold.lower()
        if mode == "zoh":
            Ad, Bd = self.discretize_ZOH(A, Bm, dtau)

        X = torch.empty(B_dim, N, 2, dtype=dtype, device=device)
        X[:, 0, :] = X_prev
        At = Ad.T
        Ct = self.C.to(dtype=dtype, device=device)

        # Time stepping loop
        for k in range(1, N):
            Xk = X_prev @ At

            if mode == "zoh":
                Xk = Xk + (u[:, k - 1, :] @ Bd.T)

            X[:, k, :] = Xk
            X_prev = Xk

        Y = (Ct @ X.transpose(1, 2)).transpose(1, 2)
        return X, Y

    def _train_ADAM(
        self,
        batch: Dict[str, Any],
        n_training_steps: int = 1000,
        learning_rate: float = 1e-1,
        log_every: int = 50,
        path_to_model: str | None = None,
    ) -> None:
        optimizer = torch.optim.Adam([{"params": self.parameters(), "lr": learning_rate}])
        if path_to_model is not None:
            checkpoint = torch.load(path_to_model)
            self.load_state_dict(checkpoint["model"])

        if "B" in batch and isinstance(batch["B"], int):
            n_total_windows = batch["B"]
        else:
            n_total_windows = 0
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    n_total_windows = v.shape[0]
                    break

        if n_total_windows == 0:
            raise ValueError("Could not determine batch size from input data.")

        pbar_ADAM = tqdm(range(n_training_steps), desc="Adam")

        for training_step in pbar_ADAM:
            optimizer.zero_grad(set_to_none=True)

            _, Y_pred = self(batch=batch)

            Tin_pred = Y_pred[..., 0]

            loss = F.huber_loss(Tin_pred, batch["theta_in_true"])

            loss.backward()
            optimizer.step()

            if (training_step + 1) % log_every == 0:
                with torch.no_grad():
                    Ria, Rim, Cin, Cm, alpha, _, _ = self.get_physical_params()
                    postfix_dict = {
                        "Loss": f"{loss.item():.4f}",
                        "Cm": f"{Cm.item():.2e}",
                        "Cin": f"{Cin.item():.2e}",
                        "Rim": f"{Rim.item():.6f}",
                        "Ria": f"{Ria.item():.6f}",
                        "α": f"{alpha.item():.2f}",
                    }
                    self._log_current_RCparams()
                    self.train_loss_history.append(loss.item())
                    pbar_ADAM.set_postfix(postfix_dict)

        pbar_ADAM.close()
        self._visualize_training_results(batch=batch)

    def _train_LBFGS(
        self,
        batch: Dict[str, Any],
        n_training_steps: int = 1000,
        log_every: int = 50,
        path_to_model: str | None = None,
    ) -> None:
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=0.1,
            max_iter=int(n_training_steps),
            line_search_fn="strong_wolfe",
        )

        pbar = tqdm(total=n_training_steps, desc="LBFGS", leave=False)
        self.training_step = 0

        current_val_loss = float("inf")

        if path_to_model is not None:
            checkpoint = torch.load(path_to_model)
            self.load_state_dict(checkpoint["model"])

        def closure():
            nonlocal current_val_loss

            optimizer.zero_grad(set_to_none=True)

            _, Y_pred = self(batch=batch)
            Tin_pred = Y_pred[..., 0]  # [B,N]
            loss = F.huber_loss(Tin_pred, batch["theta_in_true"])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            if (self.training_step + 1) % log_every == 0:
                with torch.no_grad():
                    Ria, Rim, Cin, Cm, alpha, _, _ = self.get_physical_params()
                    postfix_dict = {
                        "Loss": f"{loss.item():.4f}",
                        "Cm": f"{Cm.item():.2e}",
                        "Cin": f"{Cin.item():.2e}",
                        "Rim": f"{Rim.item():.6f}",
                        "Ria": f"{Ria.item():.6f}",
                        "α": f"{alpha.item():.2f}",
                    }
                    self._log_current_RCparams()
                    self.train_loss_history.append(loss.item())

                    pbar.set_postfix(postfix_dict)

            pbar.update(1)
            self.training_step += 1
            return loss

        self.train()
        optimizer.step(closure)
        pbar.close()
        self._visualize_training_results(batch=batch)

    def _evaluate(
        self,
        batch: Dict[str, torch.Tensor],
        t_min_train: float | None = None,
        t_span_days: float | None = None,
        dt: float | None = None,
        dtau: float | None = None,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        self.eval()

        with torch.no_grad():
            X, Y_pred = self(batch=batch)

        y_true_flat = batch["T_in_true"].detach().cpu().numpy().flatten()
        theta_pred = Y_pred[..., 0].detach().cpu()
        y_pred_flat = (batch["Tref"] + batch["dT"] * theta_pred).numpy().flatten()

        mse_global = np.mean((y_pred_flat - y_true_flat) ** 2)
        rmse_global = np.sqrt(mse_global).item()
        mae_global = np.mean(np.abs(y_pred_flat - y_true_flat)).item()

        print(f"Global RMSE: {rmse_global:.2f}")
        print(f"Global MAE:  {mae_global:.2f}")

        ckpt = {
            "model": self.state_dict(),
            "scales": {
                "t_min_train": t_min_train,
                "t_span_days": t_span_days,
                "dT": self.dT.item() if self.dT.squeeze().dim() == 0 else "B > 1, saved to batch dict",
                "Tref": self.Tref.item() if self.Tref.squeeze().dim() == 0 else "B > 1, saved to batch dict",
                "P0hc": self.P0hc.item(),
                "P0s": self.P0s.item(),
                "t0": self.t0.item(),
                "dtau": dtau,
                "dt": dt,
            },
            "building_params": {
                "learned RCs": self.write_learned_RCparams(),
                "relative_Phc_gain_air": self.raw_rel_Phc_air.item(),
                "relative_Ps_gain_air": self.raw_rel_Ps_air.item(),
                "solar shares": self.solar_shares.get_solar_shares(),
                "learnable solar features": False,
            },
            "configs": {
                "C_ref": self.C_ref,
                "R_ref": self.R_ref,
                "n_stacked_windows": batch["T_in_true"].shape[0],
                "n_timesteps": batch["T_in_true"].shape[1],
            },
            "results": {
                "MAE (Global)": mae_global,
                "RMSE (Global)": rmse_global,
            },
        }

        self.log_dir.mkdir(parents=True, exist_ok=True)

        keys_to_keep = ["building_params", "configs", "results", "scales"]
        subset = {k: ckpt[k] for k in keys_to_keep if k in ckpt}
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(subset, f, default_flow_style=False)

        torch.save(ckpt, self.log_dir / "saved_model.pth")
        print(f"\nSaved model compatible with evaluate.py to {self.log_dir}")

        plt.figure()
        plt.plot(self.train_loss_history, label="Training Loss")
        if self.evaluation_loss_history:
            plt.plot(self.evaluation_loss_history, label="Eval Loss")
        plt.grid(True)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Loss (Huber)")
        plt.title("Training Loss Evolution")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.log_dir / "loss.png", format="png")
        plt.close()

        plot_weights_evolution(learned_weights=self.RCparams_history, path=str(self.log_dir))
        plot_all_weights_evolution(learned_weights=self.RCparams_history, path=str(self.log_dir))

        self.train()
        return mae_global, rmse_global

    def prepare_batch(
        self,
        t,
        Ph,
        T_amb,
        T_in_true,
        Ps,
        Pc,
        is_eval: bool,
        tau_vec: Optional[Any] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        # convert
        t_ = self.to_tensor(t)
        Ph_ = self.to_tensor(Ph)
        T_amb_ = self.to_tensor(T_amb)
        T_in_ = self.to_tensor(T_in_true)
        Ps_ = self.to_tensor(Ps)
        Pc_ = self.to_tensor(Pc)
        tau_vec_ = self.to_tensor(tau_vec) if tau_vec is not None else None

        # required
        if t_ is None or Ph_ is None or T_amb_ is None or T_in_ is None:
            raise ValueError("t, Ph, T_amb and Tin_target are required.")

        # transform to [B,N]
        _, Ph_stacked = self._ensure_correct_shape(Ph_)
        _, T_amb_stacked = self._ensure_correct_shape(T_amb_)
        _, T_in_stacked = self._ensure_correct_shape(T_in_)

        # expand t to t_stacked
        if t_.dim() == 1:
            t_stacked = t_.unsqueeze(0).expand(Ph_stacked.shape[0], -1)
        else:
            raise ValueError("t must be [N].")

        B, N = Ph_stacked.shape
        if T_amb_stacked.shape != (B, N) or T_in_stacked.shape != (B, N):
            raise ValueError("Ph, T_amb and T_in must have same shape after transformation.")

        _, Pc_stacked = self._ensure_correct_shape(Pc_)
        if Pc_stacked.shape != (B, N):
            raise ValueError("Pc shape mismatch after transformation.")

        # either scalar Ps ([N] or [B,N]) OR directional Ps_dir ([N,4] or [B,N,4]).
        Ps_directional_stacked = None
        Ps_directional_ = Ps_
        # accept [N,4] or [B,N,4]
        if Ps_directional_.dim() == 2:
            if Ps_directional_.shape[1] != 4:
                raise ValueError("Ps_dir must have last dim 4 (N,E,S,W).")
            Ps_directional_stacked = Ps_directional_.unsqueeze(0).expand(B, -1, -1)
        elif Ps_directional_.dim() == 3:
            if Ps_directional_.shape[0] == 1:
                Ps_directional_stacked = Ps_directional_.expand(B, -1, -1)
            elif Ps_directional_.shape[0] == B:
                Ps_directional_stacked = Ps_directional_
            else:
                raise ValueError("Ps_directional has incompatible batch dim.")
            if Ps_directional_stacked.shape[2] != 4:
                raise ValueError("Ps_dir must have last dim 4 (N,E,S,W).")
        else:
            raise ValueError("Ps_dir must be [N,4] or [B,N,4].")

        # transform tau_vec if provided: scalar, [N-1], or [B,N-1]
        tau_vec_tn = None
        if tau_vec_ is not None:
            if torch.is_tensor(tau_vec_) and tau_vec_.dim() == 0:
                tau_vec_tn = tau_vec_.expand(B, N - 1)
            elif torch.is_tensor(tau_vec_) and tau_vec_.dim() == 1 and tau_vec_.shape[0] == N - 1:
                tau_vec_tn = tau_vec_.unsqueeze(0).expand(B, -1)
            elif torch.is_tensor(tau_vec_) and tau_vec_.dim() == 2 and tau_vec_.shape == (B, N - 1):
                tau_vec_tn = tau_vec_
            else:
                raise ValueError("tau_vec must be scalar, [N-1], or [B,N-1] tensor.")

        # net power and dt
        P_net_stacked = Ph_stacked - Pc_stacked
        dt = (t_[1] - t_[0]).median()

        Tin0_stacked = T_in_stacked[:, 0]  # [B]

        # non-dim the system
        tau, dtau, Pi_net, Pi_h, Pi_c, Pi_s_directional, theta_in_true, theta_amb, theta_in0 = (
            self.create_dimless_physics_input(
                t=t_stacked,
                Tin=T_in_stacked,
                Tamb=T_amb_stacked,
                Ph=Ph_stacked,
                Pc=Pc_stacked,
                Ps=Ps_directional_stacked,
                is_eval=is_eval,
            )
        )

        batch = {
            "t": t_stacked,
            "tau": tau,
            "dtau": dtau,
            "Tref": self.Tref,
            "dT": self.dT,
            "Ph": Ph_stacked,
            "Pc": Pc_stacked,
            "P_net": P_net_stacked,
            "Pi_h": Pi_h,
            "Pi_c": Pi_c,
            "Pi_net": Pi_net,
            "T_amb": T_amb_stacked,
            "theta_amb": theta_amb,
            "T_in_true": T_in_stacked,
            "theta_in_true": theta_in_true,
            "Tin0": Tin0_stacked,
            "theta_in0": theta_in0,
            "Ps_directional": Ps_directional_stacked,
            "Pi_s_directional": Pi_s_directional,
            "dt": dt,
            "tau_vec": tau_vec_tn,
            "B": B,
            "N": N,
            "device": device,
            "dtype": dtype,
        }

        return batch

    def print_learned_RCparams(self) -> None:
        R_ia, R_im, C_in, C_m, alpha, _, _ = self.get_physical_params()
        print(
            f"Learned RC parameters: \nCmass: {C_m:.2e}\nCin: {C_in:.2e}\nRim: {R_im:.2}\nRia: {R_ia:.2}\nα: {alpha:.2}\n"
        )

    def _log_current_RCparams(self) -> None:
        current = self.write_learned_RCparams()
        for name, value in current.items():
            self.RCparams_history[name].append(value)

    def write_learned_RCparams(self) -> Dict:
        R_ia, R_im, C_in, C_m, alpha, _, _ = self.get_physical_params()
        return {
            "C_mass": round(C_m.item(), 2),
            "C_in": round(C_in.item(), 2),
            "R_im": round(R_im.item(), 6),
            "R_ia": round(R_ia.item(), 6),
            "alpha": round(alpha.item(), 4),
        }

    @torch.no_grad()
    def _visualize_training_results(self, batch: Dict[str, Any]) -> None:
        X, Y_pred = self(batch=batch)

        theta_pred = Y_pred[..., 0].detach().cpu()
        T_in_pred = (batch["Tref"] + batch["dT"] * theta_pred).numpy()
        T_in_true = batch["T_in_true"].detach().cpu().numpy()
        t = batch["t"].detach().cpu().numpy()

        B = T_in_true.shape[0]
        # Pick a subset of 5 evenly spaced samples
        n_plots = min(B, 5)
        indices = np.linspace(0, B - 1, n_plots, dtype=int)

        # Create a vertical stack of subplots
        fig, axs = plt.subplots(n_plots, 1, sharex=True, dpi=300)
        axs = np.atleast_1d(axs)
        for i, idx in enumerate(indices):
            ax = axs[i]
            t_series = t[idx] / 3600
            y_true = T_in_true[idx]
            y_pred = T_in_pred[idx]

            ax.plot(t_series, y_true, label="Ground Truth", color="black", alpha=0.4)
            ax.plot(t_series, y_pred, label="Prediction", color="firebrick", linestyle="--")

            ax.set_ylabel(r"$T_\mathrm{in}$ [°C]")
            ax.set_xticks(list(i for i in range(0, int(max(t_series.squeeze())) + 2, 24)))
            # ax.set_title(f"Training Sample (Window Index: {idx})")
            ax.grid(True, linestyle="--", alpha=0.1)

            if i == 0:
                ax.legend(loc="upper right", frameon=False)

        axs[-1].set_xlabel("Time (h)")
        plt.tight_layout()
        plt.savefig(self.log_dir / "visualize_training.pdf", format="pdf")
        plt.close(fig)

    @staticmethod
    def to_tensor(
        x,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        return x
