from pathlib import Path
from typing import Tuple
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from tqdm import tqdm
import yaml

from dimlessParams import DimlessRC
from MLP import DNN, SolarShares
from utils import (
    create_log_dir,
    evaluate_predictions,
    get_data,
    plot_all_weights_evolution,
    plot_data,
    plot_data_IDAICE,
    plot_weights_evolution,
    simulate_2R2C,
)

# torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

plt.style.use(style="default")
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
        # "figure.figsize": (10, 6),
        "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage{bm} \usepackage{siunitx}",
    }
)


class PINN(nn.Module):
    def __init__(
        self,
        NN_structure: list[int] = [3, 128],
        n_frequencies: int = 3,
        encoding_method: str = "harmonic",
        use_RFF: bool = True,
        use_RWF: bool = True,
        use_solar_features: bool = True,
    ):
        super().__init__()
        self.eps = 1e-10
        self.training_steps = 0

        self.n_hidden_layers = NN_structure[0]
        self.neurons_per_layer = NN_structure[1]
        self.n_frequencies = n_frequencies
        self.encoding_method = encoding_method

        self.use_RFF = use_RFF
        self.use_RWF = use_RWF
        self.use_solar_features = use_solar_features

        self.approximate_solution = DNN(
            input_dimension=1,
            output_dimension=2,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_neurons=self.neurons_per_layer,
            use_RFF=self.use_RFF,
            n_frequencies=self.n_frequencies,
            encoding_method=self.encoding_method,
            use_RWF=self.use_RWF,
        )

        if self.use_solar_features:
            self.solar_shares = SolarShares(init_shares=(0.25, 0.25, 0.25, 0.25))

        self.dimlessRCparams = DimlessRC()
        self.RCparams_history = {name: [] for name in self.dimlessRCparams.get_phyiscal_RCparams().keys()}

        self.lambdas = {
            "data": torch.tensor(1.0, dtype=torch.float32),
            "physics_in": torch.tensor(1.0, dtype=torch.float32),
            "physics_mass": torch.tensor(1.0, dtype=torch.float32),
        }
        self.total_loss_history = []
        self.data_loss_history = []
        self.data_loss_scaled_history = []
        self.physics_loss_in_history = []
        self.physics_loss_in_scaled_history = []
        self.physics_loss_mass_history = []
        self.physics_loss_mass_scaled_history = []
        self.frames = []

    def assemble_NN_input(self, tau: torch.Tensor, initial_condition=False):
        if initial_condition:
            return tau[0:1].unsqueeze(-1)  # (N,1)
        return tau.unsqueeze(-1)  # (N,1)

    def assemble_physics_input(
        self,
        theta_amb: torch.Tensor,
        Pis: torch.Tensor,
        Pihc: torch.Tensor,
    ) -> torch.Tensor:
        ta = theta_amb.view(-1, 1)
        phc = Pihc.view(-1, 1)
        if Pis.dim() == 1:
            ps = Pis.view(-1, 1)  # -> [N,1]
            return torch.cat([ta, ps, phc], dim=1)  # [N,3]
        elif Pis.dim() == 2 and Pis.size(1) == 4:
            ps = Pis  # -> [N,4]
            return torch.cat([ta, ps, phc], dim=1)  # [N,1+4+1] = [N,6]
        else:
            raise ValueError("Pis must be [N], [N,1], or [N,4].")

    def create_dimless_physics_input(
        self,
    ) -> None:
        # set scales
        self.t_min_train = self.t.min()
        self.t0_train = 1 / 2 * (self.t.max() - self.t_min_train)  # [s]
        q95, q05 = (
            torch.quantile(self.T_in_true, torch.tensor(0.997)),
            torch.quantile(self.T_in_true, torch.tensor(0.003)),
        )
        self.dT = torch.clamp(q95 - q05, min=1.0)
        self.T_ref = self.T_in_true.mean()  # [°C]
        self.P0hc = torch.tensor(1000.0)
        self.P0s = torch.tensor(100.0)

        self.duration_seconds = (self.t.max() - self.t.min()).item()
        self.duration_days = self.duration_seconds / (3600.0 * 24.0)

        # create dimless features
        self.tau = ((self.t - self.t.min()) / self.t0_train) - 1.0  # t [-1,1]
        self.theta_in_true = (self.T_in_true - self.T_ref) / self.dT
        self.theta_mass_true = (self.T_mass_true - self.T_ref) / self.dT if self.T_mass_true is not None else None
        self.theta_amb = (self.T_amb - self.T_ref) / self.dT
        self.Pihc = self.Phc / self.P0hc
        if self.use_solar_features:
            self.Pis = self.directional_irradiances / self.P0s
        else:
            self.Pis = self.Ps / self.P0s

        # set scales in dimless param class and for fourier features
        self.dimlessRCparams._set_scales(t0=self.t0_train, P0hc=self.P0hc, P0s=self.P0s, dT=self.dT)
        self.approximate_solution._set_scales(theta_in=self.theta_in_true)

    def create_collocation_points(
        self,
        n_points: int = 4096,
        random_sampling: bool = False,
    ) -> None:
        tau_min = self.tau[0]
        tau_max = self.tau[-1]
        self.is_random_sampling = random_sampling
        if random_sampling:
            self.tau_collocation, _ = torch.sort(torch.rand(n_points) * (tau_max - tau_min) + tau_min)
        else:
            self.tau_collocation = torch.linspace(tau_min, tau_max, steps=n_points)

        Pihc_collocation = self.linear_interpolation(
            sampled_input=self.Pihc,
            sampled_t=self.tau,
            t_query=self.tau_collocation,  # , mode="zoh"
        )
        theta_amb_collocation = self.linear_interpolation(
            sampled_input=self.theta_amb,
            sampled_t=self.tau,
            t_query=self.tau_collocation,
        )

        if self.use_solar_features:
            GNc = self.linear_interpolation(self.Pis[:, 0:1], self.tau, self.tau_collocation).view(-1)
            GEc = self.linear_interpolation(self.Pis[:, 1:2], self.tau, self.tau_collocation).view(-1)
            GSc = self.linear_interpolation(self.Pis[:, 2:3], self.tau, self.tau_collocation).view(-1)
            GWc = self.linear_interpolation(self.Pis[:, 3:4], self.tau, self.tau_collocation).view(-1)
            Pis_collocation = torch.stack([GNc, GEc, GSc, GWc], dim=1)  # [Nc,4] W/m^2
        else:
            Pis_collocation = self.linear_interpolation(
                sampled_input=self.Pis,
                sampled_t=self.tau,
                t_query=self.tau_collocation,
            )

        self.collocation_points = self.assemble_physics_input(
            theta_amb=theta_amb_collocation,
            Pis=Pis_collocation,
            Pihc=Pihc_collocation,
        )
        self.collocation_input = self.assemble_NN_input(tau=self.tau_collocation)
        self.collocation_input.requires_grad_(True)

    def compute_ode_residual(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = self.approximate_solution(self.collocation_input)
        theta_in_pred, theta_thermal_mass_pred = predictions[:, 0], predictions[:, 1]

        self.dtheta_in_dtau = torch.autograd.grad(
            theta_in_pred, self.collocation_input, grad_outputs=torch.ones_like(theta_in_pred), create_graph=True
        )[0]  # (N,1)

        self.dtheta_mass_dtau = torch.autograd.grad(
            theta_thermal_mass_pred,
            self.collocation_input,
            grad_outputs=torch.ones_like(theta_thermal_mass_pred),
            create_graph=True,
        )[0]  # (N,1)

        theta_amb = self.collocation_points[:, 0]
        if self.use_solar_features:
            Pis = self.solar_shares(self.collocation_points[:, 1:5])
            Pihc = self.collocation_points[:, 5]
        else:
            Pis = self.collocation_points[:, 1]
            Pihc = self.collocation_points[:, 2]

        k_im_mass = self.dimlessRCparams.k_im_mass()
        k_im_in = self.dimlessRCparams.k_im_in()
        k_ia_in = self.dimlessRCparams.k_ia_in()
        k_Phc_in = self.dimlessRCparams.k_Phc_in()
        k_Phc_mass = self.dimlessRCparams.k_Phc_mass()
        k_Ps_in = self.dimlessRCparams.k_Ps_in()
        k_Ps_mass = self.dimlessRCparams.k_Ps_mass()

        relative_Pihc_gain_air = self.dimlessRCparams.relative_Pihc()
        relative_Pis_gain_air = self.dimlessRCparams.relative_Pis()

        f_mass = (
            k_im_mass * (theta_in_pred - theta_thermal_mass_pred)
            + (1 - relative_Pihc_gain_air) * k_Phc_mass * Pihc
            + (1 - relative_Pis_gain_air) * k_Ps_mass * Pis
        ).unsqueeze(-1)  # (N,1)
        f_in = (
            k_im_in * (theta_thermal_mass_pred - theta_in_pred)
            + k_ia_in * (theta_amb - theta_in_pred)
            + relative_Pihc_gain_air * k_Phc_in * Pihc
            + relative_Pis_gain_air * k_Ps_in * Pis
        ).unsqueeze(-1)  # (N,1)

        residual_in = self.dtheta_in_dtau - f_in
        residual_mass = self.dtheta_mass_dtau - f_mass

        return residual_in, residual_mass

    def compute_physics_losses(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_in, residual_mass = self.compute_ode_residual()
        loss_in = torch.mean(residual_in**2)
        loss_mass = torch.mean(residual_mass**2)
        return loss_in, loss_mass

    def compute_data_loss(
        self,
    ) -> torch.Tensor:
        predictions = self.approximate_solution(self.batch_input)
        theta_in_pred, _ = predictions[:, 0], predictions[:, 1]
        loss_data = torch.nn.functional.huber_loss(input=theta_in_pred, target=self.batch_theta)
        return loss_data

    def fit(
        self,
        data: pd.Series | pd.DataFrame,
        loss_weights: dict["str", float],
        n_training_steps: int = 100_000,
        n_training_steps_pre_train: int = 0,
        path_to_model_dir: str | None = None,
    ) -> None:
        self.datetime = data.datetime
        self.t = torch.tensor(data["t"].values, dtype=torch.float32)

        self.T_in_true = torch.tensor(data["T_in_true"].values, dtype=torch.float32)
        if "T_in_true_GT" in data:
            self.T_in_true_GT = torch.tensor(data["T_in_true_GT"].values, dtype=torch.float32)
        else:
            self.T_in_true_GT = None

        if "T_mass_true" in data:
            self.T_mass_true = torch.tensor(data["T_mass_true"].values, dtype=torch.float32)
        else:
            self.T_mass_true = None

        self.T_amb = torch.tensor(data["T_amb"].values, dtype=torch.float32)
        self._Ph = torch.tensor(data["Ph"].values, dtype=torch.float32)
        self._Pc = torch.tensor(data["Pc"].values, dtype=torch.float32)
        self.Phc = self._Ph - self._Pc

        if self.use_solar_features:
            # Order must match SolarMixerShares: [N, E, S, W]
            self.directional_irradiances = torch.tensor(
                data[["PsN", "PsE", "PsS", "PsW"]].to_numpy(), dtype=torch.float32
            )  # shape [N,4], units W/m²
        else:
            self.Ps = torch.tensor(data["Ps"].values, dtype=torch.float32)

        if "Punobserved" in data:
            self.Pu = torch.tensor(data["Punobserved"].values, dtype=torch.float32)
        else:
            self.Pu = None

        self.t0_in = data["T_in_true"].to_numpy()[0]
        if self.T_mass_true is not None:
            self.t0_m = self.T_mass_true.numpy()[0]

        self.n_training_samples = self.T_in_true.shape[0]

        self.create_dimless_physics_input()

        self.input = self.assemble_NN_input(
            tau=self.tau,
        )
        self.target = self.theta_in_true

        n_training_steps = int(n_training_steps)

        self.n_collocation_points = self.T_in_true.shape[0] * 4
        self.create_collocation_points(n_points=self.n_collocation_points)

        # Update the encoder frequencies based on the actual data duration
        self.approximate_solution.update_encoding_duration(
            tau=self.collocation_input, duration_days=self.duration_days
        )

        if n_training_steps_pre_train:
            self._train_ADAM(n_training_steps=n_training_steps_pre_train, loss_weights=loss_weights)
            self._train_LBFGS(
                n_training_steps=n_training_steps,
                loss_balancing_weights=loss_weights,
            )

        elif path_to_model_dir:
            checkpoint = torch.load(path_to_model_dir + "/saved_model.pth")
            self.load_state_dict(checkpoint["model"])
            self._train_LBFGS(
                n_training_steps=n_training_steps,
                loss_balancing_weights=loss_weights,
            )
        else:
            self._train_LBFGS(
                n_training_steps=n_training_steps,
                loss_balancing_weights=loss_weights,
            )

    def _train_ADAM(
        self,
        n_training_steps: int,
        loss_weights: dict[str, float],
        learning_rate: float = 1e-3,
        scheduler_gamma: float = 0.9,
        scheduler_step: int = 3000,
        log_every: int = 100,
        ramp_start_step: int = 1000,
    ) -> None:
        training_step = self.training_steps

        # loss weights
        target_data = torch.tensor(loss_weights["data"], dtype=torch.float32)
        target_physics_in = torch.tensor(loss_weights["physics_in"], dtype=torch.float32)
        target_physics_mass = torch.tensor(loss_weights["physics_mass"], dtype=torch.float32)

        self.lambdas["data"] = target_data.clone()
        ramp_up_steps = n_training_steps // 5
        if ramp_up_steps > 0:
            self.lambdas["physics_in"] = torch.tensor(0.0, dtype=torch.float32)
            self.lambdas["physics_mass"] = torch.tensor(0.0, dtype=torch.float32)
        else:
            self.lambdas["physics_in"] = target_physics_in.clone()
            self.lambdas["physics_mass"] = target_physics_mass.clone()

        # full batch
        self.batch_input = self.input
        self.batch_theta = self.target

        optimizer = torch.optim.Adam(
            [
                {"params": self.approximate_solution.parameters(), "lr": learning_rate},
                # {"params": self.solar_shares.parameters(), "lr": learning_rate},
                {"params": self.dimlessRCparams.parameters(), "lr": learning_rate * 10},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

        pbar_ADAM = tqdm(range(n_training_steps), desc="Adam")

        for _ in pbar_ADAM:
            optimizer.zero_grad()

            loss_data = self.compute_data_loss()
            loss_in, loss_mass = self.compute_physics_losses()

            if ramp_up_steps > 0:
                if training_step < ramp_start_step:
                    progress = 0.0
                else:
                    progress = min(1.0, (training_step - ramp_start_step) / float(max(1, ramp_up_steps)))
                self.lambdas["physics_in"] = target_physics_in * float(progress)
                self.lambdas["physics_mass"] = target_physics_mass * float(progress)

            total_loss = torch.log(
                self.lambdas["data"] * loss_data
                + self.lambdas["physics_in"] * loss_in
                + self.lambdas["physics_mass"] * loss_mass
            )

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.dimlessRCparams.raw_Cm.data.clamp_(max=4.0)

            # scheduler
            if (training_step + 1) % scheduler_step == 0:
                scheduler.step()

            # logging
            if training_step % log_every == 0:
                with torch.no_grad():
                    self.log_RCparams()

                    self.total_loss_history.append(loss_data.item() + loss_in.item() + loss_mass.item())
                    self.data_loss_history.append(loss_data.item())
                    self.data_loss_scaled_history.append((self.lambdas["data"] * loss_data).item())
                    self.physics_loss_in_history.append(loss_in.item())
                    self.physics_loss_in_scaled_history.append((self.lambdas["physics_in"] * loss_in).item())
                    self.physics_loss_mass_history.append(loss_mass.item())
                    self.physics_loss_mass_scaled_history.append((self.lambdas["physics_mass"] * loss_mass).item())

                    current_params = self.dimlessRCparams.get_phyiscal_RCparams()

                    pbar_ADAM.set_postfix(
                        {
                            "DataLoss": round(torch.log10(loss_data).item(), 2),
                            "ODELoss": round(torch.log10(loss_in + loss_mass).item(), 2),
                            "λpi": f"{self.lambdas['physics_in']:.2}",
                            "λpm": f"{self.lambdas['physics_mass']:.2}",
                            "Cmass": f"{current_params['C_mass']:.2e}",
                            "Cin": f"{current_params['C_in']:.2e}",
                            "Rim": f"{current_params['R_im']:.2}",
                            "Ria": f"{current_params['R_ia']:.2}",
                            "α": f"{current_params['alpha']:.2}",
                            "t0_m": round(
                                (self.approximate_solution.init_theta_mass.detach() * self.dT + self.T_ref).item(),
                                2,
                            ),
                        }
                    )
                    # self.frames.append(self.visualize_training_procedure(training_step))

            training_step += 1

        pbar_ADAM.close()
        self.visualize_training_result_plot = self.visualize_training_results()
        self.training_steps = training_step

    def _train_LBFGS(
        self,
        n_training_steps: int,
        loss_balancing_weights: dict["str", float],
        log_every: int = 100,
    ) -> None:
        pbar_LBFGS = tqdm(total=n_training_steps, desc="LBFGS")
        training_step = self.training_steps

        # full batch
        self.batch_input = self.input
        self.batch_theta = self.target

        self.lambdas["data"] = torch.tensor(loss_balancing_weights["data"], dtype=torch.float32)
        self.lambdas["physics_in"] = torch.tensor(loss_balancing_weights["physics_in"], dtype=torch.float32)
        self.lambdas["physics_mass"] = torch.tensor(loss_balancing_weights["physics_mass"], dtype=torch.float32)

        optimizer = torch.optim.LBFGS(
            list(self.approximate_solution.parameters())
            # + list(self.solar_shares.parameters())
            + list(self.dimlessRCparams.parameters()),
            lr=0.5,
            max_iter=int(n_training_steps),
            max_eval=50000,
            history_size=150,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
        )

        def closure():
            nonlocal training_step
            optimizer.zero_grad()
            loss_data = self.compute_data_loss()
            loss_in, loss_mass = self.compute_physics_losses()

            total_loss = torch.log(
                self.lambdas["data"] * loss_data
                + self.lambdas["physics_in"] * loss_in
                + self.lambdas["physics_mass"] * loss_mass
            )

            total_loss.backward()

            if training_step % log_every == 0:
                with torch.no_grad():
                    self.log_RCparams()
                    self.total_loss_history.append(loss_data.item() + loss_in.item() + loss_mass.item())
                    self.data_loss_history.append(loss_data.item())
                    self.data_loss_scaled_history.append((self.lambdas["data"] * loss_data).item())
                    self.physics_loss_in_history.append(loss_in.item())
                    self.physics_loss_in_scaled_history.append((self.lambdas["physics_in"] * loss_in).item())
                    self.physics_loss_mass_history.append(loss_mass.item())
                    self.physics_loss_mass_scaled_history.append(self.lambdas["physics_mass"] * loss_mass.item())
                    current_params = self.dimlessRCparams.get_phyiscal_RCparams()
                    pbar_LBFGS.set_postfix(
                        {
                            "DataLoss": round(torch.log10(loss_data).item(), 2),
                            "ODELoss": round(torch.log10(loss_in + loss_mass).item(), 2),
                            "Cmass": f"{current_params['C_mass']:.2e}",
                            "Cin": f"{current_params['C_in']:.2e}",
                            "Rim": f"{current_params['R_im']:.2}",
                            "Ria": f"{current_params['R_ia']:.2}",
                            "α": f"{current_params['alpha']:.2}",
                            "t0_m": round(
                                (self.approximate_solution.init_theta_mass.detach() * self.dT + self.T_ref).item(), 2
                            ),
                        }
                    )
                    # self.frames.append(self.visualize_training_procedure(current_training_step=training_step))

            pbar_LBFGS.update(1)
            training_step += 1

            return total_loss

        optimizer.step(closure)

        pbar_LBFGS.close()
        self.visualize_training_result_plot = self.visualize_training_results()
        self.training_steps = training_step

    @torch.no_grad()
    def simulate(
        self,
        t: np.ndarray,
        T_amb: np.ndarray,
        Ps: np.ndarray,
        Ph: np.ndarray,
        Pc: np.ndarray,
        initial_Tin: float,
        initial_Tmass: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        RC_params = self.dimlessRCparams.get_phyiscal_RCparams()
        RC_params["t0_in"] = initial_Tin
        RC_params["t0_m"] = initial_Tmass
        RC_params["relative_Pihc_gain_air"] = self.dimlessRCparams.relative_Pihc().item()
        RC_params["relative_Pis_gain_air"] = self.dimlessRCparams.relative_Pis().item()

        X = simulate_2R2C(params=RC_params, t=t, T_amb=T_amb, Ps=Ps, Ph=Ph, Pc=Pc)
        T_in_sim, T_mass_sim = X[:, 0], X[:, 1]
        return T_in_sim, T_mass_sim

    @torch.no_grad()
    def evaluate(
        self,
        evaluation_data: str | pd.DataFrame,
        do_example_plots: bool = False,
        target_RCParams: dict | None = None,
        tag_for_logging: str | None = None,
    ) -> Path:
        """Is introduced to evaluate the results from the PINN training.
        To see possible accuracy it sets the true indoor and mass temperature if available."""
        log_dir = create_log_dir(self, base_dir="logs", tag=tag_for_logging)
        ckpt = {
            "model": self.state_dict(),
            "scales": {
                "t_span_seconds": float(self.duration_seconds),
                "t_span_days": float(self.duration_days),
                "dt": int(torch.mean(torch.diff(self.t)).item()),
                "t0_train": float(self.t0_train),
                "t_min_train": float(self.t_min_train),
                "dT": float(self.dT),
                "T_ref": float(self.T_ref),
                "P0hc": float(self.P0hc),
                "P0s": float(self.P0s),
                "Ti0": float(self.approximate_solution.init_theta_in * self.dT + self.T_ref),
                "Tm0": float(self.approximate_solution.init_theta_mass * self.dT + self.T_ref),
            },
            "building_params": {
                "learned RCs": self.dimlessRCparams.get_phyiscal_RCparams(),
                "relative_Phc_gain_air": self.dimlessRCparams.relative_Pihc().item(),
                "relative_Ps_gain_air": self.dimlessRCparams.relative_Pis().item(),
                "solar shares": self.solar_shares.get_solar_shares() if self.use_solar_features else None,
            },
            "configurations": {
                "PINN_layers": self.n_hidden_layers,
                "PINN_neurons": self.neurons_per_layer,
                "PINN_n_frequencies": self.n_frequencies,
                "PINN_encoding_method": self.encoding_method,
                "PINN_use_RFF": self.use_RFF,
                "PINN_use_RWF": self.use_RWF,
                "PINN_learn_init_mass_temp": self.approximate_solution.learn_init_mass_temp,
                "n_collocation_points": self.n_collocation_points,
                "is_random_sampling": self.is_random_sampling,
                "random_sampling_seed": torch.initial_seed(),
                "data": self.lambdas["data"].item(),
                "physics_in": self.lambdas["physics_in"].item(),
                "physics_mass": self.lambdas["physics_mass"].item(),
            },
            "results": {},
        }
        torch.save(ckpt, log_dir / "saved_model.pth")

        if isinstance(evaluation_data, str):
            data = get_data(path_to_data=evaluation_data)
        else:
            data = evaluation_data

        if "T_mass_true" in data:
            plot_data(data=self.tensors_to_dataframe(), path=log_dir, add_tag="train")
            plot_data(data=data, path=log_dir, add_tag="evaluate")
        else:
            if self.use_solar_features:
                data["Ps"] = (
                    self.solar_shares(torch.tensor(data[["PsN", "PsE", "PsS", "PsW"]].values)).detach().numpy()
                )
            plot_data_IDAICE(data=self.tensors_to_dataframe(), path=log_dir, add_tag="train")
            plot_data_IDAICE(data=data, path=log_dir, add_tag="evaluate")

        T_in_true = data["T_in_true"].to_numpy()
        T_mass_true = data["T_mass_true"].to_numpy() if "T_mass_true" in data else None

        if "T_in_true_GT" in data:
            T_in_true_GT = data["T_in_true_GT"].to_numpy()
            T_mass_true_GT = data["T_mass_true_GT"].to_numpy() if "T_mass_true_GT" in data else None

        # Loss History -----------------------------------------------------------------------
        plt.plot(self.total_loss_history, label="Total Loss")
        plt.grid(True, alpha=0.1)
        plt.legend(frameon=False)
        plt.xlabel(r"$\propto$ Training steps")
        plt.ylabel(r"$\propto$ Loss")
        # plt.title("Training Loss")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(
            log_dir / "total_loss.pdf",
            format="pdf",
            dpi=300,
        )
        plt.close()

        plt.plot(self.data_loss_history, label=r"$L_\mathrm{data}$")
        plt.plot(self.physics_loss_in_history, label=r"$L_\mathrm{physics,in}$")
        plt.plot(self.physics_loss_mass_history, label=r"$L_\mathrm{physics,mass}$")
        plt.grid(True, alpha=0.1)
        plt.legend(frameon=False)
        plt.xlabel(r"$\propto$ Training steps")
        plt.ylabel("Loss")
        # plt.title("Training Loss")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(
            log_dir / "loss.pdf",
            dpi=300,
            format="pdf",
        )
        plt.close()

        plt.plot(self.data_loss_scaled_history, label=r"$\lambda_\mathrm{data}\,L_\mathrm{data}$")
        plt.plot(self.physics_loss_in_scaled_history, label=r"$\lambda_\mathrm{physics,in}\,L_\mathrm{physics,in}$")
        plt.plot(
            self.physics_loss_mass_scaled_history, label=r"$\lambda_\mathrm{physics,mass}\,L_\mathrm{physics,mass}$"
        )
        plt.grid(True, alpha=0.1)
        plt.legend(frameon=False)
        plt.xlabel(r"$\propto$ Training steps")
        plt.ylabel("Loss")
        # plt.title("Scaled Training Loss")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(
            log_dir / "loss_balancing.pdf",
            dpi=300,
            format="pdf",
        )
        plt.close()

        Pu_pred = np.zeros_like(data["T_amb"].to_numpy())

        T_in_sim, T_mass_sim = self.simulate(
            t=data["t"].to_numpy(),
            T_amb=data["T_amb"].to_numpy(),
            Ps=data["Ps"].to_numpy(),
            Ph=data["Ph"].to_numpy(),
            Pc=data["Pc"].to_numpy(),
            initial_Tin=T_in_true[0],
            initial_Tmass=T_mass_true[0] if T_mass_true is not None else T_in_true[0],
        )

        evaluate_predictions(
            data=data,
            T_in_true=T_in_true,
            T_in_sim=T_in_sim,
            Punobserved_pred=Pu_pred,
            T_mass_sim=T_mass_sim,
            log_dir=log_dir,
            T_mass_true=T_mass_true,
        )

        max_horizon = 4 * 24 * 7
        errors = (T_in_sim - T_in_true).reshape(-1, 1)
        n_samples = np.arange(1, len(errors[:max_horizon]) + 1)
        cumulative_mse = np.cumsum(errors[:max_horizon] ** 2) / n_samples
        cumulative_rmse = np.sqrt(cumulative_mse)
        plt.plot(data["t"].iloc[:max_horizon].to_numpy() / (60 * 60), cumulative_rmse, label="RMSE")
        plt.title("Error Evolution vs Forecast Horizon")
        plt.xlabel("Forecast Horizon (Hours)")
        plt.xticks([1, 12, 24, 48, 72, 96, 120, 144, 168])
        plt.ylabel("Error [°C]")
        plt.legend(frameon=False)
        plt.savefig(
            log_dir / "rmse_vs_horizon.pdf",
            format="pdf",
        )
        plt.close()

        if do_example_plots:
            stride = 10
            for i in [1, 3, 5]:
                mean_squred_errors = []
                start_idx = 0
                prediction_days = i
                dt = np.median(np.diff(data.t))
                n_prediction_steps = int((prediction_days * 60 * 60 * 24) / dt)
                n_steps = int((stride * 60 * 60 * 24) / dt)

                while start_idx < data.shape[0] - n_prediction_steps:
                    end_idx = start_idx + n_prediction_steps
                    data_prediction_days = data[start_idx:end_idx]
                    T_in_true_prediction_days = T_in_true[start_idx:end_idx]
                    Pu_pred_prediction_days = Pu_pred[start_idx:end_idx]
                    if T_mass_true is not None:
                        T_mass_true_predictions_days = T_mass_true[start_idx:end_idx]
                    else:
                        T_mass_true_predictions_days = None

                    T_in_sim_prediction_days, T_mass_sim_prediction_days = self.simulate(
                        t=data_prediction_days["t"].to_numpy(),
                        T_amb=data_prediction_days["T_amb"].to_numpy(),
                        Ps=data_prediction_days["Ps"].to_numpy(),
                        Ph=data_prediction_days["Ph"].to_numpy(),
                        Pc=data_prediction_days["Pc"].to_numpy(),
                        initial_Tin=T_in_true_prediction_days[0],
                        initial_Tmass=T_mass_true_predictions_days[0]
                        if T_mass_true_predictions_days is not None
                        else T_in_true_prediction_days[0],
                    )

                    mse = evaluate_predictions(
                        data=data_prediction_days,
                        T_in_true=T_in_true_prediction_days,
                        T_in_sim=T_in_sim_prediction_days,
                        Punobserved_pred=Pu_pred_prediction_days,
                        T_mass_sim=T_mass_sim_prediction_days,
                        log_dir=log_dir / Path(f"Predictions_{prediction_days}_days"),
                        T_mass_true=T_mass_true_predictions_days,
                        is_prediction=True,
                        tag=f"{start_idx}_to_{end_idx}",
                    )
                    start_idx += n_steps
                    mean_squred_errors.append(mse)
                ckpt["results"][f"{prediction_days}_day_prediction"] = round(
                    np.sqrt(np.mean(mean_squred_errors)).item(), 2
                )

        if "T_in_true_GT" in data:
            T_in_sim_GT, T_mass_sim_GT = self.simulate(
                t=data["t"].to_numpy(),
                T_amb=data["T_amb"].to_numpy(),
                Ps=data["Ps"].to_numpy(),
                Ph=data["Ph"].to_numpy(),
                Pc=data["Pc"].to_numpy(),
                initial_Tin=T_in_true_GT[0],
                initial_Tmass=T_mass_true_GT[0] if T_mass_true_GT is not None else T_in_true_GT[0],
            )
            evaluate_predictions(
                data=data,
                T_in_true=T_in_true_GT,
                T_in_sim=T_in_sim_GT,
                Punobserved_pred=Pu_pred,
                T_mass_sim=T_mass_sim_GT,
                log_dir=log_dir,
                T_mass_true=T_mass_true_GT,
                is_GT=True,
            )

        # RC params history -----------------------------------------------------------------------
        if target_RCParams:
            plot_weights_evolution(
                learned_weights=self.RCparams_history, true_weights=target_RCParams, path=str(log_dir)
            )
            plot_all_weights_evolution(
                learned_weights=self.RCparams_history, true_weights=target_RCParams, path=str(log_dir)
            )

        else:
            plot_weights_evolution(learned_weights=self.RCparams_history, path=str(log_dir))
            plot_all_weights_evolution(learned_weights=self.RCparams_history, path=str(log_dir))

        fig = self.visualize_training_result_plot
        fig.savefig(
            log_dir / "visualize_training.pdf",
            dpi=300,
            format="pdf",
        )
        plt.close(fig)
        keys_to_keep = ["scales", "building_params", "configurations", "results"]
        subset = {k: ckpt[k] for k in keys_to_keep if k in ckpt}
        with open(log_dir / "config.yaml", "w") as f:
            yaml.dump(subset, f, default_flow_style=False)
        print(f"Saved to {log_dir}")
        return log_dir

    def _ensure_sorted(self, sampled_t, sampled_input):
        if not torch.all(sampled_t[:-1] <= sampled_t[1:]):
            perm = torch.argsort(sampled_t)
            return sampled_t[perm], sampled_input[perm]
        return sampled_t, sampled_input

    def linear_interpolation(
        self,
        sampled_input: torch.Tensor,
        sampled_t: torch.Tensor,
        t_query: torch.Tensor,
        mode: str = "linear",
    ) -> torch.Tensor:
        """
        Interpolate sampled_input(t) at times t_query.

        modes:
        - "linear": linear interp of instantaneous values
        - "zoh": zero-order hold
        """
        sampled_t, sampled_input = self._ensure_sorted(sampled_t, sampled_input)
        # clamp queries into domain
        tq = t_query.clamp(min=sampled_t[0], max=sampled_t[-1])

        idx_upper = torch.searchsorted(sampled_t, tq)
        idx_lower = (idx_upper - 1).clamp(min=0)
        idx_upper = idx_upper.clamp(max=len(sampled_t) - 1)

        t0, t1 = sampled_t[idx_lower], sampled_t[idx_upper]  # shapes match tq
        f0 = sampled_input[idx_lower]  # broadcasting: f0 shape = tq.shape + data-shape
        f1 = sampled_input[idx_upper]

        if mode == "linear":
            # avoid division by zero at endpoints
            denom = t1 - t0
            w = torch.where(denom > 0, (tq - t0) / denom, torch.zeros_like(tq))
            # expand w to broadcast over data shape
            while w.dim() < f0.dim():
                w = w.unsqueeze(-1)
            return f0 + w * (f1 - f0)

        elif mode == "zoh":
            # hold previous value (f0)
            return f0

        else:
            raise ValueError("mode must be one of ['linear','zoh']")

    @torch.no_grad()
    def visualize_training_procedure(
        self,
        current_training_step: int,
    ) -> np.ndarray:
        T_in_pred, T_mass_pred = self.visualize_training()
        std_T_in = self.T_in_true.std()
        if self.T_mass_true is not None:
            std_T_mass = self.T_mass_true.std()
        else:
            std_T_mass = std_T_in

        # fig, axs = plt.subplots(3, 1, sharex=False, figsize=(15, 9), dpi=100, gridspec_kw={"height_ratios": [3, 3, 3]})
        fig, axs = plt.subplots(figsize=(10, 6))

        axs.set_title(f"Training step {current_training_step}")
        axs.plot(self.t, self.T_in_true, label="T_in_true", color="gray")
        axs.plot(self.t, T_in_pred, label="T_in_pred", color="navy")
        axs.legend()
        axs.legend(loc="upper right")
        axs.set_ylabel("Temperature ($^{\\circ}$C)")
        axs.set_ylim(self.T_in_true.min() - std_T_in, self.T_in_true.max() + std_T_in)

        # axs[1].plot(self.t, self.T_mass_true, label="T_m_true", color="gray")
        # axs[1].plot(self.t, T_mass_pred, label="T_m_pred")
        # axs[1].legend()
        # axs[1].legend(loc="upper right")
        # axs[1].set_ylabel("Temperature ($^{\\circ}$C)")
        # axs[1].set_ylim(self.T_mass_true.min() - 3 * std_T_mass, self.T_mass_true.max() + 3 * std_T_mass)

        plt.tight_layout()
        # plt.show()
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches=None)
        buf.seek(0)
        img = Image.open(buf)
        frame = np.asarray(img)
        plt.close(fig)
        return frame

    @torch.no_grad()
    def visualize_training_results(self) -> Figure:
        T_in_pred, T_mass_pred = self.visualize_training()
        std_T_in = self.T_in_true.std()

        t = self.t.numpy() / 3600

        panels = ["T_in"]
        if self.T_mass_true is not None:
            panels.append("T_mass")
        if getattr(self, "P0u", 0) != 0:
            panels.append("P_unobs")

        nrows = len(panels)
        figsize = (10, 6)
        height_ratios = [3] * nrows
        fig, axs = plt.subplots(
            nrows, 1, sharex=False, figsize=figsize, dpi=300, gridspec_kw={"height_ratios": height_ratios}
        )
        axs = np.atleast_1d(axs)
        ax_i = 0
        ax = axs[ax_i]
        ax.plot(t, self.T_in_true, label="Ground Truth", color="gray")
        ax.plot(t, T_in_pred, label="Prediction", color="navy", linestyle="--")
        ax.set_ylabel(r"$T_\mathrm{in}$ [°C]")
        ax.set_xlabel("Time (h)")
        # ax.set_xticks(list(i for i in range(0, int(max(t)) + 2, 24)))
        # ax.set_ylim(self.T_in_true.min() - std_T_in, self.T_in_true.max() + std_T_in)
        ax.legend(loc="upper right", frameon=False)
        ax_i += 1

        # if "T_mass" in panels:
        #     ax = axs[ax_i]
        #     std_T_mass = self.T_mass_true.std()
        #     ax.plot(self.t, self.T_mass_true, label="T_m_true", color="gray")
        #     ax.plot(self.t, T_mass_pred, label="T_m_pred")
        #     ax.set_ylabel("Temperature ($^{\\circ}$C)")
        #     ax.set_ylim(self.T_mass_true.min() - 3 * std_T_mass, self.T_mass_true.max() + 3 * std_T_mass)
        #     ax.legend(loc="upper right")
        #     ax_i += 1

        plt.tight_layout()
        plt.close(fig)
        return fig

    @torch.no_grad()
    def visualize_training(
        self,
        return_dimless: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:  # , np.ndarray]:
        # non-dimensionalise
        tau = ((self.t - self.t_min_train) / self.t0_train) - 1.0  # t [-1,1]

        input_data = self.assemble_NN_input(tau=tau)
        theta_in_pred, theta_m_pred = self.approximate_solution(input_data).unbind(-1)

        if return_dimless:
            return (theta_in_pred.numpy(), theta_m_pred.numpy())

        T_in_pred = theta_in_pred * self.dT + self.T_ref
        T_m_pred = theta_m_pred * self.dT + self.T_ref
        return T_in_pred.numpy(), T_m_pred.numpy()

    def plot_interpolation_diagnostics(self, modes=("linear", "zoh")):
        fig, axes = plt.subplots(len(modes) + 1, 1, figsize=(12, 4 * len(modes)), sharex="col")
        for i, mode in enumerate(modes):
            # Pis_collocation = self.linear_interpolation(
            #     sampled_input=self.Pis, sampled_t=self.tau, t_query=self.tau_collocation, mode=mode
            # )
            Pihc_collocation = self.linear_interpolation(
                sampled_input=self.Pihc, sampled_t=self.tau, t_query=self.tau_collocation, mode=mode
            )
            theta_in_true = self.linear_interpolation(
                sampled_input=self.theta_in_true, sampled_t=self.tau, t_query=self.tau_collocation, mode=mode
            )

            axes[i].scatter(self.tau_collocation.detach(), theta_in_true, label="Tin")
            # axes[i].plot(
            #     self.tau_collocation.detach(), Pis_collocation.detach(), label="Pis", color="black", linestyle="--"
            # )
            axes[i].scatter(
                self.tau_collocation.detach(),
                Pihc_collocation.detach(),
                label="resampled Pihc",
                color="black",
                linestyle="--",
            )
            axes[i].plot(self.tau.detach(), self.Pihc.detach(), label="Pihc", color="gray")
            axes[i].set_ylabel("Inputs")
            axes[i].set_title(f"Mode: {mode}")
            axes[i].legend()
        axes[2].hist(self.tau_collocation, bins=50, color="tab:blue", edgecolor="black", alpha=0.3)
        axes[2].set_ylabel("Density")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_title("Collocation Point Distribution")

        plt.tight_layout()
        plt.show()

    def log_RCparams(self):
        current = self.dimlessRCparams.get_phyiscal_RCparams()
        for name, value in current.items():
            self.RCparams_history[name].append(value)

    def tensors_to_dataframe(self):
        data_dict = {
            "datetime": self.datetime,
            "t": self.t.numpy(),
            "T_in_true": self.T_in_true.numpy(),
            "T_mass_true": self.T_mass_true.numpy() if self.T_mass_true is not None else None,
            "T_amb": self.T_amb.numpy(),
            "Ph": self._Ph.numpy(),
            "Pc": self._Pc.numpy(),
            "Ps": self.solar_shares(self.directional_irradiances).numpy()
            if self.use_solar_features
            else self.Ps.numpy(),
            "Punobserved": self.Pu.numpy() if self.Pu is not None else None,
        }
        return pd.DataFrame(data_dict)
