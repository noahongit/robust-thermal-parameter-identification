import calendar
from collections import defaultdict
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Callable, List, Optional, Tuple, Dict

from matplotlib import cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy import stats
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation, label
from scipy.signal import cont2discrete
from scipy.linalg import expm, solve
import yaml
from scipy.ndimage import median_filter

from dataclasses import dataclass
from scipy.optimize import minimize
import random

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [15, 10]


def _infer_dt_seconds(idx) -> float:
    """Median step (s) for DatetimeIndex."""
    diffs = np.diff(idx.view("i8"))  # ns
    return float(np.median(diffs)) / 1e9


def _grid_is_uniform(idx, dt_expected: float, tol_s: float = 1.0) -> bool:
    """True if all steps ~= dt_expected within tol_s and no NaNs/duplicates."""
    if not isinstance(idx, pd.DatetimeIndex):
        return False
    d = np.diff(idx.view("i8")) / 1e9  # seconds
    if len(d) == 0:  # single row
        return True
    return np.all(np.abs(d - dt_expected) <= tol_s)


def get_FLEDGED_simulator_data(path_to_data: str) -> pd.DataFrame:
    data = pd.read_csv(filepath_or_buffer=path_to_data, parse_dates=["Unnamed: 0"])
    data.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    data["t"] = (data["datetime"] - data["datetime"].iloc[0]).dt.total_seconds()  # in seconds
    data.rename(
        columns={
            "Ta": "T_amb",
            "G_Gh": "Ps",
            "heating_demand": "Ph",
            "cooling_demand": "Pc",
            "t_air": "T_in_true",
            "mass_temp": "T_mass_true",
        },
        inplace=True,
    )
    data.drop(columns=["RH", "FF"], inplace=True)
    data = data[["datetime", "t", "T_in_true", "T_mass_true", "T_amb", "Ps", "Ph", "Pc"]]
    return data


def get_IDAICE_data(
    path_to_data: str,
    start_date: str | None = None,
    end_date: str | None = None,
    dt: int | None = None,
    mode: str = "linear",
) -> pd.DataFrame:
    data = pd.read_csv(path_to_data, parse_dates=["timestamp"])
    data.rename(
        columns={
            "timestamp": "datetime",
            "Ta": "T_amb",
            "heating_demand": "Ph",
            "cooling_demand": "Pc",
            "t_air": "T_in_true",
        },
        inplace=True,
    )
    keep_cols = ["datetime", "T_amb", "Ph", "Pc", "PsS", "PsN", "PsE", "PsW", "T_in_true"]
    data = data[[c for c in keep_cols if c in data.columns]].copy()

    data = data.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if start_date or end_date:
        data = data.set_index("datetime").loc[start_date:end_date].reset_index()

    if "Pc" in data:
        data["Pc"] = data["Pc"].clip(lower=0.0)
        data.loc[data["Pc"].abs() < 1e-3, "Pc"] = 0.0
    if "Ph" in data:
        data["Ph"] = data["Ph"].clip(lower=0.0)

    data_idx = pd.DatetimeIndex(data["datetime"])
    dt_src = _infer_dt_seconds(data_idx)
    if dt is None:
        dt = int(round(dt_src))
    need_resample = (abs(dt - dt_src) > 1.0) or (not _grid_is_uniform(data_idx, dt_expected=dt))

    if need_resample:
        new_index = pd.date_range(start=data_idx[0], end=data_idx[-1], freq=f"{dt}s")
        df = data.set_index("datetime")
        out = pd.DataFrame(index=new_index)
        for col in df.columns:
            if col in ["Ph", "Pc"]:
                out[col] = df[col].reindex(new_index).ffill()
            else:
                out[col] = resample_series(df[col], new_index, mode=mode, dt_seconds=dt)
        out = out.reset_index().rename(columns={"index": "datetime"})
        dt_used = dt
    else:
        out = data.copy()
        dt_used = dt_src

    out["t"] = (out["datetime"] - out["datetime"].iloc[0]).dt.total_seconds().astype(float)
    out = out[["datetime", "t", "T_amb", "Ph", "Pc", "PsS", "PsN", "PsE", "PsW", "T_in_true"]]
    return out


def get_data(
    path_to_data: str,
    dt: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    data = get_weather_data(path_to_data, dt, start_date, end_date)
    data.rename(
        columns={
            "Ta": "T_amb",
            "G_Gh": "Ps",
            "heating_demand": "Ph",
            "cooling_demand": "Pc",
            "t_air": "T_in_true",
            "mass_temp": "T_mass_true",
        },
        inplace=True,
    )
    data = data.reset_index().rename(columns={"index": "datetime"})
    data["t"] = (data["datetime"] - data["datetime"].iloc[0]).dt.total_seconds()  # in seconds
    data.drop(columns=["RH", "FF"], inplace=True)
    data = data[["datetime", "t", "T_amb", "Ps"]]
    return data


def _infer_dt_seconds(idx) -> float:
    """Median step (s) for DatetimeIndex."""
    diffs = np.diff(idx.view("i8"))  # ns
    return float(np.median(diffs)) / 1e9


def get_weather_data(weather_data_path, dt: int, start: str | None, end: str | None, mode: str = "linear"):
    w = pd.read_csv(weather_data_path)
    w.set_index(pd.to_datetime(w.loc[:, ["year", "month", "day", "hour"]]), inplace=True)
    w = w[["Ta", "RH", "G_Gh", "G_Bn", "G_Dh", "FF"]]
    start = start or w.iloc[0].name
    end = end or w.iloc[-1].name
    w = w.loc[start:end]

    # resample timestamps
    new_index = pd.date_range(start=start, end=end, freq=f"{dt}s")

    # apply chosen interpolation mode per column
    w_resampled = pd.DataFrame(index=new_index)
    for col in w.columns:
        w_resampled[col] = resample_series(w[col], new_index, mode=mode, dt_seconds=dt)

    return w_resampled


def resample_series(series: pd.Series, new_index: np.ndarray, mode: str = "linear", dt_seconds: int = 900):
    if mode == "linear":
        return series.reindex(new_index).interpolate(method="linear")

    elif mode == "zoh":
        return series.reindex(new_index).ffill()

    elif mode == "smooth_step":
        old_index = series.index.to_numpy()
        old_values = series.to_numpy()
        new_values = np.zeros_like(new_index, dtype=float)

        for i, t in enumerate(new_index):
            idx_right = np.searchsorted(old_index, t)
            idx_left = max(0, idx_right - 1)
            idx_right = min(len(old_index) - 1, idx_right)

            t0, t1 = old_index[idx_left], old_index[idx_right]
            v0, v1 = old_values[idx_left], old_values[idx_right]

            if t1 == t0:
                new_values[i] = v0
            else:
                alpha = (t - t0) / (t1 - t0)
                s = 0.5 * (1 + np.tanh((alpha - 0.5) * (t1 - t0) / dt_seconds))
                new_values[i] = v0 * (1 - s) + v1 * s

        return pd.Series(new_values, index=new_index)

    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")


def resample_simulation_data(data: pd.DataFrame, dt_new: int, mode: str = "linear") -> pd.DataFrame:
    """
    Resample simulation data to a new timestep.
    """
    t_start = data["t"].iloc[0]
    t_end = data["t"].iloc[-1]
    new_t = np.arange(t_start, t_end + dt_new, dt_new)

    data_interp = data.set_index("t")
    data_resampled = pd.DataFrame(index=new_t)

    for col in data_interp.columns:
        if col in ["Ph", "Pc", "Punobserved"]:  # binary/step-like signals
            data_resampled[col] = data_interp[col].reindex(new_t, method="ffill")
        else:
            data_resampled[col] = resample_series(data_interp[col], new_t, mode=mode, dt_seconds=dt_new)

    data_resampled = data_resampled.reset_index().rename(columns={"index": "t"})

    if "datetime" in data.columns:
        data_resampled["datetime"] = data["datetime"].iloc[0] + pd.to_timedelta(data_resampled["t"], unit="s")

    return data_resampled


def get_simulation_data(
    path: str,
    GT_RCparams: dict[str, float],
    resampled_dt: int,
    is_evaluation: bool,
    add_Punobserved: bool = True,
    add_white_noise: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    should_save: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = get_data(
        path_to_data=path,
        dt=10,
        start_date=start_date,
        end_date=end_date,
    )
    if is_evaluation:
        X_ground_truth, Ph, Pc, _, Ps_effective = simulate_2R2C_with_thermostat(
            GT_RCparams=GT_RCparams, data=data, add_Punobserved=False
        )
        X, _, _, Punobserved, _ = simulate_2R2C_with_thermostat(
            GT_RCparams=GT_RCparams, data=data, add_Punobserved=add_Punobserved, Ph_provided=Ph, Pc_provided=Pc
        )
    else:
        X, Ph, Pc, Punobserved, Ps_effective = simulate_2R2C_with_thermostat(
            GT_RCparams=GT_RCparams, data=data, add_Punobserved=add_Punobserved
        )
        X_ground_truth, _, _, _, _ = simulate_2R2C_with_thermostat(
            GT_RCparams=GT_RCparams, data=data, add_Punobserved=False, Ph_provided=Ph, Pc_provided=Pc
        )

    T_in_true, T_mass_true = X[:-1, 0], X[:-1, 1]
    T_in_ground_truth, T_mass_ground_truth = X_ground_truth[:-1, 0], X_ground_truth[:-1, 1]

    data["T_in_true"] = T_in_true
    data["T_mass_true"] = T_mass_true
    data["T_in_true_GT"] = T_in_ground_truth
    data["T_mass_true_GT"] = T_mass_ground_truth

    data["Ph"] = Ph
    data["Pc"] = Pc
    data["Punobserved"] = Punobserved
    data["Ps_effective"] = Ps_effective

    data_resampled = resample_simulation_data(data=data, dt_new=resampled_dt)

    if add_white_noise:
        cfg = UnobsGainConfig(mode="white", white_sigma_W=1.0, seed=42)
        white_noise = generate_unobserved_gains(data_resampled, cfg)

        noisy = data_resampled["T_in_true"].to_numpy().copy()
        noisy[1:] = noisy[1:] + white_noise[1:]
        data_resampled["T_in_true"] = noisy

    if should_save:
        if is_evaluation:
            data_resampled.to_csv("evaluation_data.csv", index=False)
            data.to_csv("evaluation_data_sim_res.csv", index=False)

        else:
            data_resampled.to_csv("training_data.csv", index=False)
            data.to_csv("training_data_sim_res.csv", index=False)

    return data, data_resampled


def evaluate_predictions(
    data: pd.DataFrame,
    T_in_true: np.ndarray,
    T_in_sim: np.ndarray,
    Punobserved_pred: np.ndarray,
    T_mass_sim: np.ndarray,
    log_dir: Path,
    T_mass_true: np.ndarray | None = None,
    is_GT: bool = False,
    is_prediction: bool = False,
    tag: str | None = None,
) -> float:
    if is_prediction:
        path = log_dir
        os.makedirs(path, exist_ok=True)
    if not is_GT:
        if "Punobserved" in data:
            plt.plot(data["t"] / 3600, data["Punobserved"], label="True Gains", color="black", alpha=0.4)
            RMSE_Pu = round(np.sqrt(np.mean((data["Punobserved"] - Punobserved_pred) ** 2).item()), 2)
        else:
            RMSE_Pu = None
        plt.plot(data["t"] / 3600, Punobserved_pred, "--", color="black", label="Predicted Gains")
        plt.xlabel("Time (h)")
        plt.ylabel("Watts (W)")
        plt.legend()
        plt.grid(True)
        plt.title(f"Unobserved Gains (RMSE: {RMSE_Pu})")
        # plt.savefig(
        #     log_dir / f"Punobserved_predictions{f'_{tag}' if tag else ''}.png",
        #     format="png",
        # )
        plt.close()
    plt.plot(data["t"] / 3600, T_in_true, label="Measured Temperature", color="black", alpha=0.4)
    plt.plot(data["t"] / 3600, T_in_sim, "--", color="black", label="Simulated Indoor Temperature")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    squared_prediction_error = (T_in_true - T_in_sim) ** 2
    mse = np.mean(squared_prediction_error)
    RMSE_T_in_sim = round(np.sqrt(mse), 2)
    plt.title(f"Indoor Temperature Simulation (RMSE: {RMSE_T_in_sim})")
    plt.savefig(
        log_dir / f"RC_predictions_Tin{f'_{tag}' if tag else ''}.png"
        if not is_GT
        else log_dir / "RC_predictions_Tin_GT.png",
        format="png",
    )
    plt.close()
    if T_mass_true is not None:
        plt.plot(data["t"] / 3600, T_mass_true, label="Measured Mass Temperature", color="black", alpha=0.4)
        RMSE_T_mass_sim = round(np.sqrt(np.mean((T_mass_true - T_mass_sim) ** 2)), 2)
    else:
        RMSE_T_mass_sim = None
    plt.plot(data["t"] / 3600, T_mass_sim, "--", color="black", label="Simulated Mass Temperature")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Mass Temperature Simulation (RMSE: {RMSE_T_mass_sim})")
    # plt.savefig(
    #     log_dir / f"RC_predictions_Tmass{f'_{tag}' if tag else ''}.png"
    #     if not is_GT
    #     else log_dir / "RC_predictions_Tmass_GT.png",
    #     format="png",
    # )
    plt.close()
    return mse.item()


def plot_data_IDAICE(
    data: pd.DataFrame,
    data_resampled: pd.DataFrame | None = None,
    path: Path | None = None,
    add_tag: str | None = None,
) -> None:
    fig, axs = plt.subplots(5, 1, sharex=True)
    if data_resampled is not None:
        plot_timeseries(
            axs[0],
            data.datetime,
            data["T_in_true"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_in_true"],
            ylabel="T_in ($^{\\circ}$C)",
            label="Noisy",
        )
        plot_timeseries(
            axs[1],
            data.datetime,
            data["T_amb"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_amb"],
            ylabel="T_out ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[2],
            data.datetime,
            data["Ps"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Ps"],
            ylabel="$P_{sun}$ (W/m$^{2}$)",
        )
        # plot_timeseries(
        #     axs[4],
        #     data.datetime,
        #     data["Ps_effective"],
        #     x_sampled=data_resampled.datetime,
        #     y_sampled=data_resampled["Ps"],
        #     ylabel="$P_{s_eff}$ (W/m$^{2}$)",
        # )
        plot_timeseries(
            axs[3],
            data.datetime,
            data["Ph"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Ph"],
            ylabel="$P_{heat}$ (W)",
        )
        plot_timeseries(
            axs[4],
            data.datetime,
            data["Pc"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Pc"],
            ylabel="$P_{cool}$ (W)",
        )
    else:
        plot_timeseries(
            axs[0],
            data.datetime,
            data["T_in_true"],
            ylabel="T_in ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[1],
            data.datetime,
            data["T_amb"],
            ylabel="T_out ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[2],
            data.datetime,
            data["Ps"],
            ylabel="$P_{sun}$ (W/m$^{2}$)",
        )
        # plot_timeseries(
        #     axs[4],
        #     data.datetime,
        #     data["Ps_effective"],
        #     ylabel="$P_{s_eff}$ (W/m$^{2}$)",
        # )
        plot_timeseries(
            axs[3],
            data.datetime,
            data["Ph"],
            ylabel="$P_{heat}$ (W)",
        )
        plot_timeseries(
            axs[4],
            data.datetime,
            data["Pc"],
            ylabel="$P_{cool}$ (W)",
        )
    plt.setp(axs[4].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    if path:
        plt.savefig(
            path / f"data_{add_tag}.png",
            format="png",
        )
        plt.close()
    else:
        plt.show()


def plot_data(
    data: pd.DataFrame,
    data_resampled: pd.DataFrame | None = None,
    path: Path | None = None,
    add_tag: str | None = None,
) -> None:
    fig, axs = plt.subplots(7, 1, sharex=True)
    if data_resampled is not None:
        plot_timeseries(
            axs[0],
            data.datetime,
            data["T_in_true"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_in_true"],
            ylabel="T_in ($^{\\circ}$C)",
            label="Noisy",
        )
        plot_timeseries(
            axs[0],
            data.datetime,
            data["T_in_true_GT"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_in_true_GT"],
            ylabel="T_in ($^{\\circ}$C)",
            label="Ground Truth",
        )
        plot_timeseries(
            axs[1],
            data.datetime,
            data["T_mass_true"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_mass_true"],
            ylabel="T_mass ($^{\\circ}$C)",
            label="Noisy",
        )
        plot_timeseries(
            axs[1],
            data.datetime,
            data["T_mass_true_GT"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_mass_true_GT"],
            ylabel="T_mass ($^{\\circ}$C)",
            label="Ground Truth",
        )
        plot_timeseries(
            axs[2],
            data.datetime,
            data["T_amb"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["T_amb"],
            ylabel="T_out ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[3],
            data.datetime,
            data["Ps"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Ps"],
            ylabel="$P_{sun}$ (W/m$^{2}$)",
        )
        # plot_timeseries(
        #     axs[4],
        #     data.datetime,
        #     data["Ps_effective"],
        #     x_sampled=data_resampled.datetime,
        #     y_sampled=data_resampled["Ps"],
        #     ylabel="$P_{s_eff}$ (W/m$^{2}$)",
        # )
        plot_timeseries(
            axs[4],
            data.datetime,
            data["Ph"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Ph"],
            ylabel="$P_{heat}$ (W)",
        )
        plot_timeseries(
            axs[5],
            data.datetime,
            data["Pc"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Pc"],
            ylabel="$P_{cool}$ (W)",
        )
        plot_timeseries(
            axs[6],
            data.datetime,
            data["Punobserved"],
            x_sampled=data_resampled.datetime,
            y_sampled=data_resampled["Punobserved"],
            ylabel="$P_{unobserved}$ (W)",
        )
    else:
        plot_timeseries(
            axs[0],
            data.datetime,
            data["T_in_true"],
            ylabel="T_in ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[1],
            data.datetime,
            data["T_mass_true"],
            ylabel="T_mass ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[2],
            data.datetime,
            data["T_amb"],
            ylabel="T_out ($^{\\circ}$C)",
        )
        plot_timeseries(
            axs[3],
            data.datetime,
            data["Ps"],
            ylabel="$P_{sun}$ (W/m$^{2}$)",
        )
        # plot_timeseries(
        #     axs[4],
        #     data.datetime,
        #     data["Ps_effective"],
        #     ylabel="$P_{s_eff}$ (W/m$^{2}$)",
        # )
        plot_timeseries(
            axs[4],
            data.datetime,
            data["Ph"],
            ylabel="$P_{heat}$ (W)",
        )
        plot_timeseries(
            axs[5],
            data.datetime,
            data["Pc"],
            ylabel="$P_{cool}$ (W)",
        )
        plot_timeseries(
            axs[6],
            data.datetime,
            data["Punobserved"],
            ylabel="$P_{unobserved}$ (W)",
        )
    plt.setp(axs[6].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    if path:
        plt.savefig(
            path / f"data_{add_tag}.png",
            format="png",
        )
        plt.close()
    else:
        plt.show()


def plot_idaice_data(
    data: pd.DataFrame, path: Path | None = None, add_tag: str | None = None, solar_features: bool = False
) -> None:
    if solar_features:
        wanted = [
            ("T_in_true", r"T_in ($^{\circ}$C)"),
            ("T_amb", r"T_out ($^{\circ}$C)"),
            ("PsS", r"$P_{sun}$ S (W/m$^{2}$)"),
            ("PsN", r"$P_{sun}$ N (W/m$^{2}$)"),
            ("PsE", r"$P_{sun}$ E (W/m$^{2}$)"),
            ("PsW", r"$P_{sun}$ W (W/m$^{2}$)"),
            ("Ph", r"$P_{heat}$ (W)"),
            ("Pc", r"$P_{cool}$ (W)"),
        ]
    else:
        wanted = [
            ("T_in_true", r"T_in ($^{\circ}$C)"),
            ("T_amb", r"T_out ($^{\circ}$C)"),
            ("Ps", r"$P_{sun}$ (W/m$^{2}$)"),
            ("Ph", r"$P_{heat}$ (W)"),
            ("Pc", r"$P_{cool}$ (W)"),
        ]
    to_plot = [(c, yl) for c, yl in wanted if c in data.columns]

    fig, axs = plt.subplots(len(to_plot), 1, sharex=True)
    if len(to_plot) == 1:
        axs = [axs]

    for ax, (col, ylabel) in zip(axs, to_plot):
        plot_timeseries(ax, data.datetime, data[col], ylabel=ylabel)

    plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if path:
        plt.savefig(path / f"data_{add_tag}.png", format="png")
        plt.close()
    else:
        plt.show()


def simulate_2R2C_with_thermostat(
    GT_RCparams: dict[str, float],
    data: pd.DataFrame,
    add_Punobserved: bool,
    # --- heating
    setpoint: float = 22.0,  # heat ON below this
    delta: float = 2.0,  # heat OFF at setpoint + delta
    power_on: float = 2000.0,  # W
    active_hours: tuple = (5, 21),  # 05:00–21:00
    # --- cooling
    cool_setpoint: float = 24.0,  # cool ON above this
    cool_delta: float = 2.0,  # cool OFF at cool_setpoint - cool_delta
    power_on_cool: float = 2000.0,  # W
    active_hours_cool: tuple = (9, 22),  # 09:00–22:00
    # --- season gating: never heat in summer, never cool in winter
    winter_months: tuple = (10, 11, 12, 1, 2, 3, 4, 5),
    summer_months: tuple = (6, 7, 8, 9),
    Ph_provided: np.ndarray | None = None,
    Pc_provided: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    C_in, C_m = GT_RCparams["C_in"], GT_RCparams["C_mass"]
    R_ia, R_im = GT_RCparams["R_ia"], GT_RCparams["R_im"]
    alpha = GT_RCparams["alpha"]  # SHGC * m^2
    x0 = np.array([GT_RCparams.get("t0_in", setpoint), GT_RCparams.get("t0_m", setpoint)], dtype=float)

    N = data.shape[0]
    X = np.zeros((N + 1, 2), dtype=float)
    Ph = np.zeros(N, dtype=float)
    Pc = np.zeros(N, dtype=float)
    X[0] = x0

    relative_Phc_gain_air = GT_RCparams["relative_Phc_gain_air"]
    relative_Ps_gain_air = GT_RCparams["relative_Ps_gain_air"]
    relative_Pu_gain_air = GT_RCparams["relative_Pu_gain_air"]

    if add_Punobserved:
        unobs_cfg = UnobsGainConfig()
        Punobserved = generate_unobserved_gains(data=data, cfg=unobs_cfg)
    else:
        Punobserved = np.zeros(N, dtype=float)

    A = np.array(
        [
            [-(1.0 / (R_ia * C_in) + 1.0 / (R_im * C_in)), 1.0 / (R_im * C_in)],
            [1.0 / (R_im * C_m), -(1.0 / (R_im * C_m))],
        ]
    )

    B = np.array(
        [
            [
                relative_Phc_gain_air / C_in,
                (relative_Ps_gain_air * alpha) / C_in,
                1.0 / (R_ia * C_in),
                relative_Pu_gain_air / C_in,
            ],  # Phc  Ps  T_amb  Punobserved
            [
                (1.0 - relative_Phc_gain_air) / C_m,
                (1.0 - relative_Ps_gain_air) / C_m,
                0.0,
                (1.0 - relative_Pu_gain_air) / C_m,
            ],
        ]
    )

    T_amb = data["T_amb"].to_numpy(float)
    Ps = data["Ps"].to_numpy(float)

    dt = (data["datetime"].iloc[1] - data["datetime"].iloc[0]).total_seconds()
    A_d, B_d, *_ = cont2discrete((A, B, np.eye(2), np.zeros((2, 4))), dt=dt)

    # ---- thermostat states ----
    heater_on = False
    cooler_on = False

    T_heat_on, T_heat_off = setpoint - delta, setpoint + delta
    T_cool_on, T_cool_off = cool_setpoint + delta, cool_setpoint - cool_delta
    h_start, h_end = active_hours
    c_start, c_end = active_hours_cool

    # ---- open-loop path ----
    if (Ph_provided is not None) or (Pc_provided is not None):
        if Ph_provided is None:
            Ph_provided = np.zeros(N, dtype=float)
        if Pc_provided is None:
            Pc_provided = np.zeros(N, dtype=float)
        for k in range(N):
            Phc = float(Ph_provided[k] - Pc_provided[k])
            u_k = np.array([Phc, Ps[k], T_amb[k], Punobserved[k]], dtype=float)
            X[k + 1] = A_d @ X[k] + B_d @ u_k
        return X, Ph_provided, Pc_provided, Punobserved, Ps

    # ---- closed-loop path ----
    for k in range(N):
        ts = data["datetime"].iloc[k]
        hour, month = ts.hour, ts.month

        heat_season = month not in summer_months  # never heat in summer
        cool_season = month not in winter_months  # never cool in winter

        heat_sched = h_start <= hour < h_end
        heat_sched_next = h_start <= hour + 1 < h_end
        cool_sched = c_start <= hour < c_end
        cool_sched_next = c_start <= hour + 1 < c_end

        # Heating decision
        if heat_season and heat_sched:
            if (not heater_on) and (X[k, 0] < T_heat_on) and heat_sched_next:
                heater_on = True
            elif heater_on and (X[k, 0] >= T_heat_off):
                heater_on = False
        else:
            heater_on = False

        # Cooling decision
        if cool_season and cool_sched:
            if (not cooler_on) and (X[k, 0] > T_cool_on) and cool_sched_next:
                cooler_on = True
            elif cooler_on and (X[k, 0] <= T_cool_off):
                cooler_on = False
        else:
            cooler_on = False

        Ph[k] = power_on if heater_on else 0.0
        Pc[k] = power_on_cool if cooler_on else 0.0

        Phc = float(Ph[k] - Pc[k])
        u_k = np.array([Phc, Ps[k], T_amb[k], Punobserved[k]], dtype=float)
        X[k + 1] = A_d @ X[k] + B_d @ u_k

    return X, Ph, Pc, Punobserved, Ps


def _simulate_2R2C(
    params: dict["str", float],
    t: np.ndarray,
    Ph: np.ndarray,
    T_amb: np.ndarray,
    Ps: np.ndarray,
    Pc: np.ndarray | None = None,
    Pu: np.ndarray | None = None,
) -> np.ndarray:
    if Pu is None:
        Pu = np.zeros_like(Ps)
    if Pc is None:
        Pc = np.zeros_like(Ps)
    # assemble data for possible resampling
    data = pd.DataFrame(
        data={
            "t": t,
            "T_amb": T_amb,
            "Ph": Ph,
            "Pc": Pc,
            "Ps": Ps,
            "Pu": Pu,
        }
    )
    dt_data = int(np.mean(np.diff(t)))
    data_resampled = None

    # resample to make zoh more accurate if time steps are greater than 5s
    if dt_data > 10:
        # print("Linear interpolating to get smaller time steps for more accurate integration.")
        data_resampled = resample_simulation_data(data=data, dt_new=10)
        ## debugging reesampling of binary signal
        # plt.scatter(data["t"], data["Ph"], color="gray", label="dt=900")
        # plt.scatter(data_resampled["t"], data_resampled["Ph"], color="black", marker="x", alpha=0.8, label="dt=10")
        # plt.legend()
        # plt.show()

    if data_resampled is not None:
        u_sequence = np.column_stack(
            (
                data_resampled["Ph"] - data_resampled["Pc"],
                data_resampled["Ps"],
                data_resampled["T_amb"],
                data_resampled["Pu"],
            )
        )
        dt = np.mean(np.diff(data_resampled["t"].to_numpy()))
    else:
        u_sequence = np.column_stack((data["Ph"] - data["Pc"], data["Ps"], data["T_amb"], data["Pu"]))  # shape: (N, 3)
        dt = np.mean(np.diff(t))

    C_in, C_m = params["C_in"], params["C_mass"]
    R_ia, R_im = params["R_ia"], params["R_im"]
    alpha = params["alpha"]  # SHGC*square_meters
    x0 = np.array(
        [
            params.get("t0_in", 23.0),
            params.get("t0_m", 23.0),
        ]
    )
    relative_Phc_gain_air = params["relative_Pihc_gain_air"]
    relative_Ps_gain_air = params["relative_Pis_gain_air"]
    relative_Pu_gain_air = params["relative_Piu_gain_air"]

    A = np.array(
        [
            [-(1.0 / (R_ia * C_in) + 1.0 / (R_im * C_in)), 1.0 / (R_im * C_in)],
            [1.0 / (R_im * C_m), -(1.0 / (R_im * C_m))],
        ]
    )

    B = np.array(
        [
            [
                relative_Phc_gain_air / C_in,
                (relative_Ps_gain_air * alpha) / C_in,
                1.0 / (R_ia * C_in),
                relative_Pu_gain_air / C_in,
            ],  # Phc  Ps  T_amb  Punobserved
            [
                (1.0 - relative_Phc_gain_air) / C_m,
                (1.0 - relative_Ps_gain_air) / C_m,
                0.0,
                (1.0 - relative_Pu_gain_air) / C_m,
            ],
        ]
    )

    n = A.shape[0]
    N = u_sequence.shape[0]

    # Discretize the system
    (
        A_d,
        B_d,
        *_,
    ) = cont2discrete((A, B, np.eye(2), np.zeros((2, 4))), dt=dt)

    # Simulate
    X = np.zeros((N + 1, n))
    X[0] = x0

    for k in range(N):
        X[k + 1] = A_d @ X[k] + B_d @ u_sequence[k]

    # sample the result back to og sampling rate
    if data_resampled is not None:
        resampled_result = resample_simulation_data(
            data=pd.DataFrame(data={"t": data_resampled["t"], "Tin": X[:-1, 0], "Tmass": X[:-1, 1]}), dt_new=dt_data
        )
        return resampled_result[["Tin", "Tmass"]].to_numpy()
    else:
        return X[:-1, :]


def simulate_2R2C(
    params: dict[str, float],
    t: np.ndarray,
    Ph: np.ndarray,
    T_amb: np.ndarray,
    Ps: np.ndarray,
    Pc: np.ndarray | None = None,
    foh_cols: Tuple[int, ...] = (1, 2),  # [Phc, Ps, T_amb]
    edge_eps: float = 1e-1,
) -> np.ndarray:
    if Pc is None:
        Pc = np.zeros_like(Ph)

    U = np.column_stack([Ph - Pc, Ps, T_amb])  # shape [N,4]
    N = len(t)
    dt = float(t[1] - t[0])

    C_in, C_m = params["C_in"], params["C_mass"]
    R_ia, R_im = params["R_ia"], params["R_im"]
    alpha = params["alpha"]
    x0 = np.array([params.get("t0_in", 21.0), params.get("t0_m", 21.0)], dtype=float)
    rel_Ph_air = params["relative_Pihc_gain_air"]
    rel_Ps_air = params["relative_Pis_gain_air"]

    A = np.array(
        [
            [-(1.0 / (R_ia * C_in) + 1.0 / (R_im * C_in)), 1.0 / (R_im * C_in)],
            [1.0 / (R_im * C_m), -1.0 / (R_im * C_m)],
        ],
        dtype=float,
    )

    # [P_net, Ps, T_amb]
    B = np.array(
        [
            [rel_Ph_air / C_in, (rel_Ps_air * alpha) / C_in, 1.0 / (R_ia * C_in)],
            [(1.0 - rel_Ph_air) / C_m, (1.0 - rel_Ps_air) / C_m, 0.0],
        ],
        dtype=float,
    )

    n = A.shape[0]
    I = np.eye(n)

    def Ad_G0_G1(dt: float):
        """Return Ad, G0 = ∫0^Δ e^{As}ds, G1 = ∫0^Δ s e^{As}ds"""
        Ad = expm(A * dt)
        G0 = solve(A, Ad - I, assume_a="gen")  # A^{-1}(Ad - I)
        # G1 via two solves: A^{-2}[ e^{AΔ}(AΔ - I) + I ]
        RHS = Ad @ (A * dt - I) + I
        Y = solve(A, RHS, assume_a="gen")
        G1 = solve(A, Y, assume_a="gen")
        return Ad, G0, G1

    # Precompute column views
    b_net = B[:, 0:1]  # [n,1]
    zoh_cols = tuple(i for i in range(B.shape[1]) if i != 0 and i not in foh_cols)

    # State rollout
    X = np.zeros((N, n), dtype=float)
    X[0] = x0

    for k in range(N - 1):
        uk = U[k]  # [4]
        uk1 = U[k + 1]  # [4]
        Ad, G0, G1 = Ad_G0_G1(dt)

        # Start with pure state propagation
        x_next = Ad @ X[k]

        if len(zoh_cols) > 0:
            idx_z = np.array(zoh_cols, dtype=int)
            B_z = B[:, idx_z]  # (n, pz)
            u_z = uk[idx_z].reshape(-1, 1)  # (pz, 1)
            x_next = x_next + ((G0 @ B_z) @ u_z).ravel()

        if len(foh_cols) > 0:
            idx_f = np.array(foh_cols, dtype=int)
            B_f = B[:, idx_f]  # (n, pf)
            u_k = uk[idx_f].reshape(-1, 1)  # (pf, 1)
            u_k1 = uk1[idx_f].reshape(-1, 1)  # (pf, 1)
            B1 = (G1 @ B_f) / dt
            B0 = (G0 @ B_f) - B1
            x_next = x_next + (B0 @ u_k + B1 @ u_k1).ravel()

        # Base early contribution:
        x_next = x_next + (G0 @ b_net).ravel() * uk[0]
        # If edge between k and k+1, add correction with tau
        delta0 = uk1[0] - uk[0]
        if abs(delta0) > edge_eps:
            tau = 0.5 * dt
            if tau > 0.0:
                Gτ = solve(A, expm(A * tau) - I, assume_a="gen")
            else:
                Gτ = np.zeros_like(G0)
            x_next = x_next + ((G0 - Gτ) @ b_net).ravel() * delta0

        X[k + 1] = x_next

    return X


def _season_id(ts) -> int:
    m = ts.month
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[m]


def build_windows_by_season_indices(
    data: pd.DataFrame, window_days: int = 10, stride_days: int = 2
) -> List[Tuple[slice, int, float]]:
    t_np = data["t"].to_numpy()
    dt = float(np.median(np.diff(t_np)))
    T_win = int(round(window_days * 24 * 3600 / dt))
    T_stride = int(round(stride_days * 24 * 3600 / dt))

    def normalize_diff(series):
        diffs = np.diff(series)
        if np.std(series) == 0:
            return np.zeros_like(diffs)
        # Weighting by variation helps balance Watts vs DegC
        return np.abs(diffs) / (np.std(series) + 1e-6)

    Ph_score = normalize_diff(data["Ph"].to_numpy())
    Pc_score = normalize_diff(data["Pc"].to_numpy())

    # Calculate aggregate solar (assuming simple weighted sum for 'informativeness' check)
    Ps_agg = data[["PsN", "PsE", "PsS", "PsW"]] @ np.array([0.25, 0.25, 0.25, 0.25])
    Ps_score = normalize_diff(Ps_agg.to_numpy())

    Tin_score = normalize_diff(data["T_in_true"].to_numpy())

    out: List[Tuple[slice, int, float]] = []
    N = len(t_np)

    # Pre-calculate rolling sums could be faster, but loop is fine for small N
    for s in range(0, N - T_win + 1, T_stride):
        e = s + T_win

        # Majority-vote season
        counts = [0, 0, 0, 0]
        for i in range(s, e):
            counts[_season_id(data["datetime"].iloc[i])] += 1
        season = int(np.argmax(counts))

        # We sum the normalized variations
        score = np.sum(Ph_score[s:e]) + np.sum(Pc_score[s:e]) + np.sum(Ps_score[s:e]) + np.sum(Tin_score[s:e]) * 3.0

        out.append((slice(s, e + 1), season, float(score)))

    return out


def pick_top_per_season(
    slices_scored: List[Tuple[slice, int, float]],
    t_seconds: pd.Series | np.ndarray,
    k_per_season: int = 3,
    min_gap_days: float = 0.0,
) -> List[slice]:
    """Greedy, non-overlapping top-K per season."""
    t_np = t_seconds.to_numpy() if hasattr(t_seconds, "to_numpy") else np.asarray(t_seconds)
    dt = float(np.median(np.diff(t_np)))
    gap = int(round(min_gap_days * 24 * 3600 / dt))

    chosen: List[slice] = []
    for S in (0, 1, 2, 3):
        cand = [(sc, sl) for sl, s_id, sc in slices_scored if s_id == S]
        cand.sort(reverse=True)  # by score
        keep, used = [], []
        for score, sl in cand:
            s, e = sl.start, sl.stop
            if any(not (e + gap <= us or s >= ue + gap) for (us, ue) in used):
                continue
            used.append((s, e))
            keep.append(sl)
            if len(keep) == k_per_season:
                break
        chosen.extend(keep)
    chosen.sort(key=lambda sl: sl.start)
    return chosen


def _read_yaml(path: str) -> Tuple[int, float]:
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    # assume identical structure; raise if not present
    try:
        n_ts = doc["configs"]["n_timesteps"]
        rmse = doc["results"]
    except Exception as e:
        raise ValueError(f"YAML at {path} does not have expected keys: {e}")
    try:
        n_ts = int(n_ts)
        rmse = float(rmse)
    except Exception as e:
        raise ValueError(f"YAML at {path} has non-convertible values: {e}")
    return n_ts, rmse


def aggregate_rmse_by_timesteps(root_dir: str) -> Dict[int, List[float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".yaml", ".yml")):
                p = os.path.join(dirpath, fn)
                n_ts, rmse = _read_yaml(p)
                grouped[n_ts].append(rmse)

    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def plot_rmse_by_days(
    rmse_dict: Dict[int, List[float]], ax: Optional[plt.Axes] = None, save_path: Optional[str] = None
) -> None:
    if not rmse_dict:
        raise ValueError("rmse_dict is empty.")

    xs, ys = [], []
    for n_ts, vals in rmse_dict.items():
        x_days = n_ts / 96.0
        xs.extend([x_days] * len(vals))
        ys.extend(vals)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        created_fig = True
    else:
        fig = ax.figure

    xs_np, ys_np = np.array(xs), np.array(ys)
    jitter_amp = max(1e-6, (xs_np.max() - xs_np.min())) * 0.002
    rng = np.random.default_rng(seed=42)
    jitter = (rng.random(len(xs_np)) - 0.5) * 2 * jitter_amp
    ax.scatter(xs_np + jitter, ys_np, alpha=0.75, s=30, label="individual models")

    n_ts_sorted = sorted(rmse_dict.keys())
    x_group = np.array([n / 96.0 for n in n_ts_sorted])
    y_mean = np.array([np.mean(rmse_dict[n]) for n in n_ts_sorted])
    y_std = np.array([np.std(rmse_dict[n]) for n in n_ts_sorted])
    ax.errorbar(x_group, y_mean, yerr=y_std, fmt="-o", linewidth=2, capsize=4, label="mean ± std")

    ax.set_xlabel("Training window length (days)")
    ax.set_ylabel("RMSE")
    ax.set_title("Model accuracy vs training window length")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)


def plot_weights_evolution(learned_weights: dict, true_weights: dict | None = None, path: str | None = None):
    for name, weights_list in learned_weights.items():
        weights = np.array(weights_list)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(weights))),
                y=weights,
                mode="lines",
                name=f"{name} (estimated)",
                line={"color": "blue"},
            )
        )

        if true_weights is not None:
            true_val = true_weights.get(name, None)
            if true_val is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[0, len(weights) - 1],
                        y=[true_val, true_val],
                        mode="lines",
                        name=f"{name} (true)",
                        line={"color": "red", "dash": "dash"},
                    )
                )
                lower_bound = min(true_val * 0.5, true_val * 1.5)
                upper_bound = max(true_val * 0.5, true_val * 1.5)
                y_range = [lower_bound, upper_bound]
            else:
                y_range = None
        else:
            y_range = None

        fig.update_layout(
            title=f"Evolution of {name} Across Training Steps",
            xaxis_title="Training Step",
            yaxis_title=f"{name} Value",
            template="plotly_white",
            width=900,
            height=500,
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
            # yaxis_range=y_range,
        )

        if path is not None:
            filename = os.path.join(path, f"{name}_evolution.html")
            fig.write_html(filename)
        else:
            fig.show()


def plot_all_weights_evolution(learned_weights: dict, true_weights: dict | None = None, path: str | None = None):
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i, (name, weights_list) in enumerate(learned_weights.items()):
        weights = np.array(weights_list)
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(weights))),
                y=weights,
                mode="lines",
                name=f"{name}",
                line={"color": color},
                legendgroup=name,
            )
        )

        if true_weights is not None:
            true_val = true_weights.get(name)
            if true_val is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[0, len(weights) - 1],
                        y=[true_val, true_val],
                        mode="lines",
                        name=f"{name} (true)",
                        line={"color": color, "dash": "dash"},
                        legendgroup=name,
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title="Evolution of Parameters Across Training Steps",
        xaxis_title="Training Step",
        yaxis_title="Parameter Value",
        yaxis_type="log",
        template="plotly_white",
        width=900,
        height=600,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        yaxis={
            "type": "log",
            "exponentformat": "e",
            "showexponent": "all",
            "dtick": 2,
        },
    )

    if path is not None:
        filename = os.path.join(path, "all_params_evolution.html")
        fig.write_html(filename)
        # pdf_filename = os.path.join(path, "all_params_evolution.pdf")
        # fig.write_image(pdf_filename)
    else:
        fig.show()


def plot_timeseries(
    ax,
    x: pd.Series,
    y: pd.Series,
    ylabel: str = "",
    label: str | None = None,
    x_sampled: pd.Series | None = None,
    y_sampled: pd.Series | None = None,
):
    if label:
        ax.plot(x, y, label=label)
        if isinstance(x_sampled, pd.Series):
            ax.scatter(x_sampled, y_sampled, alpha=0.3)
        ax.legend()
    else:
        ax.plot(x, y)
        if isinstance(x_sampled, pd.Series):
            ax.scatter(x_sampled, y_sampled, alpha=0.3)

    ax.set_ylabel(ylabel)


def plot_fourier_features(
    n_frequencies: int, f_min: float = 0.5 / 86400, f_max: float = 1 / 3600, log_spacing: bool = True
) -> None:
    if log_spacing:
        freqs = f_min * 2.0 ** torch.arange(n_frequencies)
    else:
        freqs = torch.linspace(f_min, f_max, n_frequencies)
    t = np.linspace(0, 2 * 86400, 10000)  # in seconds

    for f in freqs:
        plt.plot(t / 3600, np.sin(2 * np.pi * f * t))
    plt.title("Sine part of Fourier features")
    plt.xlabel("time [h]")
    plt.ylabel("value")
    # plt.legend()
    plt.xlim(0, 48)
    plt.show()

    for f in freqs:
        plt.plot(t / 3600, np.cos(2 * np.pi * f * t))
    plt.title("Cosine part of Fourier features")
    plt.xlabel("time [h]")
    plt.ylabel("value")
    # plt.legend()
    plt.xlim(0, 48)
    plt.show()


def _create_log_dir(model: nn.Module, base_dir: str = "logs", tag: Optional[str] = None):
    model_name = model.__class__.__name__

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    parts = [model_name]
    if tag:
        parts.append(str(tag))
    parts.append(timestamp)

    log_dir = Path(base_dir) / "_".join(parts)
    log_dir.mkdir(parents=True, exist_ok=False)

    return log_dir


def create_log_dir(model: nn.Module, base_dir: str = "logs", tag: Optional[str] = None):
    model_name = model.__class__.__name__

    structure = [model.n_hidden_layers, model.neurons_per_layer]
    struct_str = "-".join(str(x) for x in structure)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    parts = [model_name, struct_str]
    if tag:
        parts.append("tag_" + str(tag))
    parts.append(timestamp)

    log_dir = Path(base_dir) / "_".join(parts)
    log_dir.mkdir(parents=True, exist_ok=False)

    return log_dir


def load_model(initialized_model: nn.Module, ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path)
    initialized_model.load_state_dict(ckpt["model"])
    return initialized_model
