import csv
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from scipy import stats
import seaborn as sns
import yaml
import re
import matplotlib.dates as mdates

from utils import get_IDAICE_data, simulate_2R2C

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
        "figure.figsize": (10, 6),
        "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage{bm} \usepackage{siunitx}",
    }
)


def setup_directories(
    path_to_log_dir: str, model_dir: Path, save_dir_name: str, n_prediction_days: int, evaluation_mode: str
) -> Tuple[str, str, str]:
    """Creates the folder structure for results."""
    base_path = os.path.join(
        path_to_log_dir,
        save_dir_name,
        model_dir.parent.name,
        model_dir.name,
        f"{n_prediction_days}days",
        evaluation_mode,
    )

    if os.path.exists(base_path):
        raise FileExistsError(f"Directory already exists: {base_path}")

    os.makedirs(base_path, exist_ok=False)
    plot_path = os.path.join(base_path, "plots")
    os.makedirs(plot_path, exist_ok=True)
    predictions_path = os.path.join(plot_path, "predictions")
    os.makedirs(predictions_path, exist_ok=True)

    return base_path, plot_path, predictions_path


# Model Loading
def load_ensemble_models(model_dir: Path, plot_path: str) -> Tuple[List[Dict], List[Dict], List[str], float]:
    """Loads all valid models from the directory and extracts parameters with UNIQUE names."""
    model_paths = [p for p in model_dir.iterdir() if (p / "saved_model.pth").exists() and not p.name.startswith("_")]
    print(f"Found {len(model_paths)} models in {model_dir}")

    RCs, solar_shares, model_names = [], [], []
    dt = None

    # Dictionary to track name frequency for uniqueness
    name_counts = {}

    for idx, path_to_model in enumerate(model_paths):
        checkpoint = torch.load(path_to_model / "saved_model.pth")

        # Extract params
        RCs.append(checkpoint["building_params"]["learned RCs"].copy())

        shares = checkpoint["building_params"]["solar shares"]
        solar_shares.append({k: shares[k] for k in ["North", "East", "South", "West"]})

        RCs[idx]["relative_Pihc_gain_air"] = checkpoint["building_params"]["relative_Phc_gain_air"]
        RCs[idx]["relative_Pis_gain_air"] = checkpoint["building_params"]["relative_Ps_gain_air"]

        dt = checkpoint["scales"]["dt"]
        t_min_train = checkpoint["scales"]["t_min_train"]

        base_name = pd.to_datetime(t_min_train, unit="s", utc=True).month_name()[:3]

        if base_name in name_counts:
            name_counts[base_name] += 1
            unique_name = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 0
            unique_name = base_name

        model_names.append(unique_name)

    _plot_rc_distribution(RCs, plot_path)

    return RCs, solar_shares, model_names, dt


def _plot_rc_distribution(RCs: List[Dict], plot_path: str):
    rc_df = pd.DataFrame(RCs)

    plt.style.use(style="default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 16,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.5,
            "figure.figsize": (10, 6),
            "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage{siunitx}",
        }
    )

    cols_to_exclude = ["relative_Pihc_gain_air", "relative_Pis_gain_air"]
    rc_df = rc_df.drop(columns=[c for c in cols_to_exclude if c in rc_df.columns], errors="ignore")

    rename_map = {"C_in": "Cin", "R_ia": "Ria", "C_mass": "Cmass", "R_im": "Rim", "alpha": r"$\alpha$"}
    rc_df = rc_df.rename(columns=rename_map)

    rc_df.boxplot(
        color="navy",  # "#B22222",
        flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
    )
    plt.yscale("log")
    # plt.xticks(ha="right")
    plt.ylabel("Value")
    plt.title("Distribution of Learned RC Parameters Across Models")
    plt.grid(axis="y", linestyle="--", alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "RC_params.png"), format="png")
    plt.close()


def _load_rcs_from_ensemble_dir(model_dir: str | Path) -> List[Dict]:
    """
    Load only the learned RC parameter dicts from an ensemble directory.
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    model_paths = sorted(
        [
            p
            for p in model_dir.iterdir()
            if p.is_dir() and (p / "saved_model.pth").exists() and not p.name.startswith("_")
        ],
        key=lambda p: p.name,
    )

    if not model_paths:
        raise FileNotFoundError(f"No valid ensemble members found in: {model_dir}")

    RCs = []
    for path_to_model in model_paths:
        checkpoint = torch.load(path_to_model / "saved_model.pth", map_location="cpu")

        rc = checkpoint["building_params"]["learned RCs"].copy()

        rc["relative_Pihc_gain_air"] = checkpoint["building_params"].get("relative_Phc_gain_air", 1.0)
        rc["relative_Pis_gain_air"] = checkpoint["building_params"].get("relative_Ps_gain_air", 1.0)

        RCs.append(rc)

    return RCs


def _prepare_rc_frames(model_rcs: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    cols_to_exclude = ["relative_Pihc_gain_air", "relative_Pis_gain_air"]
    rename_map = {
        "C_in": "Cin",
        "R_ia": "Ria",
        "C_mass": "Cmass",
        "R_im": "Rim",
        "alpha": r"$\alpha$",
    }

    cleaned = {}
    for model_name, rcs in model_rcs.items():
        df = pd.DataFrame(rcs)
        df = df.drop(columns=[c for c in cols_to_exclude if c in df.columns], errors="ignore")
        df = df.rename(columns=rename_map)
        cleaned[model_name] = df

    return cleaned


def _write_rc_parameter_summary_csv(
    model_rcs: Dict[str, List[Dict]],
    plot_path: str | Path,
    filename: str = "RC_params_summary.csv",
) -> pd.DataFrame:
    """
    Write a CSV with mean and standard error for each parameter and model.
    """
    cleaned = _prepare_rc_frames(model_rcs)
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    param_names = ["Cin", "Ria", "Cmass", "Rim", r"$\alpha$"]
    model_names = list(cleaned.keys())

    rows = []
    for param in param_names:
        row = {"parameter": param}
        for model_name in model_names:
            df = cleaned[model_name]

            if param not in df.columns:
                row[f"{model_name}_mean"] = np.nan
                row[f"{model_name}_se"] = np.nan
                row[f"{model_name}_n"] = 0
                continue

            values = df[param].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[f"{model_name}_mean"] = np.nan
                row[f"{model_name}_se"] = np.nan
                row[f"{model_name}_n"] = 0
                continue

            mean_val = float(np.mean(values))
            se_val = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

            row[f"{model_name}_mean"] = mean_val
            row[f"{model_name}_se"] = se_val
            row[f"{model_name}_n"] = int(len(values))

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(plot_path / filename, index=False)
    return stats_df


def plot_two_model_rc_parameter_comparison(
    model_dir_1: str | Path,
    model_dir_2: str | Path,
    plot_path: str | Path,
    model_name_1: str = "Inverse PINN",
    model_name_2: str = "Differentiable SSM",
    plot_filename: str = "RC_params_comparison.pdf",
    csv_filename: str = "RC_params_summary.csv",
    overlay_params: Optional[Dict[str, Dict[str, float]]] | None = None,
) -> pd.DataFrame:
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    RCs_1 = _load_rcs_from_ensemble_dir(model_dir_1)
    RCs_2 = _load_rcs_from_ensemble_dir(model_dir_2)

    model_rcs = {
        model_name_1: RCs_1,
        model_name_2: RCs_2,
    }

    _plot_rc_parameter_comparison(
        model_rcs=model_rcs,
        plot_path=str(plot_path),
        filename=plot_filename,
        overlay_params=overlay_params,
    )

    stats_df = _write_rc_parameter_summary_csv(
        model_rcs=model_rcs,
        plot_path=plot_path,
        filename=csv_filename,
    )

    return stats_df


def _plot_rc_parameter_comparison(
    model_rcs: Dict[str, List[Dict]],
    plot_path: str,
    filename: str = "RC_params_comparison.pdf",
    overlay_params: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Compare RC parameter distributions across models using boxplots,
    optionally overlaying identified parameter points.
    """

    if not model_rcs:
        raise ValueError("model_rcs is empty.")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 16,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.5,
            "figure.figsize": (22, 5),
            "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage{siunitx}",
        }
    )

    cols_to_exclude = ["relative_Pihc_gain_air", "relative_Pis_gain_air"]
    rename_map = {
        "C_in": r"$C_\mathrm{in}$",
        "R_ia": r"$R_\mathrm{ia}$",
        "C_mass": r"$C_\mathrm{m}$",
        "R_im": r"$R_\mathrm{im}$",
        "alpha": r"$\alpha$",
    }

    param_names = [r"$C_\mathrm{m}$", r"$C_\mathrm{in}$", r"$R_\mathrm{im}$", r"$R_\mathrm{ia}$", r"$\alpha$"]

    cleaned = {}
    for model_name, rcs in model_rcs.items():
        df = pd.DataFrame(rcs)
        df = df.drop(columns=[c for c in cols_to_exclude if c in df.columns], errors="ignore")
        df = df.rename(columns=rename_map)
        cleaned[model_name] = df

    model_names = list(cleaned.keys())

    fig, axes = plt.subplots(1, len(param_names), squeeze=False)
    axes = axes.flatten()

    flierprops = {
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 4,
    }

    median_color_map = {
        "Inverse PINN": "navy",
        "Differentiable SSM": "firebrick",
    }

    for i, param in enumerate(param_names):
        ax = axes[i]

        data = []
        labels = []

        for model_name in model_names:
            df = cleaned[model_name]
            if param in df.columns:
                values = df[param].dropna().values
                if len(values) > 0:
                    data.append(values)
                    labels.append(model_name)

        if not data:
            ax.set_visible(False)
            continue

        positions = np.arange(1, len(data) + 1)

        bp = ax.boxplot(
            data,
            positions=positions,
            labels=labels,
            patch_artist=False,
            flierprops=flierprops,
            widths=0.6,
        )

        if overlay_params is not None:
            for pos, label in zip(positions, labels):
                if label in overlay_params and param in overlay_params[label]:
                    val = overlay_params[label][param]

                    ax.scatter(
                        pos,
                        val,
                        color="black",
                        marker="D",
                        s=60,
                        zorder=3,
                        label=None,
                    )

        for median_line, label in zip(bp["medians"], labels):
            median_line.set_linewidth(1.5)
            median_line.set_color(median_color_map[label])

        ax.set_title(param)
        ax.set_yscale("log")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle="--", alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, filename), format="pdf", bbox_inches="tight")
    plt.close()


# Simulation & Window Processing
def run_simulation_loop(
    data: pd.DataFrame,
    RCs: List[Dict],
    solar_shares: List[Dict],
    model_names: List[str],
    dt: float,
    n_prediction_days: int,
    prediction_horizon: int,
    evaluation_mode: str,
    predictions_path: str,
    stride: int = 1,
):
    """Main loop: simulates windows, updates metrics, and generates window plots."""

    # Initialize containers
    metrics_tracker = _init_metrics_tracker(model_names)
    hourly_tracker = _init_hourly_tracker(model_names)  # New tracker for daytime plots
    aggregated_results = {"residuals": [], "predicted": [], "sun": [], "hc": [], "tamb": []}

    start_idx = 0
    plot_counter = 0
    prediction_seconds = n_prediction_days * prediction_horizon * 24 * 3600
    n_steps = int(prediction_seconds / dt)

    while start_idx < data.shape[0] - n_steps:
        end_idx = start_idx + n_steps
        window_data = data[start_idx:end_idx]

        # Get hour of day for this window
        window_hours = window_data["datetime"].dt.hour.values

        # Simulate all models for this window
        T_true = window_data["T_in_true"].values
        T_sim_stack = _simulate_window_models(window_data, RCs, solar_shares, T_true[0])

        # Calculate Aggregates (Mean, Min, Max)
        T_sim_mean = T_sim_stack.mean(axis=0)
        T_sim_min = T_sim_stack.min(axis=0)
        T_sim_max = T_sim_stack.max(axis=0)

        # Update Metrics
        season = "Heating" if window_data["Ph"].sum() >= window_data["Pc"].sum() else "Cooling"

        # Baseline
        base_res = T_true - T_true.mean()
        _update_metrics(metrics_tracker, "Baseline", season, base_res)
        _update_hourly_tracker(hourly_tracker, "Baseline", base_res, window_hours)

        # Ensemble
        ens_res = T_true - T_sim_mean
        _update_metrics(metrics_tracker, "Ensemble", season, ens_res)
        _update_hourly_tracker(hourly_tracker, "Ensemble", ens_res, window_hours)

        # Individuals
        for i, name in enumerate(model_names):
            ind_res = T_true - T_sim_stack[i]
            _update_metrics(metrics_tracker, name, season, ind_res)
            _update_hourly_tracker(hourly_tracker, name, ind_res, window_hours)

        # Store Results for Residual Plots & Evaluation
        _store_window_results(aggregated_results, window_data, T_true, T_sim_stack, T_sim_mean, evaluation_mode)

        # Plot Window (only if ensemble mode, usually)
        if evaluation_mode == "ensemble" and plot_counter == 15:
            rmse = round(np.sqrt(np.mean((T_true - T_sim_mean) ** 2)), 4)
            _plot_window_prediction(
                window_data, T_true, T_sim_mean, T_sim_stack, T_sim_min, T_sim_max, rmse, predictions_path
            )
            plot_counter = 0

        # Step forward
        plot_counter += 1
        start_idx += int((stride * 24 * 3600) / dt)

    return metrics_tracker, aggregated_results, hourly_tracker


def _simulate_window_models(data, RCs, solar_shares, t0):
    T_sims = []
    dir_irrad = data[["PsN", "PsE", "PsS", "PsW"]].to_numpy()
    t_arr = data["t"].to_numpy()
    tamb = data["T_amb"].to_numpy()
    ph = data["Ph"].to_numpy()
    pc = data["Pc"].to_numpy()

    for params, shares in zip(RCs, solar_shares):
        params["t0_in"] = t0
        params["t0_m"] = t0
        Ps_eff = dir_irrad @ np.array(list(shares.values()))

        X = simulate_2R2C(params=params, t=t_arr, T_amb=tamb, Ps=Ps_eff, Ph=ph, Pc=pc)
        T_sims.append(X[:, 0])

    return np.stack(T_sims, axis=0)


def _init_metrics_tracker(model_names):
    tracker = {
        name: {"Heating": {"sse": [], "sae": [], "n": []}, "Cooling": {"sse": [], "sae": [], "n": []}}
        for name in model_names
    }
    for extra in ["Ensemble", "Baseline"]:
        tracker[extra] = {"Heating": {"sse": [], "sae": [], "n": []}, "Cooling": {"sse": [], "sae": [], "n": []}}
    return tracker


def _init_hourly_tracker(model_names):
    # Initializes counters for 0-23 hours
    all_models = model_names + ["Ensemble", "Baseline"]
    return {name: {h: {"abs_sum": 0.0, "count": 0} for h in range(24)} for name in all_models}


def _update_metrics(tracker, name, season, residuals) -> None:
    tracker[name][season]["sse"].append(np.sum(residuals**2))
    tracker[name][season]["sae"].append(np.sum(np.abs(residuals)))
    tracker[name][season]["n"].append(len(residuals))


def _update_hourly_tracker(tracker, name, residuals, hours) -> None:
    abs_res = np.abs(residuals)
    # Aggregate by hour using bincount for speed or simple loop
    # Since hours are 0-23, we can iterate simple 24 bins
    for h in range(24):
        mask = hours == h
        if np.any(mask):
            tracker[name][h]["abs_sum"] += float(np.sum(abs_res[mask]))
            tracker[name][h]["count"] += int(np.sum(mask))


def _store_window_results(agg, data, T_true, T_stack, T_mean, mode):
    sun = data[["PsN", "PsE", "PsS", "PsW"]].sum(axis=1).values
    hc = (data["Ph"] - data["Pc"]).values
    tamb = data["T_amb"].values

    if mode == "ensemble":
        res = T_true - T_mean
        agg["residuals"].append(res)
        agg["predicted"].append(T_mean)
        agg["sun"].append(sun)
        agg["hc"].append(hc)
        agg["tamb"].append(tamb)
    elif mode == "individual":
        res = T_true - T_stack
        agg["residuals"].append(res.flatten())
        agg["predicted"].append(T_stack.flatten())
        n_models = T_stack.shape[0]
        agg["sun"].append(np.tile(sun, n_models))
        agg["hc"].append(np.tile(hc, n_models))
        agg["tamb"].append(np.tile(tamb, n_models))


def _plot_window_prediction(data, true, mean, stack, min_v, max_v, rmse, path):
    t_h = data["t"] / 3600
    plt.figure()
    plt.plot(t_h, true, label=r"True $T_\text{in}$", color="red", alpha=0.8)
    plt.plot(t_h, mean, "--", label="Mean Pred", color="black")
    for t_sim in stack:
        plt.plot(t_h, t_sim, "--", alpha=0.08, color="gray")
    plt.fill_between(t_h, min_v, max_v, color="gray", alpha=0.1, label="Uncertainty")
    # plt.plot(t_h, np.full_like(true, true[0]), ":", label="Baseline", color="blue", alpha=0.5)

    plt.xlabel("Time (h)")
    plt.ylabel("Temp [°C]")
    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.title(f"Prediction (RMSE: {rmse})")

    fname = f"Pred_{data['datetime'].min()}_{data['datetime'].max()}".replace(":", "-")
    plt.savefig(os.path.join(path, fname + ".png"), format="png")
    plt.close()


# Metric Aggregation & Final Plots
def compute_global_metrics(metrics_tracker: Dict, model_names: List[str]) -> Dict:
    """Aggregates window metrics into Global RMSE/MAE."""
    models = ["Baseline"] + model_names + ["Ensemble"]
    summary = {}

    for mod in models:
        summary[mod] = {}
        for scope, seasons in [("All", ["Heating", "Cooling"]), ("Heat", ["Heating"]), ("Cool", ["Cooling"])]:
            t_sse, t_sae, t_n = 0, 0, 0
            for s in seasons:
                d = metrics_tracker[mod][s]
                t_sse += sum(d["sse"])
                t_sae += sum(d["sae"])
                t_n += sum(d["n"])

            summary[mod][f"{scope}_RMSE"] = np.sqrt(t_sse / t_n) if t_n > 0 else 0
            summary[mod][f"{scope}_MAE"] = t_sae / t_n if t_n > 0 else 0

    return summary


def plot_metric_comparison(summary: Dict, model_names: List[str], plot_path: str):
    models = model_names + ["Ensemble"]

    for metric in ["RMSE", "MAE"]:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(models))
        width = 0.25

        y_all = [summary[m][f"All_{metric}"] for m in models]
        y_heat = [summary[m][f"Heat_{metric}"] for m in models]
        y_cool = [summary[m][f"Cool_{metric}"] for m in models]

        ax.bar(x - width, y_all, width, label="Whole Year", color="grey")
        ax.bar(x, y_heat, width, label="Heating", color="red", alpha=0.7)
        ax.bar(x + width, y_cool, width, label="Cooling", color="blue", alpha=0.7)

        ax.set_ylabel(metric)
        ax.set_title(f"Global {metric} by Model and Season")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m} Model" for m in models], rotation=45)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"Model_Comparison_{metric}.png"), dpi=300)
        plt.close()


def save_residual_plots(
    all_residuals: List,
    all_predicted: List,
    all_sun_power: List,
    all_hc_power: List,
    all_tamb: List,
    plot_path: Path,
):
    plot_path.mkdir(parents=True, exist_ok=True)

    if len(all_residuals) > 0 and isinstance(all_residuals[0], (np.ndarray, list)):
        flat_residuals = np.concatenate([np.atleast_1d(x) for x in all_residuals])
        flat_predicted = np.concatenate([np.atleast_1d(x) for x in all_predicted])
        flat_sun = np.concatenate([np.atleast_1d(x) for x in all_sun_power])
        flat_hc = np.concatenate([np.atleast_1d(x) for x in all_hc_power])
        flat_tamb = np.concatenate([np.atleast_1d(x) for x in all_tamb])
    else:
        flat_residuals = np.array(all_residuals)
        flat_predicted = np.array(all_predicted)
        flat_sun = np.array(all_sun_power)
        flat_hc = np.array(all_hc_power)
        flat_tamb = np.array(all_tamb)

    mean = float(np.mean(flat_residuals))
    std = float(np.std(flat_residuals, ddof=1))
    mae = float(np.mean(np.abs(flat_residuals)))
    rmse = np.sqrt(np.mean(np.square(flat_residuals)))

    # 1. Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(flat_residuals, bins=50, density=True, alpha=0.9)
    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(np.min(flat_residuals), np.max(flat_residuals), 500)
        kde = gaussian_kde(flat_residuals)
        plt.plot(xs, kde(xs), linewidth=1.5)
    except Exception:
        pass
    stats_text = f"N={flat_residuals.size}\nmean={mean:.3g}\nstd={std:.3g}\nMAE={mae:.3g}\nRMSE={rmse:.3g}"
    plt.text(
        0.98,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.1},
    )
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title("Residuals: Histogram")
    plt.grid(True, linestyle="--", alpha=0.1)
    plt.savefig(plot_path / "residuals_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Boxplot
    plt.figure(figsize=(7, 3))
    plt.boxplot(flat_residuals, vert=False, widths=0.6)
    plt.xlabel("Residual")
    plt.title("Residuals: Boxplot")
    plt.grid(True, axis="x", linestyle="--", alpha=0.1)
    plt.savefig(plot_path / "residuals_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Residuals vs Predicted
    plt.figure(figsize=(7, 5))
    # Downsample if too large for scatter
    if len(flat_predicted) > 10000:
        idx = np.random.choice(len(flat_predicted), 10000, replace=False)
        plt.scatter(flat_predicted[idx], flat_residuals[idx], s=8, alpha=0.5)
    else:
        plt.scatter(flat_predicted, flat_residuals, s=8, alpha=0.5)
    plt.axhline(0, linestyle="--", linewidth=1, color="k")
    plt.xlabel("Predicted $T_\text{in}$ (°C)")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")

    # Median Trend
    order = np.argsort(flat_predicted)
    xp = flat_predicted[order]
    rp = flat_residuals[order]
    # Simple moving average or median for trend line to avoid filter errors on small data
    window = max(5, int(len(rp) / 50))
    if window % 2 == 0:
        window += 1
    if len(rp) > window:
        med = median_filter(rp, size=window)
        # Plot only subset to save time if large
        plt.plot(xp[::100], med[::100], linewidth=1.2, color="orange", label="Median Trend")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.1)
    plt.savefig(plot_path / "residuals_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. QQ Plot
    plt.figure(figsize=(5, 5))
    stats.probplot(flat_residuals, dist="norm", plot=plt)
    plt.title("QQ-plot vs Normal")
    plt.savefig(plot_path / "residuals_qq.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Scatter plots
    def _plot_scatter(x, y, xlabel, name):
        plt.figure(figsize=(7, 5))
        if len(x) > 10000:
            idx = np.random.choice(len(x), 10000, replace=False)
            plt.scatter(x[idx], y[idx], s=8, alpha=0.5)
        else:
            plt.scatter(x, y, s=8, alpha=0.5)
        plt.axhline(0, linestyle="--", linewidth=1, color="k")
        plt.xlabel(xlabel)
        plt.ylabel("Residual")
        plt.title(f"Residuals vs {name}")
        plt.grid(True, linestyle="--", alpha=0.1)
        plt.savefig(
            plot_path / f"residuals_vs_{name.lower().replace(' ', '_').replace('/', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    _plot_scatter(flat_sun, flat_residuals, "Total Sun Power [W/m²]", "Sun Power")
    _plot_scatter(flat_hc, flat_residuals, "Net Heating/Cooling Power (Ph - Pc) [W]", "Heating Cooling Power")
    _plot_scatter(flat_tamb, flat_residuals, "Outdoor Temperature ($T_{amb}$) [°C]", "Tamb")


def evaluate_prediction(
    data: pd.DataFrame,
    model_dir: Path,
    save_dir_name: str,
    n_prediction_days: int,
    evaluation_mode: str,
    path_to_log_dir: str = "results/",
    prediction_horizon: int = 1,
) -> None:
    # Setup
    path, plot_path, pred_path = setup_directories(
        path_to_log_dir, model_dir, save_dir_name, n_prediction_days, evaluation_mode
    )

    # Load Models
    RCs, solar_shares, model_names, dt = load_ensemble_models(model_dir, plot_path)

    # Run Simulation Loop
    metrics_tracker, results_data, hourly_tracker = run_simulation_loop(
        data, RCs, solar_shares, model_names, dt, n_prediction_days, prediction_horizon, evaluation_mode, pred_path
    )

    # Compute & Plot Global Metrics
    metrics_summary = compute_global_metrics(metrics_tracker, model_names)
    plot_metric_comparison(metrics_summary, model_names, plot_path)

    # Save Residual Analysis Plots
    save_residual_plots(
        all_residuals=results_data["residuals"],
        all_predicted=results_data["predicted"],
        all_sun_power=results_data["sun"],
        all_hc_power=results_data["hc"],
        all_tamb=results_data["tamb"],
        plot_path=Path(plot_path),
    )

    # Save Final YAML
    print(f"Saved results to {path}")
    model_names = []
    final_results = {}

    for model_name, metrics in metrics_summary.items():
        rmse_key = "All_RMSE"
        mae_key = "All_MAE"
        if model_name in model_names:
            model_name += "1"
        if rmse_key in metrics:
            final_results[f"{model_name}_RMSE"] = round(float(metrics[rmse_key]), 4)
        if mae_key in metrics:
            final_results[f"{model_name}_MAE"] = round(float(metrics[mae_key]), 4)
        model_names.append(model_name)

    with open(Path(path) / "results.yaml", "w") as f:
        yaml.dump(final_results, f, default_flow_style=False)

    # Save Hourly (Daytime) Error Profiles
    # Convert to pure MAE list [0..23] for saving
    hourly_export = {}
    for name, hrs in hourly_tracker.items():
        mae_profile = []
        for h in range(24):
            if hrs[h]["count"] > 0:
                mae_profile.append(float(hrs[h]["abs_sum"] / hrs[h]["count"]))
            else:
                mae_profile.append(0.0)
        hourly_export[name] = mae_profile

    with open(Path(path) / "daytime_profiles.json", "w") as f:
        json.dump(hourly_export, f, indent=4)


def plot_multi_model_error_comparison(
    experiment_group_name: Path,
    search_dir: str = "Evaluate",
    path_to_log_dir: str = "results/",
    filter_n_days: int | None = None,
    max_plotted_prediction_horizon: int = 10,
    metric: str = "MAE",
    save_format: str = "png",
    hide_specific_model_name: bool = False,
) -> None:
    """
    Plots the error metric (MAE/RMSE) of the ENSEMBLE vs. Horizon.
    Also plots the BASELINE for comparison.
    Skips models starting with '_'.
    """

    log_path = Path(path_to_log_dir)
    experiment_group_name = Path(experiment_group_name)
    metric = metric.upper()

    if metric not in ["RMSE", "MAE"]:
        print(f"Error: Metric '{metric}' not supported.")
        return

    results_root = log_path / search_dir / experiment_group_name

    # Define keys for Ensemble and Baseline
    metric_key_map = {
        "RMSE": ("Ensemble_RMSE", "Baseline_RMSE"),
        "MAE": ("Ensemble_MAE", "Baseline_MAE"),
    }
    ens_key, base_key = metric_key_map[metric]

    if not results_root.exists():
        print(f"Error: Directory not found: {results_root}")
        return

    print(f"Scanning for results in: {results_root}")

    extracted_data = {}  # {model_name: {horizon: value}}
    baseline_data = {}  # {horizon: [values_from_different_models]}

    horizon_pattern = re.compile(r"^(\d+)days$")

    yaml_files = list(results_root.rglob("results.yaml"))

    if not yaml_files:
        print("No results.yaml files found.")
        return

    for yaml_file in yaml_files:
        try:
            path_parts = yaml_file.parts
            horizon = None
            model_name = None

            # Identify Horizon and Model Name from path
            for i, part in enumerate(reversed(path_parts)):
                match = horizon_pattern.match(part)
                if match:
                    horizon = int(match.group(1))
                    model_name_idx = len(path_parts) - 1 - i - 1
                    model_name = path_parts[model_name_idx]
                    break

            if horizon is None or model_name is None:
                continue

            # NEW: Skip models starting with underscore
            if model_name.startswith("_"):
                continue

            if filter_n_days is not None:
                pattern = re.compile(rf"(?<!\d){filter_n_days}days")

                if not pattern.search(model_name):
                    continue

            with open(yaml_file, "r") as f:
                res = yaml.safe_load(f)

                # Extract Ensemble Value
                ens_val = res.get(ens_key)
                if ens_val is not None:
                    if model_name not in extracted_data:
                        extracted_data[model_name] = {}
                    extracted_data[model_name][horizon] = ens_val

                # Extract Baseline Value
                base_val = res.get(base_key)
                if base_val is not None:
                    if horizon not in baseline_data:
                        baseline_data[horizon] = []
                    baseline_data[horizon].append(base_val)

        except Exception as e:
            print(f"Skipping {yaml_file}: {e}")

    if not extracted_data:
        print(f"No data found matching keys for {metric}.")
        return

    summary_path = results_root / "Summary"
    summary_path.mkdir(exist_ok=True)

    def _custom_sort_key(name):
        days_match = re.search(r"(\d+)days", name)
        if days_match:
            days = int(days_match.group(1))
            prefix = name[: days_match.start()]
        else:
            days = 999
            prefix = name
        return (days, prefix)

    sorted_models = sorted(extracted_data.keys(), key=_custom_sort_key)
    if len(sorted_models) <= 2:
        cmap = ["navy", "firebrick"]
    else:
        cmap = cm.viridis(np.linspace(0, 0.9, len(sorted_models)))

    plt.figure()

    # Plot Ensemble Models
    for idx, model_name in enumerate(sorted_models):
        data_points = extracted_data[model_name]
        horizons = sorted(data_points.keys())

        # Filter horizons if max_plotted_prediction_horizon is set
        if max_plotted_prediction_horizon is not None:
            horizons = [h for h in horizons if h <= max_plotted_prediction_horizon]

        # Skip if no data points remain after filtering
        if not horizons:
            continue

        values = [data_points[h] for h in horizons]

        safe_label = model_name.replace("_", r"\_")
        if hide_specific_model_name:
            if safe_label.startswith("D"):
                safe_label = "DSSM"
            else:
                safe_label = "PINN"

        plt.plot(
            horizons,
            values,
            marker="o",
            linestyle="-",
            markersize=6,
            label=safe_label,
            color=cmap[idx],
            alpha=0.8,
        )

    plt.xlabel("Prediction Horizon [Days]")
    plt.ylabel(f"Ensemble {metric} ($T_{{\\text{{in}}}}$) [\\si{{\\degreeCelsius}}]")
    # plt.yscale("log")
    plt.grid(True, alpha=0.1)
    if hide_specific_model_name:
        plt.legend(loc="upper left", frameon=False)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Ensemble")

    # Set X-Ticks based on extracted data, respecting the limit
    all_horizons = set()
    for d in extracted_data.values():
        all_horizons.update(d.keys())

    if max_plotted_prediction_horizon is not None:
        all_horizons = {h for h in all_horizons if h <= max_plotted_prediction_horizon}

    if all_horizons:
        plt.xticks(sorted(list(all_horizons)))

    plt.tight_layout()

    filename_suffix = f"_filter{filter_n_days}" if filter_n_days is not None else ""
    limit_suffix = f"_limit{max_plotted_prediction_horizon}" if max_plotted_prediction_horizon is not None else ""

    save_file = (
        summary_path
        / f"{metric}_vs_Horizon_Comparison_{experiment_group_name.name}{filename_suffix}{limit_suffix}.{save_format}"
    )

    plt.savefig(save_file, bbox_inches="tight", format=save_format)
    plt.close()

    print(f"Comparison plot saved to: {save_file}")


def plot_individual_models_horizon(
    experiment_group_name: Path,
    model_config: str,
    search_dir: str = "Evaluate",
    path_to_log_dir: str = "results/",
    metric: str = "MAE",
    save_format: str = "png",
) -> None:
    """
    Plots the error metric of INDIVIDUAL models (and Ensemble/Baseline)
    against the prediction horizon for a SPECIFIC model configuration.

    Args:
        experiment_group_name: The parent folder of the experiment
        model_config: The specific training configuration folder name (e.g. "7days")
    """

    log_path = Path(path_to_log_dir)
    experiment_group_name = Path(experiment_group_name)
    metric = metric.upper()

    # Path to the specific configuration (e.g. .../Ensemble_Predictions/Exp1/7days)
    config_root = log_path / search_dir / experiment_group_name / model_config

    if not config_root.exists():
        print(f"Error: Model configuration directory not found: {config_root}")
        print(f"Check if '{model_config}' exists inside '{experiment_group_name}'")
        return

    print(f"Scanning {config_root} for horizons...")

    data_store = {}

    horizon_pattern = re.compile(r"^(\d+)days$")

    for horizon_dir in config_root.iterdir():
        if not horizon_dir.is_dir():
            continue

        match = horizon_pattern.match(horizon_dir.name)
        if not match:
            continue

        horizon = int(match.group(1))

        found_yaml = False
        for result_file in horizon_dir.rglob("results.yaml"):
            try:
                with open(result_file, "r") as f:
                    res = yaml.safe_load(f)

                if not res:
                    continue

                # Extract metrics
                for key, value in res.items():
                    if key.endswith(f"_{metric}"):
                        # Extract model name (e.g., "Jan" from "Jan_MAE")
                        model_name = key.replace(f"_{metric}", "")

                        if model_name not in data_store:
                            data_store[model_name] = {}

                        # Store data
                        data_store[model_name][horizon] = value

                found_yaml = True
                # Break after finding the first valid results.yaml to avoid duplicates
                break

            except Exception as e:
                print(f"Warning: Error reading {result_file}: {e}")

        if found_yaml:
            print(f"  Found data for horizon: {horizon} days")

    if not data_store:
        print(f"No data found for metric {metric}. Ensure 'results.yaml' exists and contains individual model keys.")
        return

    plt.figure()
    # Get all unique horizons found
    all_horizons = sorted(list(set(h for m in data_store for h in data_store[m])))

    # Define special styles
    special_styles = {
        "Ensemble": {"color": "navy", "lw": 1.5, "ls": "-", "zorder": 10},
        "Baseline": {"color": "firebrick", "lw": 1.5, "ls": "--", "zorder": 9},
    }

    regular_models = [m for m in data_store.keys() if m not in special_styles]

    months_order = {
        m: i
        for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    }
    regular_models.sort(key=lambda x: (0, months_order[x]) if x in months_order else (1, x))

    cmap = cm.get_cmap("viridis", len(regular_models))

    for idx, model in enumerate(regular_models):
        horizons = sorted(data_store[model].keys())
        values = [data_store[model][h] for h in horizons]

        plt.plot(horizons, values, marker="o", markersize=4, label=model, color=cmap(idx), alpha=0.7, linewidth=1.0)

    for model, style in special_styles.items():
        if model in data_store:
            horizons = sorted(data_store[model].keys())
            values = [data_store[model][h] for h in horizons]

            plt.plot(
                horizons,
                values,
                marker="s",
                markersize=6,
                label=f"\\textbf{{{model}}}",
                color=style["color"],
                linewidth=style["lw"],
                linestyle=style["ls"],
                zorder=style["zorder"],
            )

    plt.xlabel("Prediction Horizon [Days]")
    plt.ylabel(f"{metric} ($T_{{\\text{{in}}}}$) [\\si{{\\degreeCelsius}}]")
    plt.title(f"Individual Model Performance vs Horizon (Model: {model_config})")

    if all_horizons:
        plt.xticks(all_horizons)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Models")
    plt.grid(True, alpha=0.1)
    plt.tight_layout()

    summary_path = config_root / "Summary"
    summary_path.mkdir(exist_ok=True)
    save_path = summary_path / f"Individual_Models_vs_Horizon_{metric}_{model_config}"

    plt.savefig(f"{save_path}.{save_format}", format=save_format, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {save_path}.{save_format}")


def plot_comparative_mae_distributions(
    experiment_group_name: Path,
    configs: List[str],  # Accepts N models
    horizon_days: int = 5,
    metric: str = "MAE",
    search_dir: str = "Evaluate",
    path_to_log_dir: str = "results/",
    save_format: str = "png",
) -> None:
    """
    Compares the distribution of errors (MAE/RMSE) of INDIVIDUAL models across multiple approaches.
    Generates a vertical boxplot comparison (Models on X-axis).
    """

    log_path = Path(path_to_log_dir)
    experiment_group = Path(experiment_group_name)
    metric = metric.upper()

    def get_individual_errors(config_name):
        base_eval_path = log_path / search_dir / experiment_group
        target_path = base_eval_path / config_name / f"{horizon_days}days" / "individual" / "results.yaml"

        # Fallback path logic
        if not target_path.exists():
            target_path = (
                log_path
                / search_dir
                / experiment_group
                / config_name
                / f"{horizon_days}days"
                / "individual"
                / "results.yaml"
            )

        if not target_path.exists():
            print(f"Warning: Could not find individual results.yaml for {config_name}")
            return np.array([])

        with open(target_path, "r") as f:
            res = yaml.safe_load(f)

        errors = []
        for key, val in res.items():
            if key.endswith(f"_{metric}"):
                model_name = key.replace(f"_{metric}", "")
                # Exclude aggregates
                if model_name not in ["Ensemble", "Baseline"] and not model_name.startswith("_"):
                    errors.append(val)
        return np.array(errors)

    plot_data = []
    labels = []
    stats_labels = []

    for cfg in configs:
        data = get_individual_errors(cfg)
        if len(data) > 0:
            plot_data.append(data)
            clean_name = cfg.replace("_", " ")
            if clean_name.startswith("D"):
                clean_name = "Differentiable SSM"
            else:
                if len(configs) > 2:
                    clean_name = clean_name
                else:
                    clean_name = "Inverse PINN"
            labels.append(clean_name)
            stats_labels.append(f"$\\mu={data.mean():.3f}, \\sigma={data.std(ddof=1):.3f}$")

    if not plot_data:
        print(f"Error: No data found for specified configurations.")
        return

    fig, ax = plt.subplots(1, 1)

    if len(plot_data) <= 2:
        colors = ["navy", "firebrick"]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))

    bplot = ax.boxplot(
        plot_data,
        vert=True,
        patch_artist=True,
        labels=labels,
        widths=0.5,
        zorder=5,
        flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "none", "markeredgecolor": "none"},
    )

    # Colorize boxes
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    # Customize medians
    for median in bplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Add Jittered Scatter points
    # Vertical: X is index (jittered), Y is error
    for i, d in enumerate(plot_data):
        y = d
        x = np.random.normal(i + 1, 0.04, size=len(y))

        ax.scatter(x, y, alpha=0.8, color="black", s=15, zorder=10, edgecolors="white", linewidth=0.5)

    # ax.set_title(f"Comparative Distribution of {metric} ({horizon_days} Day{'s' if horizon_days > 1 else ''})")
    ax.set_ylabel(f"{metric} [\\si{{\\degreeCelsius}}]")

    # if len(configs) > 3 or any(len(l) > 10 for l in labels):
    #     plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.1)
    ax.xaxis.grid(False)

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", alpha=0.6, label=l) for i, l in enumerate(stats_labels)
    ]

    ax.legend(handles=legend_handles, loc="best", fontsize=18, frameon=False)

    output_dir = log_path / search_dir / experiment_group / "Comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(configs) > 3:
        config_str = "Multi_Model_Comparison"
    else:
        config_str = "_vs_".join(configs)

    filename = f"Boxplot_{config_str}_{horizon_days}days_{metric}.{save_format}"
    save_path = output_dir / filename
    plt.savefig(save_path, format=save_format, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to: {save_path}")


def plot_expected_MAE_against_prediction_horizon(
    experiment_group_name: Path,
    configs: List[str],
    metric: str = "MAE",
    search_dir: str = "Evaluate_7",
    path_to_log_dir: str = "results/",
) -> None:
    """
    Calculate and plot the expected MAE and corresponding standard error
    for fixed prediction horizons and both model classes.

    Fixed prediction horizons:
        [1, 2, 3, 5, 7] days

    The function:
    1. Scans the result folders of the given configs
    2. Collects individual model errors for the fixed horizons
    3. Computes expected MAE = mean(MAE_i)
    4. Computes standard error SE = s / sqrt(N), with sample std using ddof=1
    5. Plots expected MAE with error bars against prediction horizon
    6. Saves the aggregated results to a CSV file with columns:
       model, horizon, mae, se
    """
    PREDICTION_HORIZONS = (1, 2, 3, 5, 7)

    log_path = Path(path_to_log_dir)
    experiment_group = Path(experiment_group_name)
    metric = metric.upper()
    search_root = log_path / search_dir / experiment_group

    if not search_root.exists():
        print(f"Error: Search directory not found: {search_root}")
        return

    print(f"Scanning {search_root} for expected {metric} calculation...")

    grouped_data = {
        "PINN": {h: [] for h in PREDICTION_HORIZONS},
        "Differentiable SSM": {h: [] for h in PREDICTION_HORIZONS},
    }

    days_pattern = re.compile(r"S?(\d+)days")

    for config in configs:
        model_dir = search_root / config

        if not model_dir.exists():
            print(f"Warning: Config directory not found, skipping: {model_dir}")
            continue

        match = days_pattern.search(config)
        if not match:
            print(f"Warning: Could not parse training days from config: {config}")
            continue

        model_type = "Differentiable SSM" if config.startswith("D") else "PINN"

        for horizon in PREDICTION_HORIZONS:
            result_file = model_dir / f"{horizon}days" / "individual" / "results.yaml"

            if not result_file.exists():
                found = list(model_dir.rglob(f"{horizon}days/individual/results.yaml"))
                result_file = found[0] if found else None

            if result_file is None or not result_file.exists():
                print(f"Warning: No results file found for {config}, horizon {horizon} days")
                continue

            try:
                with open(result_file, "r") as f:
                    res = yaml.safe_load(f)

                if not isinstance(res, dict):
                    print(f"Warning: Unexpected YAML format in {result_file}")
                    continue

                errors = [
                    val
                    for key, val in res.items()
                    if key.endswith(f"_{metric}")
                    and not key.startswith("Ensemble")
                    and not key.startswith("Baseline")
                    and not key.startswith("_")
                    and isinstance(val, (int, float))
                ]

                if errors:
                    grouped_data[model_type][horizon].extend(errors)
                else:
                    print(f"Warning: No matching {metric} values found in {result_file}")

            except Exception as e:
                print(f"Error reading {result_file}: {e}")

    aggregated_rows = []
    plot_data = {}

    for model_type, horizon_map in grouped_data.items():
        valid_horizons = []
        means = []
        sems = []
        mins = []
        maxs = []

        print(f"\nModel: {model_type}")

        for horizon in PREDICTION_HORIZONS:
            values = horizon_map[horizon]

            if len(values) == 0:
                print(f"  Horizon {horizon}d: no data")
                continue

            mean_val = float(np.mean(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))

            if len(values) > 1:
                std_val = float(np.std(values, ddof=1))
                se_val = float(std_val / np.sqrt(len(values)))
            else:
                std_val = 0.0
                se_val = 0.0

            print(f"  Horizon {horizon}d: E[{metric}] = {mean_val:.6f}, SE = {se_val:.6f}, N = {len(values)}")

            aggregated_rows.append(
                {
                    "model": model_type,
                    "horizon": horizon,
                    "mae": mean_val,
                    "se": se_val,
                }
            )

            valid_horizons.append(horizon)
            means.append(mean_val)
            sems.append(se_val)
            mins.append(min_val)
            maxs.append(max_val)

        plot_data[model_type] = {
            "horizons": np.array(valid_horizons, dtype=float),
            "means": np.array(means, dtype=float),
            "sems": np.array(sems, dtype=float),
            "mins": np.array(mins, dtype=float),
            "maxs": np.array(maxs, dtype=float),
        }

    # plt.figure(figsize=(8, 5))

    for model_type, data in plot_data.items():
        if len(data["horizons"]) == 0:
            continue

        color = "navy" if model_type == "PINN" else "firebrick"
        label = "Inverse PINN" if model_type == "PINN" else model_type

        plt.plot(
            data["horizons"],
            data["means"],
            linestyle="-",
            alpha=0.4,
            color=color,
            zorder=2 if label == "Inverse PINN" else 1,
        )

        plt.errorbar(
            data["horizons"],
            data["means"],
            yerr=data["sems"],
            fmt="o",
            capsize=4,
            alpha=1.0,
            color=color,
            label=label,
            zorder=4 if label == "Inverse PINN" else 3,
        )

        # line = plt.plot(
        #     data["horizons"],
        #     data["means"],
        #     marker="o",
        #     linestyle="-",
        #     label=model_type,
        # )[0]

        # plt.fill_between(
        #     data["horizons"],
        #     data["mins"],
        #     data["maxs"],
        #     alpha=0.2,
        #     color=line.get_color(),
        # )

    plt.xlabel("Prediction horizon [days]")
    plt.ylabel(f"Mean {metric} [\\si{{\\degreeCelsius}}]")
    # plt.title(f"Expected {metric} vs prediction horizon")
    plt.xticks(PREDICTION_HORIZONS)
    plt.grid(True, alpha=0.1)
    plt.legend(frameon=False)
    plt.tight_layout()

    plot_path = search_root / "Comparisons" / f"expected_{metric.lower()}_against_prediction_horizon_{configs}.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    # plt.show()

    print(f"\nSaved plot to: {plot_path}")

    csv_path = search_root / "Comparisons" / f"expected_{metric.lower()}_against_prediction_horizon_{configs}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "horizon", "mae", "se"])
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"Saved CSV to: {csv_path}")


def plot_mae_vs_training_data_size(
    experiment_group_name: Path,
    configs: List[str],
    prediction_horizon: int = 1,
    metric: str = "MAE",
    error_type: str = "sem",
    search_dir: str = "Evaluate_TDS_NEW",
    path_to_log_dir: str = "results/",
    save_format: str = "png",
) -> None:
    """
    Plots Mean MAE/RMSE (with error bars) vs. Training Data Volume (Window Length).
    Groups directories by checking if they are 'DSSM' (starting with D) or 'PINN' (others).
    """

    log_path = Path(path_to_log_dir)
    experiment_group = Path(experiment_group_name)
    metric = metric.upper()
    search_root = log_path / search_dir / experiment_group

    if not search_root.exists():
        print(f"Error: Search directory not found: {search_root}")
        return

    print(f"Scanning {search_root} for data volume comparison...")

    grouped_data = {"PINN": {}, "Differentiable SSM": {}}
    days_pattern = re.compile(r"S?(\d+)days")

    for config in configs:
        model_dir = search_root / config

        if not model_dir.exists():
            continue

        # Extract Data Volume (Days)
        match = days_pattern.search(config)
        if not match:
            continue
        days = int(match.group(1))

        # Determine Model Type
        if config.startswith("D"):
            model_type = "Differentiable SSM"
        else:
            model_type = "PINN"

        # Load Results
        result_file = model_dir / f"{prediction_horizon}days" / "individual" / "results.yaml"

        if not result_file.exists():
            found = list(model_dir.rglob(f"{prediction_horizon}days/individual/results.yaml"))
            result_file = found[0] if found else None

        if result_file:
            try:
                with open(result_file, "r") as f:
                    res = yaml.safe_load(f)

                # Collect errors (exclude Ensemble/Baseline)
                errors = [
                    val
                    for key, val in res.items()
                    if key.endswith(f"_{metric}")
                    and not key.startswith("Ensemble")
                    and not key.startswith("Baseline")
                    and not key.startswith("_")
                ]

                if errors:
                    if days not in grouped_data[model_type]:
                        grouped_data[model_type][days] = []
                    grouped_data[model_type][days].extend(errors)
            except Exception as e:
                print(f"Error reading {result_file}: {e}")

    plt.figure()
    markers = ["o", "s", "^", "D"]
    colors = {"PINN": "navy", "Differentiable SSM": "black"}

    has_data = False

    for idx, (label, data_map) in enumerate(grouped_data.items()):
        if not data_map:
            continue

        has_data = True
        sorted_days = sorted(data_map.keys())

        means = np.array([np.mean(data_map[d]) for d in sorted_days])

        if error_type == "minmax":
            mins = np.array([np.min(data_map[d]) for d in sorted_days])
            maxs = np.array([np.max(data_map[d]) for d in sorted_days])
            lower_err = means - mins
            upper_err = maxs - means
            yerr = [lower_err, upper_err]
            capsize = 4

        elif error_type == "sem":
            stds = np.array([np.std(data_map[d], ddof=1) for d in sorted_days])
            ns = np.array([len(data_map[d]) for d in sorted_days])
            sems = stds / np.sqrt(ns)
            yerr = sems
            capsize = 3

        else:
            stds = np.array([np.std(data_map[d]) for d in sorted_days])
            lower_errors = np.minimum(stds, means)  # Clip at 0
            yerr = [lower_errors, stds]
            capsize = 5

        # plt.errorbar(
        #     sorted_days,
        #     means,
        #     yerr=yerr,
        #     label=label,
        #     fmt=f"-{markers[idx % len(markers)]}",
        #     color=colors.get(label, "black"),
        #     capsize=capsize,
        #     linewidth=1.0,
        #     markersize=6,
        #     alpha=0.8 if label == "PINN" else 0.3,
        #     zorder=2 if label == "PINN" else 1,
        # )

        color = "navy" if label == "PINN" else "firebrick"
        label = "Inverse PINN" if label == "PINN" else label

        plt.plot(
            sorted_days,
            means,
            linestyle="-",
            alpha=0.4,
            color=color,
            zorder=3 if label == "Inverse PINN" else 1,
        )

        plt.errorbar(
            sorted_days,
            means,
            yerr=yerr,
            fmt="o",
            capsize=4,
            alpha=1.0,
            color=color,
            label=label,
            zorder=4 if label == "Inverse PINN" else 2,
        )
        # plt.yscale("log")

    if not has_data:
        print("No valid data found to plot.")
        plt.close()
        return

    plt.xlabel("Training Data Size [Days]")
    plt.ylabel(f"Mean {metric} [\\si{{\\degreeCelsius}}]")
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    plt.xticks([2, 4, 7])
    # plt.title(f"Prediction Horizon: {prediction_horizon} Day{'s' if prediction_horizon > 1 else ''}")
    plt.grid(True, alpha=0.1)
    plt.legend(frameon=False)
    plt.tight_layout()

    output_dir = log_path / search_dir / experiment_group / "Comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"Training_Data_Size_{metric}_{prediction_horizon}day_horizon.{save_format}"
    save_path = output_dir / fname

    plt.savefig(save_path, format=save_format, bbox_inches="tight")
    plt.close()

    print(f"Data volume plot saved to: {save_path}")


def plot_prediction_error_vs_daytime(
    experiment_group_name: Path,
    model_config: str,
    models_to_plot: List[str] = ["Ensemble", "Baseline"],
    search_dir: str = "Evaluate",
    path_to_log_dir: str = "results/",
    save_format: str = "png",
) -> None:
    """
    Plots the Mean Absolute Error (MAE) vs Time of Day for the 1-day ahead prediction.

    Args:
        experiment_group_name: The parent experiment folder.
        model_config: The specific training config (e.g. "7days").
        models_to_plot: List of model names to plot (e.g. ["Ensemble", "Baseline", "Jan", "Feb"]).
    """

    log_path = Path(path_to_log_dir)
    experiment_group_name = Path(experiment_group_name)

    # Target 1-day prediction specifically
    target_horizon = "1days"

    # Path: .../Ensemble_Predictions/{experiment}/{config}/1days
    search_path = log_path / search_dir / experiment_group_name / model_config / target_horizon

    if not search_path.exists():
        print(f"Error: 1-day prediction directory not found: {search_path}")
        return

    # Find the daytime_profiles.json
    profile_path = None
    for f_path in search_path.rglob("daytime_profiles.json"):
        profile_path = f_path
        break

    if not profile_path:
        print(f"Error: 'daytime_profiles.json' not found in {search_path}. Run simulation first.")
        return

    with open(profile_path, "r") as f:
        profiles = json.load(f)

    # Plotting
    plt.figure()
    hours = np.arange(24)

    # Styles
    special_styles = {
        "Ensemble": {"color": "navy", "lw": 1.0, "ls": "-", "zorder": 10},
        "Baseline": {"color": "firebrick", "lw": 1.0, "ls": "--", "zorder": 9},
    }

    # Handle regular models color map
    regular_models = [m for m in models_to_plot if m not in special_styles]
    months_order = {
        m: i
        for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    }
    regular_models.sort(key=lambda x: (0, months_order[x]) if x in months_order else (1, x))
    cmap = cm.get_cmap("viridis", len(regular_models)) if regular_models else None

    # Plot requested models
    reg_idx = 0
    for model in models_to_plot:
        if model not in profiles:
            print(f"Warning: Model '{model}' not found in saved profiles. Skipping.")
            continue

        y_vals = profiles[model]

        if model in special_styles:
            s = special_styles[model]
            plt.plot(
                hours,
                y_vals,
                label=model,
                color=s["color"],
                lw=s["lw"],
                ls=s["ls"],
                zorder=s["zorder"],
            )
        else:
            c = cmap(reg_idx)
            plt.plot(hours, y_vals, label=model, color=c, lw=1.5, alpha=0.8)
            reg_idx += 1

    plt.xlabel("Time of Day [Hour]")
    plt.ylabel(r"MAE ($T_{\text{in}}$) [\si{\degreeCelsius}]")
    plt.title(f"Day-Ahead Prediction Error vs Time of Day\n(Config: {model_config})")
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim(0, 23)
    plt.grid(True, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Models")
    plt.tight_layout()

    # Save
    summary_path = log_path / search_dir / experiment_group_name / model_config / "Summary"
    summary_path.mkdir(parents=True, exist_ok=True)

    # Filename includes model names if few, else generic
    if len(models_to_plot) < 5:
        models_str = "_".join(models_to_plot)
    else:
        models_str = "Selected_Models"

    save_file = summary_path / f"Error_vs_Daytime_{models_str}_{model_config}.{save_format}"
    plt.savefig(save_file, format=save_format, bbox_inches="tight")
    plt.close()

    print(f"Daytime error plot saved to: {save_file}")


def _simulate_single_model(data: pd.DataFrame, path_to_model: str | Path) -> np.ndarray:
    path_to_model = Path(path_to_model)
    checkpoint_path = path_to_model / "saved_model.pth" if path_to_model.is_dir() else path_to_model

    if not checkpoint_path.exists():
        # Fallback: maybe the user passed a direct path like "simple_lstm.pth"
        if path_to_model.exists():
            checkpoint_path = path_to_model
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # --- LSTM MODEL ---
    if checkpoint.get("model_type") == "LSTM":
        return _simulate_lstm(data, checkpoint)

    # --- PHYSICS (PINN/2R2C) MODEL ---
    else:
        return _simulate_physics(data, checkpoint)


def _simulate_lstm(data, checkpoint):
    """Auto-regressive inference for LSTM."""
    hp = checkpoint["hyperparams"]

    model = LSTM(input_dim=hp["input_dim"], hidden_dim=hp["hidden_dim"], num_layers=hp.get("num_layers", 2))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mu = checkpoint["scalers"]["mean"]
    std = checkpoint["scalers"]["std"]
    feature_cols = checkpoint.get("feature_cols", ["T_in_true", "T_amb", "Ph", "Pc", "PsN", "PsE", "PsS", "PsW"])

    # Check if we need to de-normalize the output
    target_is_normed = checkpoint.get("is_target_normalized", False)

    df_vals = data[feature_cols].values.astype(np.float32)
    T_initial = df_vals[0, 0]

    predictions = [T_initial]
    current_T = T_initial

    with torch.no_grad():
        hidden = None
        for t in range(len(df_vals) - 1):
            input_row = df_vals[t].copy()
            input_row[0] = current_T

            # Normalize Input
            input_tensor = torch.tensor(input_row)
            input_norm = (input_tensor - mu) / std
            input_norm = input_norm.view(1, 1, -1)

            out, hidden = model(input_norm, hidden)

            if target_is_normed:
                # Output is a Z-score. Convert back to °C.
                # T_in is at index 0
                pred_T = out.item() * std[0].item() + mu[0].item()
            else:
                # Output is raw °C (Legacy support)
                pred_T = out.item()

            predictions.append(pred_T)
            current_T = pred_T

    return np.array(predictions)


def _simulate_physics(data, checkpoint):
    """Original physics simulation logic."""
    params = checkpoint["building_params"]["learned RCs"].copy()

    # Safe retrieval of relative gains
    params["relative_Pihc_gain_air"] = checkpoint["building_params"].get("relative_Phc_gain_air", 1.0)
    params["relative_Pis_gain_air"] = checkpoint["building_params"].get("relative_Ps_gain_air", 1.0)

    raw_shares = checkpoint["building_params"]["solar shares"]
    solar_weights = np.array([raw_shares[k] for k in ["North", "East", "South", "West"]])

    t_arr = data["t"].to_numpy()
    tamb = data["T_amb"].to_numpy()
    ph = data["Ph"].to_numpy()
    pc = data["Pc"].to_numpy()
    dir_irrad = data[["PsN", "PsE", "PsS", "PsW"]].to_numpy()
    ps_eff = dir_irrad @ solar_weights

    t0 = data["T_in_true"].iloc[0]
    params["t0_in"] = t0
    params["t0_m"] = t0

    X = simulate_2R2C(params=params, t=t_arr, T_amb=tamb, Ps=ps_eff, Ph=ph, Pc=pc)
    return X[:, 0]


def plot_model_comparison(
    data: pd.DataFrame,
    pinn_path: str | Path,
    dssm_path: str | Path,
    lstm_path: str | Path,
    save_path: str | Path,
    experiment_group_name: str | Path = "training_w500Tank+-1,0deg",
    save_format: str = "png",
    verbose: bool = False,
):
    """
    Loads two specific model checkpoints and plots their simulated trajectories
    against the ground truth.
    """

    # Simulate both models
    if verbose:
        print("--- Simulating PINN Model ---")
    T_sim_pinn = _simulate_single_model(data, pinn_path)

    if verbose:
        print("--- Simulating DSSM Model ---")
    T_sim_dssm = _simulate_single_model(data, dssm_path)

    if verbose:
        print("--- Simulating LSTM Model ---")
    T_sim_LSTM = _simulate_single_model(data, lstm_path)

    # Ground Truth
    plt.plot(data["datetime"], data["T_in_true"], label=r"True $T_\text{in}$", color="black", alpha=0.5)

    # Model 1
    plt.plot(data["datetime"], T_sim_pinn, label="PINN", linestyle="--", color="navy")

    # Model 2
    plt.plot(data["datetime"], T_sim_dssm, label="DSSM", linestyle="--", color="firebrick")
    plt.plot(data["datetime"], T_sim_LSTM, label="LSTM", linestyle="--", color="#007A00")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    # # Rotate ticks as requested
    # plt.xticks(rotation=45)

    plt.legend(frameon=False)
    plt.ylabel("Temperature [°C]")
    plt.grid(True, linestyle="--", alpha=0.1)
    plt.tight_layout()

    save_file = (
        Path("results")
        / save_path
        / Path(experiment_group_name)
        / Path("Comparisons")
        / f"Trajectory_Comparison.{save_format}"
    )

    plt.savefig(
        save_file,
        format=save_format,
        bbox_inches="tight",
    )
    plt.close()


def plot_physical_plausibility(
    data: pd.DataFrame,
    model_path: str,
    label: str,
    save_path: str | Path,
    experiment_group_name: str | Path = "training_w500Tank+-1,0deg",
    save_format: str = "png",
    verbose: bool = False,
):
    """
    Evaluates physical plausibility by forcing the heating input (Ph) to constant values
    (+3000W and -3000W) and observing the trajectory response.

    Args:
        data: The evaluation dataframe.
        model_path_1: Path to the first model (e.g., Physics/PINN).
        model_path_2: Path to the second model (e.g., LSTM).
    """

    # Create Datasets
    # Scenario A: Constant Max Heating
    data_hot_4kw = data.copy()
    data_hot_4kw["Ph"] = 4000.0
    data_hot_4kw["Pc"] = 0.0

    data_hot_2kw = data.copy()
    data_hot_2kw["Ph"] = 2000.0
    data_hot_2kw["Pc"] = 0.0

    # Scenario B: Constant Max Cooling
    data_cold_4kw = data.copy()
    data_cold_4kw["Ph"] = 0.0
    data_cold_4kw["Pc"] = 4000.0

    data_cold_2kw = data.copy()
    data_cold_2kw["Ph"] = 0.0
    data_cold_2kw["Pc"] = 2000.0

    if verbose:
        print("--- Simulating Scenario: Max Heating ---")
    sim_hot_4kw = _simulate_single_model(data_hot_4kw, model_path)
    sim_hot_2kw = _simulate_single_model(data_hot_2kw, model_path)

    if verbose:
        print("--- Simulating Scenario: Max Cooling ---")
    sim_cold_4kw = _simulate_single_model(data_cold_4kw, model_path)
    sim_cold_2kw = _simulate_single_model(data_cold_2kw, model_path)

    plt.plot(data["datetime"], data["T_in_true"], label=r"True $T_\text{in}$", color="gray", alpha=0.3)
    # plt.fill_between(
    #     data["datetime"], sim_hot_2kw, sim_hot_4kw, color="firebrick", alpha=0.2, label=f"2-4 kW Heating ({label})"
    # )
    plt.plot(data["datetime"], sim_hot_4kw, color="firebrick", label="4kW Heating")
    # plt.plot(data["datetime"], sim_hot_2kw, color="firebrick", alpha=0.6, label="2kW Heating")

    # plt.fill_between(
    #     data["datetime"], sim_cold_2kw, sim_cold_4kw, color="navy", alpha=0.2, label=f"2-4 kW Cooling ({label})"
    # )
    plt.plot(data["datetime"], sim_cold_4kw, color="navy", label="4kW Cooling")
    # plt.plot(data["datetime"], sim_cold_2kw, color="navy", alpha=0.6, label="2kW Cooling")

    plt.ylabel("Temperature [°C]")
    # plt.title(label)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.1)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    # plt.xticks(rotation=45)

    save_file = (
        Path("results")
        / save_path
        / Path(experiment_group_name)
        / Path("Comparisons")
        / f"plausibility_plot_{label}.{save_format}"
    )

    plt.tight_layout()
    plt.savefig(
        save_file,
        format=save_format,
        bbox_inches="tight",
    )
    plt.close()


def plot_physical_plausibility_comparison(
    data: pd.DataFrame,
    model_path_PINN: str,
    model_path_DSSM: str,
    save_path: str | Path,
    experiment_group_name: str | Path = "training_w500Tank+-1,0deg",
    save_format: str = "png",
    verbose: bool = False,
):
    """
    Evaluates physical plausibility by forcing the heating input (Ph) to constant values
    (+3000W and -3000W) and observing the trajectory response.

    Args:
        data: The evaluation dataframe.
        model_path_1: Path to the first model (e.g., Physics/PINN).
        model_path_2: Path to the second model (e.g., LSTM).
    """

    # Create Datasets
    # Scenario A: Constant Max Heating
    data_hot = data.copy()
    data_hot["Ph"] = 4000.0
    data_hot["Pc"] = 0.0

    # Scenario B: Constant Max Cooling
    data_cold = data.copy()
    data_cold["Ph"] = 0.0
    data_cold["Pc"] = 4000.0

    if verbose:
        print("--- Simulating Scenario: Max Heating ---")
    sim_hot_PINN = _simulate_single_model(data_hot, model_path_PINN)
    sim_hot_DSSM = _simulate_single_model(data_hot, model_path_DSSM)

    if verbose:
        print("--- Simulating Scenario: Max Cooling ---")
    sim_cold_PINN = _simulate_single_model(data_cold, model_path_PINN)
    sim_cold_DSSM = _simulate_single_model(data_cold, model_path_DSSM)

    plt.plot(data["datetime"], data["T_in_true"], label=r"True $T_\text{in}$", color="gray", alpha=0.3)
    plt.plot(data["datetime"], sim_hot_PINN, label=r"Max Heat (PINN)", color="firebrick")
    plt.plot(data["datetime"], sim_hot_DSSM, label="Max Heat (DSSM)", color="firebrick", alpha=0.5)

    plt.plot(data["datetime"], sim_cold_PINN, label="Max Cool (PINN)", color="navy")
    plt.plot(data["datetime"], sim_cold_DSSM, label="Max Cool (DSSM)", color="navy", alpha=0.5)
    plt.ylabel("Temperature [°C]")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.1)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    # plt.xticks(rotation=45)

    save_file = (
        Path("results")
        / save_path
        / Path(experiment_group_name)
        / Path("Comparisons")
        / f"comparison_plausibility_plot.{save_format}"
    )

    plt.tight_layout()
    plt.savefig(
        save_file,
        format=save_format,
        bbox_inches="tight",
    )
    plt.close()


def plot_monthly_models_comparison(
    experiment_group_name: Path,
    configs: List[str],
    horizon: int = 1,
    month: str | None = None,
    search_dir: str = "Evaluate_7",
    path_to_log_dir: str = "results/",
    metric: str = "MAE",
    save_format: str = "pdf",
):
    """
    Bar plot:
    X-axis: training window
    Y-axis: full-year MAE
    """

    log_path = Path(path_to_log_dir)
    experiment_group = Path(experiment_group_name)
    metric = metric.upper()

    data = {}

    for cfg in configs:
        result_file = log_path / search_dir / experiment_group / cfg / f"{horizon}days" / "individual" / "results.yaml"

        if not result_file.exists():
            print(f"Warning: {result_file} not found")
            continue

        with open(result_file, "r") as f:
            res = yaml.safe_load(f)

        model_type = "DSSM" if cfg.startswith("D") else "PINN"

        for key, val in res.items():
            if (
                key.endswith(f"_{metric}")
                and not key.startswith("Ensemble")
                and not key.startswith("Baseline")
                and not key.startswith("_")
            ):
                model_name = key.replace(f"_{metric}", "")

                if model_name not in data:
                    data[model_name] = {}

                data[model_name][model_type] = val

    if not data:
        print("No data found.")
        return

    if month is not None:
        month = month[:3].capitalize()

        filtered_data = {}
        for model_name, vals in data.items():
            if model_name == month:
                filtered_data[model_name] = vals

        if not filtered_data:
            print(f"No models found for month '{month}'")
            return

        data = filtered_data

    def sort_key(name):
        month_order = {
            m: i
            for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        }
        base = name.split("_")[0]
        suffix = int(name.split("_")[1]) if "_" in name else 0
        return (month_order.get(base, 99), suffix)

    sorted_models = sorted(data.keys(), key=sort_key)

    x = np.arange(len(sorted_models))
    width = 0.35

    pinn_vals = []
    dssm_vals = []

    for m in sorted_models:
        pinn_vals.append(data[m].get("PINN", np.nan))
        dssm_vals.append(data[m].get("DSSM", np.nan))

    plt.figure()

    plt.bar(
        x - width / 2,
        pinn_vals,
        width,
        label="Inverse PINN",
        color="navy",
    )

    plt.bar(
        x + width / 2,
        dssm_vals,
        width,
        label="Differentiable SSM",
        color="firebrick",
    )

    plt.xticks(x, [m.replace("_", r"\_") for m in sorted_models], rotation=45)
    plt.xlabel("Training Window")
    plt.ylabel(f"{metric} [\\si{{\\degreeCelsius}}]")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False)

    plt.tight_layout()

    save_path = log_path / search_dir / experiment_group / "Comparisons"
    save_path.mkdir(parents=True, exist_ok=True)

    fname = save_path / f"Monthly_Model_PairedBars_{horizon}days_{metric}.{save_format}"
    plt.savefig(fname, format=save_format, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison to: {fname}")


if __name__ == "__main__":
    days = 7
    save_dir_name = f"Evaluate_TDS_NEW"
    model_pinn_5D2P4M = f"{days}days_5D2P4M_1dayStride"
    # model_pinn_5D4P4M = f"{days}days_5D4P4M"
    model_pinn_5D2P4M_6FFs = f"{days}days_5D2P4M_6FFs"
    # model_pinn_1D1P1M = f"{days}days_1D1P1M"
    # model_pinn_5D4P2M = f"{days}days_5D4P2M"
    # model_pinn_S1 = f"S{days}days_4more"
    # model_pinn_S2 = f"S{days}days_NEWLATEST"
    # # model_pinn_S2 = f"S{days}days_8more"
    # model_pinn_S = f"S{days}days"
    model_dssm = f"D{days}days"

    data_eval_sim_res = get_IDAICE_data(
        path_to_data="data/IDAICE/v005___mit 500 Liter Speicher ___+-1,5 deg/data.csv",
        start_date="2019-01-01 03:00:00",
        end_date="2019-12-31 03:00:00",
    )

    # # prediction_horizons = [1, 2, 3, 5, 7]
    # path = Path(f"results/training_w500Tank+-1,0deg/{model_dssm}/")
    # # path = Path(f"results/training_w500Tank+-1,0deg/{model_pinn_5D4P4M}/")
    # for _ in prediction_horizons:
    #     metrics1 = evaluate_prediction(
    #         data=data_eval_sim_res,
    #         model_dir=path,
    #         save_dir_name=save_dir_name,
    #         n_prediction_days=int(_),
    #         evaluation_mode="individual",
    #     )
    #     metrics2 = evaluate_prediction(
    #         data=data_eval_sim_res,
    #         model_dir=path,
    #         save_dir_name=save_dir_name,
    #         n_prediction_days=int(_),
    #         evaluation_mode="ensemble",
    #     )

    # plot_multi_model_error_comparison(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     search_dir=save_dir_name,
    #     filter_n_days=days,
    # )

    # plot_multi_model_error_comparison(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     search_dir=save_dir_name,
    #     hide_specific_model_name=True,
    #     save_format="svg",
    # )

    # load_ensemble_models(model_dir=path, plot_path="results/")

    # plot_individual_models_horizon(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     search_dir=save_dir_name,
    #     model_config=model_pinn_5D2P4M,
    # )
    # plot_individual_models_horizon(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     search_dir=save_dir_name,
    #     model_config=model_dssm,
    # )
    # plot_prediction_error_vs_daytime(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     search_dir=save_dir_name,
    #     model_config=model_pinn1,
    # )
    # plot_prediction_error_vs_daytime(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     search_dir=save_dir_name,
    #     model_config=model_dssm,
    # )

    # plot_expected_MAE_against_prediction_horizon(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     configs=[
    #         # model_pinn_S,
    #         model_pinn_5D2P4M,
    #         model_dssm,
    #     ],
    #     search_dir=save_dir_name,
    #     metric="MAE",
    # )

    # plot_two_model_rc_parameter_comparison(
    #     model_dir_1="results/training_w500Tank+-1,0deg/7days_5D2P4M",
    #     model_dir_2="results/training_w500Tank+-1,0deg/D7days",
    #     plot_path="results/Evaluate_TDS_NEW/training_w500Tank+-1,0deg/Comparisons",
    # )

    # plot_two_model_rc_parameter_comparison(
    #     model_dir_1="results/training_w500Tank+-1,0deg/7days_5D2P4M",
    #     model_dir_2="results/training_w500Tank+-1,0deg/D7days",
    #     plot_path="results/Evaluate_TDS_NEW/training_w500Tank+-1,0deg/Comparisons",
    #     overlay_params={
    #         "Inverse PINN": {
    #             "Cin": 23979552.0,
    #             "Cmass": 632151040.0,
    #             "Ria": 0.014902,
    #             "Rim": 0.000539,
    #             r"$\alpha$": 44.7428,
    #         },
    #         "Differentiable SSM": {
    #             "Cin": 9901470.18,
    #             "Cmass": 57744630.65,
    #             "Ria": 0.009646,
    #             "Rim": 0.000207,
    #             r"$\alpha$": 68.448,
    #         },
    #     },
    # )

    # plot_monthly_models_comparison(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     configs=[
    #         "2days_5D2P4M_1dayStride",
    #         "D2days",
    #     ],
    #     search_dir="Evaluate_TDS_NEW",
    #     horizon=1,
    # )

    # plot_monthly_models_comparison(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     configs=[
    #         "2days_5D2P4M_1dayStride",
    #         "D2days",
    #     ],
    #     search_dir="Evaluate_TDS_NEW",
    #     month="Jan",
    #     horizon=1,
    # )

    # plot_comparative_mae_distributions(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     configs=[model_pinn_5D2P4M, model_dssm],
    #     search_dir=save_dir_name,
    #     horizon_days=1,
    #     metric="MAE",
    #     save_format="pdf",
    # )

    # plot_mae_vs_training_data_size(
    #     experiment_group_name=Path("training_w500Tank+-1,0deg"),
    #     configs=[
    #         f"{2}days_5D2P4M_1dayStride",
    #         f"{4}days_5D2P4M_1dayStride",
    #         f"{7}days_5D2P4M_1dayStride",
    #         # f"{14}days_newFF",
    #         # f"{21}days",
    #         f"D{2}days",
    #         f"D{4}days",
    #         f"D{7}days_1dayStride",
    #         # f"D{14}days",
    #         # f"D{21}days",
    #     ],
    #     prediction_horizon=1,
    #     metric="MAE",
    #     save_format="pdf",
    # )

    # # PLausibility Plots -------------------------------------------------------------------------------------
    # days = 7
    # save_dir_name = f"Evaluate_{days}"

    # data_eval = get_IDAICE_data(
    #     path_to_data="data/IDAICE/v005___A___+-1,0 deg___Winter/data.csv",
    #     start_date="2019-01-28 03:00:00",
    #     end_date="2019-02-03 03:00:00",
    #     solar_features=True,
    # )

    # model_path_PINN = (
    #     "results/training_w500Tank+-1,0deg/7days_5D2P4M/PINN_3-128_tag_75Data_4Mass_2Phys_20260121-040210"
    # )
    # model_path_DSSM = "results/training_w500Tank+-1,0deg/D7days/DifferentiableSSM_7_20260119-190336"
    # model_path_LSTM = "results/training_w500Tank+-1,0deg/LSTM_0109_0809/simple_lstm.pth"
    # plot_model_comparison(
    #     data_eval,
    #     pinn_path=model_path_PINN,
    #     dssm_path=model_path_DSSM,
    #     lstm_path=model_path_LSTM,
    #     save_path=save_dir_name,
    # )

    # data_eval = get_IDAICE_data(
    #     path_to_data="data/IDAICE/v005___A___+-1,0 deg___Winter/data.csv",
    #     start_date="2019-01-28 03:00:00",
    #     end_date="2019-01-29 03:00:00",
    #     solar_features=True,
    # )
    # plot_physical_plausibility(
    #     data_eval,
    #     model_path=model_path_PINN,
    #     save_path=save_dir_name,
    #     label="PINN",
    # )
    # plot_physical_plausibility(
    #     data_eval,
    #     model_path=model_path_DSSM,
    #     save_path=save_dir_name,
    #     label="DSSM",
    # )
    # plot_physical_plausibility(
    #     data_eval,
    #     model_path=model_path_LSTM,
    #     save_path=save_dir_name,
    #     label="LSTM",
    # )
    # plot_physical_plausibility_comparison(
    #     data_eval,
    #     model_path_PINN=model_path_PINN,
    #     model_path_DSSM=model_path_DSSM,
    #     save_path=save_dir_name,
    # )
