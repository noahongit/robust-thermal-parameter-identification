from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml
from utils import build_windows_by_season_indices, pick_top_per_season, plot_batch_data, stack_windows_to_tensors
from DifferentiableSSM import DifferentiableSSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(
    data_train: pd.DataFrame,
    data_eval: pd.DataFrame,
    window_days: int,
    prediction_horizon: int = 1,
    n_training_steps: int = 100_000,
    n_training_steps_pre_train: int = 3000,
    path_to_model: str | None = None,
    tag: str | None = None,
    verbose: bool = False,
) -> Path:
    dt_train = data_train["t"].iloc[1] - data_train["t"].iloc[0]
    assert dt_train == data_eval["t"].iloc[1] - data_eval["t"].iloc[0]

    steps_per_day = int(round(86400 / dt_train))
    window_length_in_dt = steps_per_day * window_days
    stride_in_dt_train = steps_per_day * 1

    prediction_horizon_in_dt = steps_per_day * prediction_horizon
    stride_in_dt_eval = steps_per_day * prediction_horizon  # advance one prediction horizon

    # Train -------------------------------------------------------------------------------------------------------
    t_shared, Ph, Pc, Ps_directional, T_amb, Tin = stack_windows_to_tensors(
        data=data_train,
        window_length_in_dt=window_length_in_dt,
        stride_in_dt=stride_in_dt_train,
        use_solar_features=True,
    )

    # Evaluate -------------------------------------------------------------------------------------------------------
    t_shared_eval, Ph_eval, Pc_eval, Ps_directional_eval, T_amb_eval, Tin_eval = stack_windows_to_tensors(
        data=data_eval,
        window_length_in_dt=prediction_horizon_in_dt,
        stride_in_dt=stride_in_dt_eval,
        use_solar_features=True,
    )
    model = DifferentiableSSM(tag=tag).to(dtype=torch.float64)

    if verbose:
        print("\nTrain model...")

    batch = model.prepare_batch(
        t=t_shared,
        Ph=Ph,
        T_amb=T_amb,
        T_in_true=Tin,
        Ps=Ps_directional,
        Pc=Pc,
        is_eval=False,
    )
    model._train_ADAM(batch=batch, n_training_steps=n_training_steps_pre_train, path_to_model=path_to_model)
    model._train_LBFGS(batch=batch, n_training_steps=n_training_steps, path_to_model=path_to_model)

    if verbose:
        print("\nEvaluate model...")

    batch_eval = model.prepare_batch(
        t=t_shared_eval,
        Ph=Ph_eval,
        T_amb=T_amb_eval,
        T_in_true=Tin_eval,
        Ps=Ps_directional_eval,
        Pc=Pc_eval,
        is_eval=True,
    )
    MAE, RMSE = model._evaluate(
        batch=batch_eval,
        t_min_train=data_train["t"].iloc[0].item(),
        t_span_days=window_days,
        dt=dt_train.item(),
        dtau=batch["dtau"].squeeze().item() if batch["dtau"].squeeze().dim() == 0 else None,
    )

    plot_batch_data(
        batch=batch,
        start_datetime=data_train["datetime"].iloc[0],
        path=model.log_dir,
        tag="train",
    )

    plot_batch_data(
        batch=batch_eval,
        start_datetime=data_train["datetime"].iloc[0],
        path=model.log_dir,
        tag="eval",
    )

    model_path = model.log_dir.joinpath(f"{window_days}")
    return model_path


def train_windows(
    data_train: pd.DataFrame,
    data_eval: pd.DataFrame,
    window_days: int = 10,
    stride_days: int = 10,
    k_per_season: int = 3,
    min_gap_days: float = 1.0,
) -> None:
    candidates = build_windows_by_season_indices(data=data_train, window_days=window_days, stride_days=stride_days)
    slices = pick_top_per_season(
        slices_scored=candidates,
        t_seconds=data_train["t"],
        k_per_season=k_per_season,
        min_gap_days=min_gap_days,
    )

    model_paths: List[Path] = []
    tag = str(window_days)
    for idx, sl in enumerate(slices, 1):
        subset_train = data_train.iloc[sl].reset_index(drop=True)
        print(
            f"Training window {idx}/{len(slices)}: Index [{sl.start}:{sl.stop}], Date: "
            + str(subset_train["datetime"].min())
            + " to "
            + str(subset_train["datetime"].max())
        )
        model_path = train_and_evaluate(
            data_train=subset_train,
            data_eval=data_eval,
            window_days=window_days,
            tag=tag,
        )
        model_paths.append(model_path)
