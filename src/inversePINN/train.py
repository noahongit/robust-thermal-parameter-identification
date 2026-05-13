from pathlib import Path
from typing import List

import pandas as pd
from PINN import PINN
from utils import build_windows_by_season_indices, pick_top_per_season
import imageio.v2 as imageio


def train_and_evaluate(
    data_train: pd.DataFrame,
    data_eval: pd.DataFrame,
    loss_weights: dict[str, float] = {
        "data": 5.0,
        "physics_in": 2.0,
        "physics_mass": 4.0,
    },
    n_training_steps: int = 100_000,
    n_training_steps_pre_train: int = 30_000,
    path_to_model_dir: str | None = None,
    tag: str | None = None,
) -> Path:
    pinn = PINN()
    pinn.fit(
        data=data_train,
        n_training_steps=n_training_steps,
        n_training_steps_pre_train=n_training_steps_pre_train,
        loss_weights=loss_weights,
        path_to_model_dir=path_to_model_dir,
    )
    model_path = pinn.evaluate(
        evaluation_data=data_eval,
        tag_for_logging=tag,
    )

    # analyse interpolation
    # pinn.plot_interpolation_diagnostics()

    # analyse training dynamics
    # output_filename = "logs/pinn_optimization.gif"
    # print(f"Saving GIF to {output_filename}...")

    # # Add a pause at the start (duplicate first frame) and end (duplicate last frame)
    # final_frames = [pinn.frames[0]] * 5 + pinn.frames + [pinn.frames[-1]] * 10

    # imageio.mimsave(output_filename, final_frames, fps=15, loop=0)
    # print("Done.")

    return model_path


def train_windows(
    data_train: pd.DataFrame,
    data_eval: pd.DataFrame,
    window_days: int = 10,
    stride_days: int = 10,
    k_per_season: int = 1,
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
    tag = str(window_days) + "5D4P4M"
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
            tag=tag,
        )
        model_paths.append(model_path)
