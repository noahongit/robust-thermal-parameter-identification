from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
from train import train_and_evaluate, train_windows
from utils import (
    get_IDAICE_data,
    plot_idaice_data,
)


if __name__ == "__main__":
    data_train_sim_res = get_IDAICE_data(
        path_to_data="../data/IDAICE/v005___mit 500 Liter Speicher ___+-1,0 deg/data.csv",
        start_date="2019-01-01 03:00:00",
        end_date="2019-12-31 03:00:00",
    )

    data_eval_sim_res = get_IDAICE_data(
        path_to_data="../data/IDAICE/v005___mit 500 Liter Speicher ___+-1,5 deg/data.csv",
        start_date="2019-01-01 03:00:00",
        end_date="2019-12-31 03:00:00",
    )
    # data_eval_sim_res = get_IDAICE_data(
    #     path_to_data="data/IDAICE/v005___A___+-1,0 deg___Winter/data.csv",
    #     start_date="2019-01-15 03:00:00",
    #     end_date="2019-02-10 03:00:00",
    #     solar_features=True,
    # )

    # plot_idaice_data(data=data_train_sim_res, solar_features=True)
    # plot_idaice_data(data=data_eval_sim_res, solar_features=True)

    for _ in [7]:
        train_windows(
            data_train=data_train_sim_res,
            data_eval=data_eval_sim_res,
            window_days=int(_),
            stride_days=1,
            k_per_season=3,
        )
