import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import glofas
from src.bangladesh import get_glofas_data as ggd

pd.options.mode.chained_assignment = None

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
STATION = "Bahadurabad_glofas"
FFWC_DIR = DATA_DIR / "private/exploration/bgd/ffwc"

# Event definition
EVENT_WATER_THRESH = 19.5 + 0.85
EVENT_NDAYS_THRESH = 3


GLOFAS_EXPLORATION_FOLDER = DATA_DIR / "public/exploration/bgd/glofas"


def get_glofas_df(
    glofas_dir: Path = GLOFAS_EXPLORATION_FOLDER,
    district_list: list = None,
    year_min: int = 1979,
    year_max: int = 2021,
) -> pd.DataFrame:
    """Get GloFAS data from the exploration directory -- from the 2020
    analysis."""
    glofas_df = pd.DataFrame(columns=district_list)
    for year in range(year_min, year_max):
        glofas_filename = Path(f"{year}.csv")
        glofas_df = glofas_df.append(
            pd.read_csv(glofas_dir / glofas_filename, index_col=0)
        )
    glofas_df.index = pd.to_datetime(glofas_df.index, format="%Y-%m-%d")
    return glofas_df


def get_glofas_reanalysis(version: int = 3):
    glofas_reanalysis = glofas.GlofasReanalysis()
    da_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_iso3=ggd.COUNTRY_ISO3, version=version
    )[STATION]
    return da_glofas_reanalysis


def get_glofas_forecast(version: int = 3, leadtimes: list = ggd.LEADTIMES):
    glofas_forecast = glofas.GlofasForecast()
    da_glofas_forecast_dict = {
        leadtime: glofas_forecast.read_processed_dataset(
            country_iso3=ggd.COUNTRY_ISO3, version=version, leadtime=leadtime,
        )[STATION]
        for leadtime in leadtimes
    }
    da_glofas_forecast_dict = shift_dates(da_glofas_forecast_dict)
    return convert_dict_to_da(da_glofas_forecast_dict)


def get_glofas_reforecast(
    version: int = 3, interp: bool = True, leadtimes: list = ggd.LEADTIMES
):
    glofas_reforecast = glofas.GlofasReforecast()
    da_glofas_reforecast_dict = {
        leadtime: glofas_reforecast.read_processed_dataset(
            country_iso3=ggd.COUNTRY_ISO3, version=version, leadtime=leadtime,
        )[STATION]
        for leadtime in leadtimes
    }
    if interp:
        da_glofas_reforecast_dict = interp_dates(da_glofas_reforecast_dict)
    da_glofas_reforecast_dict = shift_dates(da_glofas_reforecast_dict)
    return convert_dict_to_da(da_glofas_reforecast_dict)


def shift_dates(da_dict):
    return {
        leadtime: da.assign_coords(
            time=da.time.values + np.timedelta64(leadtime, "D")
        )
        for leadtime, da in da_dict.items()
    }


def interp_dates(da_dict):
    return {
        leadtime: da.interp(
            time=pd.date_range(da.time.min().values, da.time.max().values),
            method="linear",
        )
        for leadtime, da in da_dict.items()
    }


def convert_dict_to_da(da_glofas_dict):
    # Create time array that accounts for all the shifts
    time = np.arange(
        da_glofas_dict[min(da_glofas_dict.keys())].time.values[0],
        da_glofas_dict[max(da_glofas_dict.keys())].time.values[-1]
        + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    da_glofas_dict = {
        leadtime: da_glofas.reindex({"time": time})
        for leadtime, da_glofas in da_glofas_dict.items()
    }
    data = np.array(
        [da_glofas.values for da_glofas in da_glofas_dict.values()]
    )
    # Create data array with all lead times, as well as ensemble members
    # (number) and timestep
    return xr.DataArray(
        data=data,
        dims=["leadtime", "number", "time"],
        coords=dict(
            number=list(da_glofas_dict.values())[
                0
            ].number,  # ensemble member number
            time=time,
            leadtime=list(da_glofas_dict.keys()),
        ),
    )


def get_da_glofas_summary(da_glofas):
    nsig_max = 3
    percentile_dict = {
        **{"median": 50.0},
        **{f"{n}sig+": norm.cdf(n) * 100 for n in range(1, nsig_max + 1)},
        **{
            f"{n}sig-": (1 - norm.cdf(n)) * 100 for n in range(1, nsig_max + 1)
        },
    }
    coord_names = ["leadtime", "time"]
    data_vars_dict = {
        var_name: (
            coord_names,
            np.percentile(da_glofas, percentile_value, axis=1),
        )
        for var_name, percentile_value in percentile_dict.items()
    }
    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(time=da_glofas.time, leadtime=da_glofas.leadtime),
    )


def read_in_ffwc():
    # Read in data from Sazzad that has forecasts
    ffwc_wl_filename = "Bahadurabad_WL_forecast20172019.xlsx"
    ffwc_leadtimes = [1, 2, 3, 4, 5]

    # Need to combine the three sheets
    df_ffwc_wl_dict = pd.read_excel(
        FFWC_DIR / ffwc_wl_filename,
        sheet_name=None,
        header=[1],
        index_col="Date",
    )
    df_ffwc_wl = (
        df_ffwc_wl_dict["2017"]
        .append(df_ffwc_wl_dict["2018"])
        .append(df_ffwc_wl_dict["2019"])
        .rename(
            columns={
                f"{leadtime*24} hrs": f"ffwc_{leadtime}day"
                for leadtime in ffwc_leadtimes
            }
        )
    ).drop(
        columns=["Observed WL"]
    )  # drop observed because we will use the mean later
    # Convert date time to just date
    df_ffwc_wl.index = df_ffwc_wl.index.floor("d")

    # Then read in the older data (goes back much futher)
    FFWC_RL_HIS_FILENAME = (
        "2020-06-07 Water level data Bahadurabad Upper danger level.xlsx"
    )
    ffwc_rl_name = "{}/{}".format(FFWC_DIR, FFWC_RL_HIS_FILENAME)
    df_ffwc_wl_old = pd.read_excel(ffwc_rl_name, index_col=0, header=0)
    df_ffwc_wl_old.index = pd.to_datetime(
        df_ffwc_wl_old.index, format="%d/%m/%y"
    )
    df_ffwc_wl_old = df_ffwc_wl_old[["WL"]].rename(columns={"WL": "observed"})[
        df_ffwc_wl_old.index < df_ffwc_wl.index[0]
    ]
    df_ffwc_wl = pd.concat([df_ffwc_wl_old, df_ffwc_wl])

    # Read in the more recent file from Hassan
    ffwc_full_data_filename = "SW46.9L_19-11-2020.xls"
    df_ffwc_wl_full = (
        pd.read_excel(
            FFWC_DIR / ffwc_full_data_filename, index_col="DateTime"
        ).rename(columns={"WL(m)": "observed"})
    )[["observed"]]

    # Mutliple observations per day. Find mean and std
    df_ffwc_wl_full["date"] = df_ffwc_wl_full.index.date
    df_ffwc_wl_full = (df_ffwc_wl_full.groupby("date").agg(["mean", "std"]))[
        "observed"
    ].rename(columns={"mean": "observed", "std": "obs_std"})
    df_ffwc_wl_full.index = pd.to_datetime(df_ffwc_wl_full.index)

    # Combine with first DF

    df_ffwc_wl = pd.merge(
        df_ffwc_wl_full[["obs_std"]],
        df_ffwc_wl,
        left_index=True,
        right_index=True,
        how="outer",
    )
    df_ffwc_wl.update(df_ffwc_wl_full, overwrite=False)

    return df_ffwc_wl


def get_events(df_ffwc_wl):
    groups = get_groups_above_threshold(
        df_ffwc_wl["observed"], EVENT_WATER_THRESH
    )

    # Only take those that are 3 consecutive days
    groups = [
        group for group in groups if group[1] - group[0] >= EVENT_NDAYS_THRESH
    ]

    # Mark the first date in each series as TP
    events = [group[0] + EVENT_NDAYS_THRESH - 1 for group in groups]

    df_ffwc_wl["event"] = False
    df_ffwc_wl["event"][events] = True

    return df_ffwc_wl


def get_groups_above_threshold(observations, threshold):
    return np.where(
        np.diff(np.hstack(([False], observations > threshold, [False])))
    )[0].reshape(-1, 2)
