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
from src.indicators.flooding.floodscan import floodscan
from src.malawi import get_glofas_data as ggd
from src.malawi import get_floodscan_data as gfd

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
GLOFAS_DIR = DATA_DIR / "processed/malawi/GLOFAS_Data"
STATION = "glofas_1"


def get_floodscan_processed(adm: int = 2):
    floodscan_processed = floodscan.Floodscan()
    df_floodscan = floodscan_processed.read_processed_dataset(
        country_name=gfd.COUNTRY_NAME, country_iso3=gfd.COUNTRY_ISO3, adm_level=adm,
    )
    return df_floodscan


def get_glofas_reanalysis(version: int = 3, station: str = STATION):
    glofas_reanalysis = glofas.GlofasReanalysis()
    da_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_iso3=ggd.COUNTRY_ISO3, version=version
    )[station]
    return da_glofas_reanalysis


def get_glofas_forecast(
    version: int = 3, leadtimes: list = ggd.LEADTIMES, station: str = STATION
):
    glofas_forecast = glofas.GlofasForecast()
    da_glofas_forecast_dict = {
        leadtime: glofas_forecast.read_processed_dataset(
            country_iso3=ggd.COUNTRY_ISO3, version=version, leadtime=leadtime,
        )[station]
        for leadtime in leadtimes
    }
    da_glofas_forecast_dict = shift_dates(da_glofas_forecast_dict)
    return convert_dict_to_da(da_glofas_forecast_dict)


def get_glofas_reforecast(
    version: int = 3,
    interp: bool = True,
    leadtimes: list = ggd.LEADTIMES,
    station: str = STATION,
):
    glofas_reforecast = glofas.GlofasReforecast()
    da_glofas_reforecast_dict = {
        leadtime: glofas_reforecast.read_processed_dataset(
            country_iso3=ggd.COUNTRY_ISO3, version=version, leadtime=leadtime,
        )[station]
        for leadtime in leadtimes
    }
    if interp:
        da_glofas_reforecast_dict = interp_dates(da_glofas_reforecast_dict)
    da_glofas_reforecast_dict = shift_dates(da_glofas_reforecast_dict)
    return convert_dict_to_da(da_glofas_reforecast_dict)


def shift_dates(da_dict):
    return {
        leadtime: da.assign_coords(time=da.time.values + np.timedelta64(leadtime, "D"))
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
    data = np.array([da_glofas.values for da_glofas in da_glofas_dict.values()])
    # Create data array with all lead times, as well as ensemble members (number)
    # and timestep
    return xr.DataArray(
        data=data,
        dims=["leadtime", "number", "time"],
        coords=dict(
            number=list(da_glofas_dict.values())[0].number,  # ensemble member number
            time=time,
            leadtime=list(da_glofas_dict.keys()),
        ),
    )


def get_da_glofas_summary(da_glofas):
    nsig_max = 3
    percentile_dict = {
        **{"median": 50.0},
        **{f"{n}sig+": norm.cdf(n) * 100 for n in range(1, nsig_max + 1)},
        **{f"{n}sig-": (1 - norm.cdf(n)) * 100 for n in range(1, nsig_max + 1)},
    }
    coord_names = ["leadtime", "time"]
    data_vars_dict = {
        var_name: (coord_names, np.percentile(da_glofas, percentile_value, axis=1))
        for var_name, percentile_value in percentile_dict.items()
    }

    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(time=da_glofas.time, leadtime=da_glofas.leadtime),
    )

