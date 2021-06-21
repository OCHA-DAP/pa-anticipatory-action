from typing import List, Dict
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev
from scipy.stats import rankdata
import xskillscore as xs
import matplotlib.pyplot as plt

from src.indicators.flooding.glofas import glofas


logger = logging.getLogger(__name__)


def get_glofas_reanalysis(
    country_iso3: str, version: int = glofas.DEFAULT_VERSION
) -> xr.Dataset:
    glofas_reanalysis = glofas.GlofasReanalysis()
    ds_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_iso3=country_iso3, version=version
    )
    return ds_glofas_reanalysis


def get_glofas_forecast(
    country_iso3: str, leadtimes: List[int], version: int = glofas.DEFAULT_VERSION
) -> xr.Dataset:
    glofas_forecast = glofas.GlofasForecast()
    ds_glofas_forecast_dict = {
        leadtime: glofas_forecast.read_processed_dataset(
            country_iso3=country_iso3,
            leadtime=leadtime,
            version=version,
        )
        for leadtime in leadtimes
    }
    ds_glofas_forecast_dict = _shift_dates(ds_glofas_forecast_dict)
    return _convert_dict_to_ds(ds_glofas_forecast_dict)


def get_glofas_reforecast(
    country_iso3: str,
    leadtimes: List[int],
    interp: bool = True,
    version: int = glofas.DEFAULT_VERSION,
) -> xr.Dataset:
    glofas_reforecast = glofas.GlofasReforecast()
    ds_glofas_reforecast_dict = {
        leadtime: glofas_reforecast.read_processed_dataset(
            country_iso3=country_iso3,
            version=version,
            leadtime=leadtime,
        )
        for leadtime in leadtimes
    }
    if interp:
        ds_glofas_reforecast_dict = _interp_dates(ds_glofas_reforecast_dict)
    ds_glofas_reforecast_dict = _shift_dates(ds_glofas_reforecast_dict)
    return _convert_dict_to_ds(ds_glofas_reforecast_dict)


def _shift_dates(ds_dict) -> Dict[int, xr.Dataset]:
    return {
        leadtime: ds.assign_coords(time=ds.time.values + np.timedelta64(leadtime, "D"))
        for leadtime, ds in ds_dict.items()
    }


def _interp_dates(ds_dict) -> Dict[int, xr.Dataset]:
    # Sort the ensemble members to preserve the properties throughout the interpolation
    for leadtime, ds in ds_dict.items():
        for station in ds.keys():
            ds[station].values = np.sort(ds[station].values, axis=0)
        ds_dict[leadtime] = ds
    # Interpolate
    return {
        leadtime: ds.interp(
            time=pd.date_range(ds.time.min().values, ds.time.max().values),
            method="linear",
        )
        for leadtime, ds in ds_dict.items()
    }


def _convert_dict_to_ds(ds_glofas_dict) -> xr.Dataset:
    # Create time array that accounts for all the shifts
    time = np.arange(
        ds_glofas_dict[min(ds_glofas_dict.keys())].time.values[0],
        ds_glofas_dict[max(ds_glofas_dict.keys())].time.values[-1]
        + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    # Re-index all the arrays by the full time array
    ds_glofas_dict = {
        leadtime: ds_glofas.reindex({"time": time})
        for leadtime, ds_glofas in ds_glofas_dict.items()
    }
    # Concatenate the arrays together. Use the already present 'step'
    # variable (which represents the lead time), but then rename it
    # and convert it to a simple integer.
    return (
        xr.concat(ds_glofas_dict.values(), "step")
        .assign_coords(step=list(ds_glofas_dict.keys()))
        .rename({"step": "leadtime"})
    )


def get_return_periods(
    ds_reanalysis: xr.Dataset, years=None, method="analytical", show_plot=False
) -> pd.DataFrame:
    """
    :param ds_reanalysis: GloFAS reanalysis dataset
    :param years: Return period years to compute
    :param method: Either "analytical" or "empirical"
    :return: Dataframe with return period years as index and stations as columns
    """
    if years is None:
        years = [1.5, 2, 3, 5, 10, 20]
    stations = list(ds_reanalysis.keys())
    df_rps = pd.DataFrame(columns=stations, index=years)
    for station in stations:
        if method == "analytical":
            f_rp = _get_return_period_function_analytical(
                ds_reanalysis=ds_reanalysis, station=station, show_plot=show_plot
            )
        elif method == "empirical":
            f_rp = _get_return_period_function_empirical(
                ds_reanalysis=ds_reanalysis, station=station
            )
        else:
            logger.error(f"{method} is not a valid keyword for method")
            return None
        df_rps[station] = np.round(f_rp(years))
    return df_rps


def _get_return_period_function_analytical(
    ds_reanalysis: xr.Dataset, station: str, show_plot: bool
):
    df_rp = _get_return_period_df(ds_reanalysis=ds_reanalysis, station=station)
    discharge = df_rp["discharge"]
    shape, loc, scale = gev.fit(
        discharge, loc=discharge.median(), scale=discharge.median() / 2
    )
    x = np.linspace(discharge.min(), discharge.max(), 100)
    if show_plot:
        fig, ax = plt.subplots()
        ax.hist(discharge, density=True, bins=20)
        ax.plot(x, gev.pdf(x, shape, loc, scale))
        ax.set_title(station)
        plt.show()
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1 - y)
    return interp1d(y, x)


def _get_return_period_function_empirical(ds_reanalysis: xr.Dataset, station: str):
    df_rp = _get_return_period_df(ds_reanalysis=ds_reanalysis, station=station)
    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return interp1d(df_rp["rp"], df_rp["discharge"])


def _get_return_period_df(ds_reanalysis: xr.Dataset, station: str):
    df_rp = (
        ds_reanalysis.to_dataframe()[[station]]
        .rename(columns={station: "discharge"})
        .resample(rule="A", kind="period")
        .max()
        .sort_values(by="discharge", ascending=False)
    )
    df_rp["year"] = df_rp.index.year
    return df_rp


def get_crps(
    ds_reanalysis: xr.Dataset,
    ds_reforecast: xr.Dataset,
    normalization: str = None,
    thresh: [float, Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    :param ds_reanalysis: GloFAS reanalysis xarray dataset
    :param ds_reforecast: GloFAS reforecast xarray dataset
    :param normalization: (optional) Can be 'mean' or 'std', reanalysis metric to divide the CRPS
    :param thresh: (optional) Either a single value, or a dictionary with format {station name: thresh}
    :return: DataFrame with station column names and leadtime index
    """
    stations = list(ds_reanalysis.keys())
    leadtimes = ds_reforecast.leadtime.values
    df_crps = pd.DataFrame(index=leadtimes, columns=stations)

    for station in stations:
        for leadtime in leadtimes:
            forecast = ds_reforecast[station].sel(leadtime=leadtime).dropna(dim="time")
            observations = ds_reanalysis[station].reindex({"time": forecast.time})
            if normalization == "mean":
                norm = observations.mean().values
            elif normalization == "std":
                norm = observations.std().values
            elif normalization is None:
                norm = 1
            # TODO: Add error for other normalization values
            if thresh is not None:
                # Thresh can either be dict of floats, or float
                try:
                    thresh_to_use = thresh[station]
                except TypeError:
                    thresh_to_use = thresh
                idx = observations > thresh_to_use
                forecast, observations = forecast[:, idx], observations[idx]
            crps = (
                xs.crps_ensemble(observations, forecast, member_dim="number").values
                / norm
            )
            df_crps.loc[leadtime, station] = crps

    return df_crps


def get_groups_above_threshold(observations, threshold, min_duration=1):
    groups = np.where(np.diff(observations > threshold, prepend=False, append=False))[
        0
    ].reshape(-1, 2)
    return [group for group in groups if group[1] - group[0] >= min_duration]


def get_glofas_activations(da_glofas, thresh, ndays):
    vals = da_glofas.values
    groups = get_groups_above_threshold(vals, thresh, ndays)
    df_glofas_act = pd.DataFrame(groups, columns=["start_index", "end_index"])
    df_glofas_act["num_days"] = (
        df_glofas_act["end_index"] - df_glofas_act["start_index"]
    )
    df_glofas_act["start_date"] = df_glofas_act["start_index"].apply(
        lambda x: da_glofas.time[x].values
    )
    df_glofas_act["end_date"] = df_glofas_act["end_index"].apply(
        lambda x: da_glofas.time[x].values
    )
    return df_glofas_act


def get_rank(observations: np.array, forecast: np.array) -> np.array:
    # Create array of both obs and forecast
    rank_array = np.concatenate(([observations], forecast))
    # Calculate rank and take 0th array, which should be the obs
    rank = rankdata(rank_array, axis=0)[0]
    return rank


def calc_mpe(observations: np.array, forecast: np.array) -> float:
    mean_forecast = forecast.mean(axis=0)
    denominator = observations
    return (
        ((mean_forecast - observations) / denominator).sum()
        / len(observations.time)
        * 100
    )


def get_same_obs_and_forecast(
    da_observations: xr.DataArray, da_forecast: xr.DataArray, leadtime: int
) -> (xr.DataArray, xr.DataArray):
    """
    For the GloFAS reanalysis and reforecast at a particular station, get matching data
    ranges for the two datasets
    :param da_observations: GloFAS reanalysis at a particular station
    :param da_forecast: GloFAS reforecast at a particular station
    :param leadtime: Leadtime
    :return: Observations and forecast with overlapping values only
    """
    forecast = da_forecast.sel(leadtime=leadtime).dropna(dim="time")
    observations = da_observations.reindex({"time": forecast.time}).dropna(dim="time")
    forecast = forecast.reindex({"time": observations.time})
    return observations, forecast
