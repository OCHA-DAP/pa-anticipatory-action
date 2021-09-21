import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

import indicators.flooding.glofas.glofas_forecast
import indicators.flooding.glofas.glofas_reanalysis
from indicators.flooding.glofas import glofas
from utils_general.statistics import (
    get_return_period_function_analytical,
    get_return_period_function_empirical,
    calc_crps,
)

logger = logging.getLogger(__name__)


def get_glofas_reanalysis(
    country_iso3: str, version: int = glofas.DEFAULT_VERSION, **kwargs
) -> xr.Dataset:
    glofas_reanalysis = indicators.flooding.glofas.glofas_reanalysis.GlofasReanalysis(**kwargs)
    ds_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_iso3=country_iso3, version=version
    )
    return ds_glofas_reanalysis


def _get_glofas_forecast_base(
    is_reforecast: bool,
    country_iso3: str,
    leadtimes: List[int],
    interp: bool = False,
    shift_dates: bool = True,
    version: int = glofas.DEFAULT_VERSION,
    split_by_leadtimes: bool = False,
    **kwargs,
) -> xr.Dataset:
    if is_reforecast:
        glofas_forecast = indicators.flooding.glofas.glofas_forecast.GlofasReforecast(**kwargs)
    else:
        glofas_forecast = indicators.flooding.glofas.glofas_forecast.GlofasForecast(**kwargs)
    if split_by_leadtimes:
        ds_glofas_forecast_dict = {
            leadtime: glofas_forecast.read_processed_dataset(
                country_iso3=country_iso3,
                version=version,
                leadtime=leadtime,
            )
            for leadtime in leadtimes
        }
    else:
        # Split up the dataset into different leadtimes, because then
        # it's easier to do the shifts
        ds_glofas_forecast = glofas_forecast.read_processed_dataset(
            country_iso3=country_iso3, version=version
        )
        ds_glofas_forecast_dict = {
            leadtime: ds_glofas_forecast.sel(
                step=np.timedelta64(leadtime, "D")
            )
            for leadtime in leadtimes
        }
    if interp:
        ds_glofas_forecast_dict = _interp_dates(ds_glofas_forecast_dict)
    if shift_dates:
        ds_glofas_forecast_dict = _shift_dates(ds_glofas_forecast_dict)
    return _convert_dict_to_ds(ds_glofas_forecast_dict)


def get_glofas_forecast(
    country_iso3: str,
    leadtimes: List[int],
    version: int = glofas.DEFAULT_VERSION,
    split_by_leadtimes=False,
) -> xr.Dataset:
    return _get_glofas_forecast_base(
        is_reforecast=False,
        country_iso3=country_iso3,
        leadtimes=leadtimes,
        version=version,
        split_by_leadtimes=split_by_leadtimes,
    )


def get_glofas_reforecast(
    country_iso3: str,
    leadtimes: List[int],
    interp: bool = True,
    shift_dates: bool = True,
    version: int = glofas.DEFAULT_VERSION,
    split_by_leadtimes: bool = False,
    **kwargs,
) -> xr.Dataset:
    return _get_glofas_forecast_base(
        is_reforecast=True,
        country_iso3=country_iso3,
        leadtimes=leadtimes,
        interp=interp,
        shift_dates=shift_dates,
        version=version,
        split_by_leadtimes=split_by_leadtimes,
        **kwargs,
    )


def _shift_dates(ds_dict) -> Dict[int, xr.Dataset]:
    return {
        leadtime: ds.assign_coords(
            time=ds.time.values + np.timedelta64(leadtime, "D")
        )
        for leadtime, ds in ds_dict.items()
    }


def _interp_dates(ds_dict) -> Dict[int, xr.Dataset]:
    # Sort the ensemble members to preserve the properties throughout
    # the interpolation
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


def get_glofas_forecast_summary(ds_glofas_forecast):
    percentiles = np.arange(0, 105, 5)
    coord_names = ["percentile", "leadtime", "time"]
    data_vars_dict = {
        station: (
            coord_names,
            np.percentile(ds_glofas_forecast[station], percentiles, axis=1),
        )
        for station in ds_glofas_forecast.keys()
    }
    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(
            time=ds_glofas_forecast.time,
            leadtime=ds_glofas_forecast.leadtime,
            percentile=percentiles,
        ),
    )


def get_return_periods(
    ds_glofas: xr.Dataset,
    years: list = None,
    method: str = "analytical",
    show_plots: bool = False,
    extend_factor: int = 1,
) -> pd.DataFrame:
    """
    :param ds_glofas: GloFAS reanalysis or forecast/reforecast dataset
    :param years: Return period years to compute
    :param method: Either "analytical" or "empirical"
    :param show_plots: If method is analytical, can show the histogram and GEV
    distribution overlaid
    :param extend_factor: If method is analytical, can extend the interpolation
    range to reach higher return periods
    :return: Dataframe with return period years as index and stations as
    columns
    """
    if years is None:
        years = [1.5, 2, 5, 10, 20]
    stations = list(ds_glofas.keys())
    df_rps = pd.DataFrame(columns=stations, index=years)
    for station in stations:
        df_rp = _get_return_period_df(ds_glofas=ds_glofas, station=station)
        if method == "analytical":
            f_rp = get_return_period_function_analytical(
                df_rp=df_rp,
                rp_var="discharge",
                show_plots=show_plots,
                plot_title=station,
                extend_factor=extend_factor,
            )
        elif method == "empirical":
            f_rp = get_return_period_function_empirical(
                df_rp=df_rp,
                rp_var="discharge",
            )
        else:
            logger.error(f"{method} is not a valid keyword for method")
            return None
        df_rps[station] = np.round(f_rp(years))
    return df_rps


def _get_return_period_df(ds_glofas: xr.Dataset, station: str):
    df_rp = (
        ds_glofas.to_dataframe()[[station]]
        .rename(columns={station: "discharge"})
        .resample(rule="A", kind="period")
        .max()
        .sort_values(by="discharge", ascending=False)
    )
    df_rp["year"] = df_rp.index.year
    return df_rp


def get_crps_glofas(
    ds_reanalysis: xr.Dataset,
    ds_reforecast: xr.Dataset,
    normalization: str = None,
    thresh: [float, Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    :param ds_reanalysis: GloFAS reanalysis xarray dataset
    :param ds_reforecast: GloFAS reforecast xarray dataset
    :param normalization: (optional) Can be 'mean' or 'std', reanalysis metric
    to divide the CRPS
    :param thresh: (optional) Either a single value, or a dictionary with
    format {station name: thresh} to select values greater than thresh
    :return: DataFrame with station column names and leadtime index
    """
    stations = list(ds_reanalysis.keys())
    leadtimes = ds_reforecast.leadtime.values
    df_crps = pd.DataFrame(index=leadtimes, columns=stations)

    for station in stations:
        for leadtime in leadtimes:
            forecast = (
                ds_reforecast[station]
                .sel(leadtime=leadtime)
                .dropna(dim="time")
            )
            observations = ds_reanalysis[station].reindex(
                {"time": forecast.time}
            )

            if normalization == "mean":
                norm = observations.mean().item()
            elif normalization == "std":
                norm = observations.std().item()
            else:
                norm = normalization
            if thresh is not None:
                # Thresh can either be dict of floats, or float
                try:
                    thresh_to_use = thresh[station]
                except TypeError:
                    thresh_to_use = thresh
                idx = observations > thresh_to_use
                forecast, observations = forecast[:, idx], observations[idx]

            df_crps.loc[leadtime, station] = calc_crps(
                observations,
                forecast,
                normalization=norm,
            )

    return df_crps


def get_groups_above_threshold(
    observations: np.ndarray,
    threshold: float,
    min_duration: int = 1,
    additional_condition: np.ndarray = None,
) -> List:
    """
    Get indices where consecutive values are equal to or above a
    threshold :param observations: The array of values to search for
    groups (length N) :param threshold: The threshold above which the
    values must be :param min_duration: The minimum group size (default
    1) :param additional_condition: (optional) Any additional condition
    the values must satisfy (array-like of bools, length N) :return:
    list of arrays with indices
    """
    condition = observations >= threshold
    if additional_condition is not None:
        condition = condition & additional_condition
    groups = np.where(np.diff(condition, prepend=False, append=False))[
        0
    ].reshape(-1, 2)
    return [group for group in groups if group[1] - group[0] >= min_duration]


def get_dates_list_from_data_array(
    da: xr.DataArray, threshold: float, min_duration: int = 1
) -> List[np.datetime64]:
    """
    Given a data array of a smoothly varying quantity over time,
    get the dates of an event occurring where the quantity crosses
    some threshold for a specified duration. If the duration is more than
    one timestep, then the event date is defined as the timestep when
    the duration is reached.
    :param da: Data array with the main quantity
    :param threshold: Threshold >= which an event is defined
    :param min_duration: Number of timesteps above the quantity to be
    considered an event
    :return: List of event dates
    """
    groups = get_groups_above_threshold(
        observations=da.to_masked_array(),
        threshold=threshold,
        min_duration=min_duration,
    )
    return [da.time[group[0] + min_duration - 1].data for group in groups]


def get_detection_stats(
    true_event_dates: np.ndarray,
    forecasted_event_dates: np.ndarray,
    days_before_buffer: int,
    days_after_buffer: int,
) -> dict:
    """
    Give a list of true and forecasted event dates, calculate how many
    true / false positives and false negatives occurred
    :param true_event_dates: A list of dates when the true events occurred
    :param forecasted_event_dates: A list of dates when the events were
    forecasted to occur
    :param days_before_buffer: How many days before the forecasted date the
    true event can occur. Usually set to the lead time or a small number
    (even 0)
    :param days_after_buffer: How many days after the forecasted date the
    true event can occur. Can usually be a generous number
    like 30, since forecasting too early isn't usually an issue
    :return: dictionary with parameters
    """
    df_detected = pd.DataFrame(
        0, index=np.array(true_event_dates), columns=["detected"]
    )
    FP = 0
    # Loop through the forecasted event
    for forecasted_event in forecasted_event_dates:
        # Calculate the offset from the true dates
        days_offset = (true_event_dates - forecasted_event) / np.timedelta64(
            1, "D"
        )
        # Calculate which true events were detected by this forecast event
        detected = (days_offset >= -1 * days_before_buffer) & (
            days_offset <= days_after_buffer
        )
        df_detected.loc[detected, "detected"] += 1
        # If there were no detections at all, it's a FP
        if not sum(detected):
            FP += 1
    return {
        # TP is the number of true events that were detected
        "TP": sum(df_detected["detected"] > 0),
        # FN is the number of true events that were not detected
        "FN": sum(df_detected["detected"] == 0),
        "FP": FP,
    }


def get_more_detection_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute precision, recall, F1, POD and FAR
    :param df: Dataframe with columns TP, FP and FN
    :return: Dataframe with additional stats columns
    """
    # Convert everything to float to avoid zero division errors
    for q in ["TP", "FP", "FN"]:
        df[q] = df[q].astype("float")
    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["recall"] = df["TP"] / (df["TP"] + df["FN"])
    df["F1"] = 2 / (1 / df["precision"] + 1 / df["recall"])
    df["POD"] = df["recall"]
    df["FAR"] = 1 - df["precision"]
    for q in ["TP", "FP", "FN"]:
        df[q] = df[q].astype("int")
    return df


def round_to_n(x: float, n: int) -> int:
    """
    Round float x to the nearest multiple of n
    :param x: The number to be rounded
    :param n: The integer multiple to round to
    :return: Rounded integer
    """
    return (np.around(x / n, decimals=0) * n).astype(int)


def get_rank(observations: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    # Create array of both obs and forecast
    rank_array = np.concatenate(([observations], forecast))
    # Calculate rank and take 0th array, which should be the obs
    rank = rankdata(rank_array, axis=0)[0]
    return rank


def get_same_obs_and_forecast(
    da_observations: xr.DataArray, da_forecast: xr.DataArray, leadtime: int
) -> (xr.DataArray, xr.DataArray):
    """
    For the GloFAS reanalysis and reforecast at a particular station,
    get matching data ranges for the two datasets :param
    da_observations: GloFAS reanalysis at a particular station :param
    da_forecast: GloFAS reforecast at a particular station :param
    leadtime: Leadtime :return: Observations and forecast with
    overlapping values only
    """
    forecast = da_forecast.sel(leadtime=leadtime).dropna(dim="time")
    observations = da_observations.reindex({"time": forecast.time}).dropna(
        dim="time"
    )
    forecast = forecast.reindex({"time": observations.time})
    return observations, forecast
