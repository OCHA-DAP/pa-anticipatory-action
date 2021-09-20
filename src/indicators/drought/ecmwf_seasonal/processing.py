import sys
import os
from pathlib import Path
import geopandas as gpd
from rasterstats import zonal_stats
import logging
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)

from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.indicators.drought.config import Config
from src.utils_general.statistics import calc_crps
from src.utils_general.raster_manipulation import compute_raster_statistics

logger = logging.getLogger(__name__)


def get_ecmwf_forecast(
    country_iso3: str, version: int = 5, **kwargs
) -> xr.Dataset:
    """
    Retrieve the processed dataset with the forecast for each
    publication date and corresponding lead times Args: version: version
    of forecast model that was used (only changes once every couple of
    years)
    """
    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast(**kwargs)
    ds_ecmwf_forecast = ecmwf_forecast.read_processed_dataset(
        country_iso3=country_iso3,
        version=version,
    )
    ds_ecmwf_forecast = convert_tprate_precipitation(ds_ecmwf_forecast)

    return ds_ecmwf_forecast


def get_ecmwf_forecast_by_leadtime(country_iso3, version: int = 5):
    """
    Reshape dataset to have the time variable as the month during the
    forecast was valid instead of the month the forecast was published
    Args: version: version of forecast model that was used (only changes
    once every couple of years)

    Returns: dataset with valid month per publication data-leadtime
    """
    ds_ecmwf_forecast = get_ecmwf_forecast(
        country_iso3=country_iso3, version=version
    )
    ds_ecmwf_forecast_dict = dates_per_leadtime(ds_ecmwf_forecast)
    return convert_dict_to_da(ds_ecmwf_forecast_dict)


def compute_stats_per_admin(
    country,
    adm_level=1,
    pcode_col="ADM1_PCODE",
    add_col: List[str] = None,
    use_cache=True,
    interpolate=True,
    date_list=None,
):
    config = Config()
    parameters = config.parameters(country)
    country_iso3 = parameters["iso3_code"]

    country_data_raw_dir = os.path.join(
        config.DATA_DIR, config.PUBLIC_DIR, config.RAW_DIR, country_iso3
    )
    country_data_processed_dir = os.path.join(
        config.DATA_DIR, config.PUBLIC_DIR, config.PROCESSED_DIR, country_iso3
    )
    adm_boundaries_path = os.path.join(
        country_data_raw_dir,
        config.SHAPEFILE_DIR,
        parameters[f"path_admin{adm_level}_shp"],
    )

    # read the forecasts
    ds = get_ecmwf_forecast_by_leadtime(country_iso3)

    if interpolate:
        # read observed data to get resolution to interpolate to
        # interpolate forecast data such that it has the same resolution
        # as the observed values using "nearest" as interpolation method
        # and not "linear" because the forecasts are designed to have
        # sharp edged and not be smoothed
        # now standard upsampling 4 times, can be made variable
        new_lon = np.arange(
            ds.longitude[0] - 0.125, ds.longitude[-1] + 0.25, 0.25
        )
        new_lat = np.arange(
            ds.latitude[0] + 0.125, ds.latitude[-1] - 0.25, -0.25
        )

        ds = ds.interp(latitude=new_lat, longitude=new_lon, method="nearest")

    # loop over dates
    if date_list is None:
        date_list = ds.time.values
    for date in date_list:
        date_dt = pd.to_datetime(date)
        if interpolate:
            output_filename = (
                f"{parameters['iso3_code'].lower()}"
                f"_seasonal-monthly-single-levels_v5_interp_{date_dt.year}"
                f"_{date_dt.month}_adm{adm_level}_stats.csv"
            )
        else:
            output_filename = (
                f"{parameters['iso3_code'].lower()}"
                f"_seasonal-monthly-single-levels_v5_{date_dt.year}"
                f"_{date_dt.month}_adm{adm_level}_stats.csv"
            )
        output_path = os.path.join(
            country_data_processed_dir, "ecmwf", output_filename
        )
        # If caching is on and file already exists, don't download again
        if use_cache and Path(output_path).exists():
            logger.debug(
                f"{output_path} already exists and cache is set to True,"
                " skipping"
            )
        else:
            ds_sel = ds.sel(time=date)
            gdf_adm = gpd.read_file(adm_boundaries_path)
            df = compute_raster_statistics(
                gdf_adm,
                pcode_col,
                ds_sel.rio.write_crs("EPSG:4326"),
                lon_coord="longitude",
                lat_coord="latitude",
            )
            df = df.merge(
                gdf_adm[add_col + [pcode_col]], on=pcode_col, how="left"
            )
            df["date"] = date_dt
            df.to_csv(output_path)


# TODO: create function to retrieve the stats file


def compute_zonal_stats(
    ds,
    raster_transform,
    adm_path,
    adm_col,
    percentile_list=np.arange(10, 91, 10),
):
    # compute statistics on level in adm_path for all dates in ds
    df_list = []
    for leadtime in ds.leadtime.values:
        for number in ds.number.values:
            df = gpd.read_file(adm_path)[[adm_col, "geometry"]]
            ds_date = ds.sel(number=number, leadtime=leadtime)
            df[["mean_cell", "max_cell", "min_cell"]] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date.values,
                    affine=raster_transform,
                    nodata=np.nan,
                )
            )[["mean", "max", "min"]]

            df[
                [f"percentile_{str(p)}" for p in percentile_list]
            ] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date.values,
                    affine=raster_transform,
                    nodata=np.nan,
                    stats=" ".join(
                        [f"percentile_{str(p)}" for p in percentile_list]
                    ),
                )
            )[
                [f"percentile_{str(p)}" for p in percentile_list]
            ]

            df["number"] = number
            df["leadtime"] = leadtime

            df_list.append(df)
        df_hist = pd.concat(df_list)
        # drop the geometry column, else csv becomes huge
        df_hist = df_hist.drop("geometry", axis=1)

    return df_hist


def convert_tprate_precipitation(da):
    """
    The ECMWF seasonal forecast reports precipitation as tprate, which
    is in meter/second. To convert this to the total precipitation in a
    month in meter, we multiply the tprate by the number of seconds in a
    month Thereafter we multiply by 1000 to get the total millimeters in
    a month Args: da: xarray dataset containing the seasonal forecast
    data

    Returns: da: xarray dataset with conversion from tprate to total
        precipitation in mm

    """
    da["precip"] = (
        da["tprate"] * da["time"].dt.days_in_month * 24 * 3600 * 1000
    )

    return da


def dates_per_leadtime(da):
    """
    Create a dict with one key-value pair per leadtime And compute the
    month for which the value was forecasted Args: da: xarray dataset
    containing the ecmwf seasonal forecast per publication date

    Returns: da_lead_dict: dict of xarray datasets with entry per
        leadtime

    """
    leadtimes = da["step"].values
    # create a dict with values per leadtime
    da_dict = {leadtime: da.sel(step=leadtime) for leadtime in leadtimes}
    # recompute time to be the month the forecast is valid, instead of
    # the publication month the forecast is monthly, so add leadtime in
    # months leadtime of 1 indicates the forecast is valid during the
    # publication month, so add leadtime-1 months to time i.e. the
    # outputted time is the start date the forecast applies to
    da_lead_dict = {
        leadtime: da_lt.assign_coords(
            time=da_lt["time"].values.astype("datetime64[M]")
            + np.array(leadtime - 1, "timedelta64[M]")
        )
        for leadtime, da_lt in da_dict.items()
    }

    return da_lead_dict


def convert_dict_to_da(da_dict):
    # compute months for which at least one forecast was available
    time = np.arange(
        da_dict[min(da_dict.keys())].time.values[0].astype("datetime64[M]"),
        da_dict[max(da_dict.keys())].time.values[-1].astype("datetime64[M]")
        + np.array(1, "timedelta64[M]"),
        dtype="datetime64[M]",
    )

    # include all dates for which a forecast was available for each
    # leadtime dataset even if not all those dates had a forecast for
    # the given leadtime needed to afterwards merge the different
    # leadtimes into one dataset
    da_lead_dict = {
        leadtime: da_lead.reindex({"time": time})
        for leadtime, da_lead in da_dict.items()
    }

    # convert to dataarray instead of dataset
    # needed to create a new dataarray
    # need to select one variable (precip) for dimensions to match
    data = np.array([da_lead["precip"] for da_lead in da_lead_dict.values()])

    # Create data array with all lead times, where time indicates the
    # start date during which the forecast was valid
    return xr.DataArray(
        data=data,
        # order of dims matters here!
        dims=["leadtime", "time", "number", "latitude", "longitude"],
        coords=dict(
            number=list(da_lead_dict.values())[
                0
            ].number,  # ensemble member number
            time=time,
            leadtime=list(da_lead_dict.keys()),
            longitude=list(da_lead_dict.values())[0].longitude,
            latitude=list(da_lead_dict.values())[0].latitude,
        ),
    )


def get_crps_ecmwf(
    observations: xr.DataArray,
    forecasts: xr.DataArray,
    normalization: str = None,
    thresh: float = None,
) -> pd.DataFrame:
    """
    Assumes there is no missing data in observations or forecasts
    :param observations: data-array with observed values
    :param forecasts: data-array with forecasted values
    :param normalization: (optional) can be None, a number, 'mean' or 'std',
    reanalysis metric to divide the CRPS
    :param threshold: (optional) only select values smaller or equal to
    this number
    :return: DataFrame with leadtime index containing the crps
    """
    leadtimes = forecasts.leadtime.values
    df_crps = pd.DataFrame(index=leadtimes)

    for leadtime in leadtimes:
        forecasts_lt = forecasts.sel(leadtime=leadtime).dropna(
            dim="time", how="all"
        )
        # make sure that time periods overlap, for calc_crps
        forecasts_lt = forecasts_lt.sel(
            time=slice(observations.time.min(), observations.time.max())
        )
        observations = observations.sel(
            time=slice(forecasts_lt.time.min(), forecasts_lt.time.max())
        )

        if thresh is not None:
            # cannot index on multidimensional arrays,
            # e.g. when having lon and lat
            # xr.where does work on multidimensional arrays
            observations = observations.where(observations <= thresh)
            forecasts_lt = forecasts_lt.where(observations <= thresh)

        crps = calc_crps(
            observations,
            forecasts_lt,
            normalization=normalization,
        )
        df_crps.loc[leadtime, "crps"] = crps
    return df_crps
