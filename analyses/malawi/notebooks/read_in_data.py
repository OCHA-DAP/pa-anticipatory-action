import sys
import os
from pathlib import Path
import geopandas as gpd
from rasterstats import zonal_stats

import numpy as np
import pandas as pd
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought import ecmwf_seasonal
from src.malawi import get_ecmwf_seasonal_data as gesd
from src.indicators.drought.config import Config

#TODO: not sure which functions to include here and which in the ecmwf_seasonal file

def get_ecmwf_forecast(version: int = 5):
    """
    Retrieve the processed dataset with the forecast for each publication date and corresponding lead times
    Args:
        version: version of forecast model that was used (only changes once every couple of years)
    """
    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast()
    ds_ecmwf_forecast = ecmwf_forecast.read_processed_dataset(
            country_name=gesd.COUNTRY_NAME,
            country_iso3=gesd.COUNTRY_ISO3,
            version=version,
        )
    ds_ecmwf_forecast = convert_tprate_precipitation(ds_ecmwf_forecast)

    return ds_ecmwf_forecast

def get_ecmwf_forecast_by_leadtime(version: int = 5):
    """
    Reshape dataset to have the time variable as the month during the forecast was valid
    instead of the month the forecast was published
    Args:
        version: version of forecast model that was used (only changes once every couple of years)

    Returns:
        dataset with valid month per publication data-leadtime
    """
    ds_ecmwf_forecast = get_ecmwf_forecast(version)
    ds_ecmwf_forecast_dict = dates_per_leadtime(ds_ecmwf_forecast)
    return convert_dict_to_da(ds_ecmwf_forecast_dict)

#TODO: not sure if this is the best structure, should it instead be inside a class?
def compute_stats_per_admin(country,adm_level=1):
    config = Config()
    parameters = config.parameters(country)

    country_data_raw_dir = os.path.join(config.DATA_DIR, config.RAW_DIR, country)
    country_data_processed_dir = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, country)
    adm_boundaries_path = os.path.join(country_data_raw_dir, config.SHAPEFILE_DIR, parameters[f"path_admin{adm_level}_shp"])

    ds = get_ecmwf_forecast_by_leadtime()
    ds=ds.sel(time="2020-01")
    df=compute_zonal_stats(ds,ds.rio.transform(),adm_boundaries_path,parameters[f"shp_adm{adm_level}c"])

    df.to_csv(os.path.join(country_data_processed_dir,"ecmwf_seasonal_data",f"{parameters['iso3_code']}_seasonal-monthly-single-levels_v5_adm{adm_level}_stats.csv"))

#TODO: create function to retrieve the stats file

def compute_zonal_stats(ds, raster_transform, adm_path,adm_col):
    # compute statistics on level in adm_path for all dates in ds
    df_list = []
    for date in ds.time.values:
        for leadtime in ds.leadtime.values:
            for number in ds.number.values:
                df = gpd.read_file(adm_path)[[adm_col,"geometry"]]
                ds_date = ds.sel(time=date,number=number,leadtime=leadtime)

                df[["mean_cell", "max_cell", "min_cell"]] = pd.DataFrame(
                    zonal_stats(vectors=df, raster=ds_date.values, affine=raster_transform, nodata=np.nan))[
                    ["mean", "max", "min"]]

                percentile_list = [10, 20, 30, 40, 50, 60, 70, 80]
                df[[f"percentile_{str(p)}" for p in percentile_list]] = pd.DataFrame(
                    zonal_stats(vectors=df, raster=ds_date.values, affine=raster_transform, nodata=np.nan,
                                stats=" ".join([f"percentile_{str(p)}" for p in percentile_list])))[
                    [f"percentile_{str(p)}" for p in percentile_list]]

                df["date"] = pd.to_datetime(date)
                df["number"] = number
                df["leadtime"] = leadtime

                df_list.append(df)
            df_hist = pd.concat(df_list)
            df_hist = df_hist.sort_values(by="date")
            #drop the geometry column, else csv becomes huge
            df_hist=df_hist.drop("geometry",axis=1)

    return df_hist



    return df_hist


def convert_tprate_precipitation(da):
    """
    The ECMWF seasonal forecast reports precipitation as tprate, which is in meter/second.
    To convert this to the total precipitation in a month in meter, we multiply the tprate by the number of seconds in a month
    Thereafter we multiply by 1000 to get the total millimeters in a month
    Args:
        da: xarray dataset containing the seasonal forecast data

    Returns:
        da: xarray dataset with conversion from tprate to total precipitation in mm

    """
    da["precip"] = da["tprate"] * da["time"].dt.days_in_month * 24 * 3600 * 1000

    return da

def dates_per_leadtime(da):
    """
    Create a dict with one key-value pair per leadtime
    And compute the month for which the value was forecasted
    Args:
        da: xarray dataset containing the ecmwf seasonal forecast per publication date

    Returns:
        da_lead_dict: dict of xarray datasets with entry per leadtime

    """
    leadtimes = da["step"].values
    #create a dict with values per leadtime
    da_dict = {
        leadtime: da.sel(step=leadtime)
        for leadtime in leadtimes
    }
    #recompute time to be the month the forecast is valid, instead of the publication month
    #the forecast is monthly, so add leadtime in months
    #leadtime of 1 indicates the forecast is valid during the publication month, so add leadtime-1 months to time
    #i.e. the outputted time is the start date the forecast applies to
    da_lead_dict = {
        leadtime: da_lt.assign_coords(
            time=da_lt["time"].values.astype('datetime64[M]') + np.array(leadtime - 1, 'timedelta64[M]')
        )
        for leadtime, da_lt in da_dict.items()
    }

    return da_lead_dict


def convert_dict_to_da(da_dict):
    #compute months for which at least one forecast was available
    time = np.arange(
        da_dict[min(da_dict.keys())].time.values[0].astype('datetime64[M]'),
        da_dict[max(da_dict.keys())].time.values[-1].astype('datetime64[M]')
        + np.array(1, "timedelta64[M]"),
        dtype="datetime64[M]",
    )

    #include all dates for which a forecast was available for each leadtime dataset
    #even if not all those dates had a forecast for the given leadtime
    #needed to afterwards merge the different leadtimes into one dataset
    da_lead_dict = {
        leadtime: da_lead.reindex({"time": time})
        for leadtime, da_lead in da_dict.items()
    }

    #convert to dataarray instead of dataset
    #needed to create a new dataarray
    #need to select one variable (precip) for dimensions to match
    data = np.array([da_lead["precip"] for da_lead in da_lead_dict.values()])

    # Create data array with all lead times, where time indicates the start date during which the forecast was valid
    return xr.DataArray(
        data=data,
        #order of dims matters here!
        dims=["leadtime", "time", "number", "latitude", "longitude"],
        coords=dict(
            number=list(da_lead_dict.values())[0].number,  # ensemble member number
            time=time,
            leadtime=list(da_lead_dict.keys()),
            longitude=list(da_lead_dict.values())[0].longitude,
            latitude=list(da_lead_dict.values())[0].latitude
        ),
    )