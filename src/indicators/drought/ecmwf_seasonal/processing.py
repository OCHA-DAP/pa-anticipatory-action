import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.utils_general.raster_manipulation import compute_raster_statistics
from src.utils_general.statistics import calc_crps

logger = logging.getLogger(__name__)

ECMWF_API_FILEPATH = (
    "private/processed/{country_iso3}/ecmwf/"
    "seasonal-monthly-individual-members/prate/"
    "mwi_seasonal-monthly-individual-members_prate.nc"
)


def get_ecmwf_forecast(
    country_iso3: str, version: int = 5, source_cds: bool = True, **kwargs
) -> xr.Dataset:
    """
    Retrieve the processed dataset with the forecast for each
    publication date and corresponding lead times
    :param country_iso3: iso3 code of the country of interest
    :param version: version of the ecmwf model to use
    :param source_cds: whether the data comes from CDS, or the ECMWF API
    :param kwargs: other args that can be given to EcmwfSeasonalForecast()
    :return: dataset with the ecmwf forecasts
    """
    """
     Args: version: version
    of forecast model that was used (only changes once every couple of
    years)
    """
    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast(**kwargs)
    if source_cds:
        ds_ecmwf_forecast = ecmwf_forecast.read_processed_dataset(
            country_iso3=country_iso3,
            version=version,
        )
    else:
        dataset_path = Path(
            os.environ["AA_DATA_DIR"]
        ) / ECMWF_API_FILEPATH.format(country_iso3=country_iso3)
        ds_ecmwf_forecast = xr.load_dataset(dataset_path)
    ds_ecmwf_forecast = convert_tprate_precipitation(ds_ecmwf_forecast)

    return ds_ecmwf_forecast


def get_ecmwf_forecast_by_leadtime(
    country_iso3, version: int = 5, source_cds: bool = True, **kwargs
):
    """
    Reshape dataset to have the time variable as the month during the
    forecast was valid instead of the month the forecast was published
    Args: version: version of forecast model that was used (only changes
    once every couple of years)
    :param country_iso3: iso3 code of country of interest
    :param version: version of the ecmwf model to use
    :param source_cds: whether the data comes from CDS, or the ECMWF API
    :param kwargs: other args that can be given to get_ecmwf_forecast()
    :return: dataset with data, grouped by leadtime
    """

    ds_ecmwf_forecast = get_ecmwf_forecast(
        country_iso3=country_iso3,
        version=version,
        source_cds=source_cds,
        **kwargs,
    )
    ds_ecmwf_forecast_dict = dates_per_leadtime(ds_ecmwf_forecast)
    return convert_dict_to_da(ds_ecmwf_forecast_dict).dropna(
        dim="time", how="all"
    )


def get_stats_filepath(
    iso3: str,
    config: Config,
    date: datetime,
    adm_level: int,
    use_unrounded_area_coords: bool,
    resolution: float = None,
    all_touched: bool = False,
    version: int = None,
) -> Path:
    """
    Retrieve the path to the statsfile with the given parameters
    :param iso3: iso3 code of the country of interest
    :param config: Config() instance
    :param date: the date of interest
    If None, data is in original resolution
    :param adm_level: the admin level the data is aggregated to
    :param use_unrounded_area_coords: Generally meant to be False,
        needed for backward compatibility with some historical data.
        If True, no rounding to the coordinates will be done which results in
        a shift in data which is interpolated
    :param resolution: resolution the data is changed to.
    :param all_touched: if True, include all cells touching the region
    If False, only include cells with their centre within the region
    :param version: ecmwf model version that is used,
    if None the default version will be used
    :return: path to the stats file
    """

    if version is None:
        version = config.DEFAULT_VERSION

    filename = f"{iso3.lower()}_seasonal-monthly-single-levels_v{version}"
    if use_unrounded_area_coords:
        filename += "_unrounded-coords"
    if resolution is not None:
        filename += f"_res{resolution}"
    if all_touched:
        filename += "_all-touched"
    filename += f"_{date.year}_{date.month}_adm{adm_level}_stats.csv"

    country_data_processed_dir = (
        Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3
    )
    ecmwf_processed_dir = country_data_processed_dir / config.ECMWF_DIR

    stats_dir = ecmwf_processed_dir / "seasonal-monthly-single-levels"
    if use_unrounded_area_coords:
        stats_dir = stats_dir / "unrounded-coords"

    return stats_dir / filename


def compute_stats_per_admin(
    iso3,
    adm_level: int = 1,
    pcode_col: str = "ADM1_PCODE",
    add_col: List[str] = None,
    use_cache: bool = True,
    resolution: float = None,
    date_list: List[str] = None,
    source_cds: bool = True,
    use_unrounded_area_coords: bool = False,
    all_touched: bool = False,
):
    """
    compute several statistics on admin level retrieved
    from the raster data and save these to a file
    :param iso3: iso3 code of the country of interest
    :param adm_level: admin level to aggregate the data to
    :param pcode_col: column in the shapefile that contains the pcode
    :param add_col: other columns that should be added from the shapefile
    :param use_cache: if True, don't update the file if it already exists
    :param resolution: Change the resolution of the longitude and
    latitude to this number
    If None, don't change the resolution of the original data
    :param date_list: list of dates to compute stats for. If None, the stats
    will be computed for all dates in ds
    :param source_cds: whether the data comes from CDS, or the ECMWF API
    :param use_unrounded_area_coords: Generally meant to be False,
        needed for backward compatibility with some historical data.
        If True, no rounding to the coordinates will be done which results in
        shifted and interpolated data
    :param all_touched: if True, include all cells touching the region
    If False, only include cells with their centre within the region
    """
    config = Config()
    parameters = config.parameters(iso3)

    country_data_raw_dir = os.path.join(
        config.DATA_DIR, config.PUBLIC_DIR, config.RAW_DIR, iso3
    )
    adm_boundaries_path = os.path.join(
        country_data_raw_dir,
        config.SHAPEFILE_DIR,
        parameters[f"path_admin{adm_level}_shp"],
    )

    # read the forecasts
    ds = get_ecmwf_forecast_by_leadtime(
        iso3,
        source_cds=source_cds,
        use_unrounded_area_coords=use_unrounded_area_coords,
    )
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)

    if add_col is None:
        add_col = []
    # loop over dates
    if date_list is None:
        date_list = ds.time.values
    for date in date_list:
        date_dt = pd.to_datetime(date)
        output_path = get_stats_filepath(
            iso3=iso3,
            config=config,
            date=date_dt,
            adm_level=adm_level,
            use_unrounded_area_coords=use_unrounded_area_coords,
            resolution=resolution,
            all_touched=all_touched,
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
            df_lt_list = []
            for lt in ds_sel.leadtime.values:
                ds_sel_lt = ds_sel.sel(leadtime=lt)
                # reproject only working on 2D&3D arrays
                # so only do after selecting the date and leadtime..
                if resolution is not None:
                    ds_sel_lt = ds_sel_lt.rio.reproject(
                        ds_sel_lt.rio.crs,
                        resolution=resolution,
                        resampling=Resampling.nearest,
                        nodata=np.nan,
                    )
                    # we use longitude and latitude in other places
                    # so stick to those namings
                    ds_sel_lt = ds_sel_lt.rename(
                        {"x": "longitude", "y": "latitude"}
                    )
                df_lt = compute_raster_statistics(
                    gdf_adm,
                    pcode_col,
                    ds_sel_lt,
                    lon_coord="longitude",
                    lat_coord="latitude",
                    all_touched=all_touched,
                )
                df_lt_list.append(df_lt)
            df = pd.concat(df_lt_list)
            df = df.merge(
                gdf_adm[[pcode_col] + add_col], on=pcode_col, how="left"
            )
            df["date"] = date_dt
            df.to_csv(output_path, index=False)


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
