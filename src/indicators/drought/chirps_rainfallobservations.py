import logging
import os
import urllib
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp

logger = logging.getLogger(__name__)


def download_chirps_daily(config, year, resolution="25", write_crs=False):
    """
    Download the CHIRPS data for year from their ftp server Args: config
    (Config): config for the drought indicator year (str or int): year
    for which the data should be downloaded in YYYY format resolution
    (str): resolution of the data to be downloaded. Can be 25 or 05
    """
    chirps_dir = os.path.join(config.GLOBAL_DIR, config.CHIRPS_DIR)
    Path(chirps_dir).mkdir(parents=True, exist_ok=True)
    chirps_filepath = os.path.join(
        chirps_dir,
        config.CHIRPS_NC_FILENAME_RAW.format(year=year, resolution=resolution),
    )
    # TODO: decide if only download if file doesn't exist. Not sure if
    # ever gets updated
    today = datetime.now()
    # often data is uploaded at a later date, so update those files that
    # are in the current year and in the previous year if at the start
    # of new year
    year_update = (today - timedelta(days=60)).year
    if not os.path.exists(chirps_filepath) or int(year) >= year_update:
        try:
            if os.path.exists(chirps_filepath):
                os.remove(chirps_filepath)
            download_ftp(
                config.CHIRPS_FTP_URL_GLOBAL_DAILY.format(
                    year=year, resolution=resolution
                ),
                chirps_filepath,
            )
            if write_crs:
                # Xarray (python) expects a crs and cannot read this for
                # some undefined reason from the current file, so for
                # this purpose save it as a separate file that includes
                # the crs In R when working with bricks, this issue
                # doesn't seem to appear
                # ds=rioxarray.open_rasterio(chirps_filepath)
                ds = xr.load_dataset(chirps_filepath)
                chirps_filepath_crs = os.path.join(
                    chirps_dir,
                    config.CHIRPS_NC_FILENAME_CRS.format(
                        year=year, resolution=resolution
                    ),
                )
                if os.path.exists(chirps_filepath_crs):
                    os.remove(chirps_filepath_crs)
                ds.rio.write_crs("EPSG:4326").to_netcdf(chirps_filepath_crs)
        except urllib.error.HTTPError as e:
            chirps_url = config.CHIRPS_FTP_URL_GLOBAL_DAILY.format(
                year=year, resolution=resolution
            )
            logging.error(
                f"{e}. Date might be later than last reported datapoint."
                f" URL:{chirps_url}"
            )


def download_chirps_monthly(
    config,
    use_cache: bool = True,
):
    """
    Download global chirps dataset containing monthly entries and save
    to file Args: config (Config): config for the drought indicator
    use_cache: if True, don't download if filename already exists
    """
    # If caching is on and file already exists, don't download again
    if use_cache and config.CHIRPS_MONTHLY_RAW_PATH.exists():
        logger.debug(
            f"{config.CHIRPS_MONTHLY_RAW_PATH} already exists and cache is set"
            " to True, skipping"
        )
        return config.CHIRPS_MONTHLY_RAW_PATH
    Path(config.CHIRPS_MONTHLY_RAW_PATH.parent).mkdir(
        parents=True, exist_ok=True
    )
    logger.debug(f"Querying for {config.CHIRPS_MONTHLY_RAW_PATH}...")
    download_ftp(
        config.CHIRPS_FTP_URL_GLOBAL_MONTHLY, config.CHIRPS_MONTHLY_RAW_PATH
    )
    logger.debug(
        f"...successfully downloaded {config.CHIRPS_MONTHLY_RAW_PATH}"
    )
    return config.CHIRPS_MONTHLY_RAW_PATH


def clip_chirps_monthly_bounds(config, country_iso3: str, use_cache=True):
    """Clip the global chirps dataset to the boundaries of country_name
    This will enable faster processing Clipping can take max half an
    hour."""
    parameters = config.parameters(country_iso3)
    adm0_bound_path = (
        Path(config.DATA_DIR)
        / config.PUBLIC_DIR
        / config.RAW_DIR
        / country_iso3
        / config.SHAPEFILE_DIR
        / parameters["path_admin0_shp"]
    )

    chirps_monthly_country_filepath = get_filepath_chirps_monthly(
        country_iso3, config
    )
    if use_cache and chirps_monthly_country_filepath.exists():
        logger.debug(
            f"{chirps_monthly_country_filepath} already exists and cache is"
            " set to True, skipping"
        )
        return chirps_monthly_country_filepath

    Path(chirps_monthly_country_filepath.parent).mkdir(
        parents=True, exist_ok=True
    )
    logger.debug(
        f"Clipping global data to {config.CHIRPS_MONTHLY_RAW_PATH}..."
    )
    # would like to rioxarray but seems slower/crashing with clip
    ds = xr.load_dataset(config.CHIRPS_MONTHLY_RAW_PATH).rio.write_crs(
        "EPSG:4326"
    )
    gdf_adm1 = gpd.read_file(adm0_bound_path)
    ds_country = ds.rio.set_spatial_dims(
        x_dim="longitude", y_dim="latitude"
    ).rio.clip(gdf_adm1["geometry"], ds.rio.crs, all_touched=True)
    ds_country.to_netcdf(chirps_monthly_country_filepath)
    logger.debug(
        "...successfully saved clipped data to"
        f" {config.CHIRPS_MONTHLY_RAW_PATH}"
    )
    return chirps_monthly_country_filepath


def get_filepath_chirps_monthly(country_iso3: str, config: Config):
    chirps_country_dir = (
        Path(config.DATA_DIR)
        / config.PUBLIC_DIR
        / config.PROCESSED_DIR
        / country_iso3
        / config.CHIRPS_DIR
    )
    chirps_monthly_country_dir = chirps_country_dir / config.CHIRPS_MONTHLY_DIR

    chirps_monthly_country_filepath = (
        chirps_monthly_country_dir
        / config.CHIRPS_MONTHLY_COUNTRY_FILENAME.format(
            country_iso3=country_iso3
        )
    )

    return chirps_monthly_country_filepath


def get_filepath_seasonal_lowertercile_raster(
    country_iso3: str, config: Config
):
    chirps_country_dir = (
        Path(config.DATA_DIR)
        / config.PUBLIC_DIR
        / config.PROCESSED_DIR
        / country_iso3
        / config.CHIRPS_DIR
    )

    chirps_seasonal_country_dir = (
        chirps_country_dir / config.CHIRPS_SEASONAL_DIR
    )

    chirps_seasonal_lowertercile_country_filepath = (
        chirps_seasonal_country_dir
        / config.CHIRPS_SEASONAL_LOWERTERCILE_COUNTRY_FILENAME.format(
            country_iso3=country_iso3
        )
    )

    chirps_seasonal_tercile_bounds_country_filepath = (
        chirps_seasonal_country_dir
        / config.CHIRPS_SEASONAL_TERCILE_BOUNDS_FILENAME.format(
            country_iso3=country_iso3
        )
    )

    return (
        chirps_seasonal_lowertercile_country_filepath,
        chirps_seasonal_tercile_bounds_country_filepath,
    )


def compute_seasonal_lowertercile_raster(
    config,
    country_iso3: str,
    use_cache: bool = True,
):
    # number of months that is considered a season
    seas_len = 3

    chirps_monthly_country_filepath = get_filepath_chirps_monthly(
        country_iso3, config
    )
    (
        chirps_seasonal_lowertercile_country_filepath,
        chirps_seaonal_tercile_bounds_country_filepath,
    ) = get_filepath_seasonal_lowertercile_raster(country_iso3, config)

    if use_cache and chirps_seasonal_lowertercile_country_filepath.exists():
        logger.debug(
            f"{chirps_seasonal_lowertercile_country_filepath} already exists"
            " and cache is set to True, skipping"
        )
        return chirps_seasonal_lowertercile_country_filepath

    Path(chirps_seasonal_lowertercile_country_filepath.parent).mkdir(
        parents=True, exist_ok=True
    )
    logger.debug("Computing lower tercile values...")
    ds = xr.load_dataset(chirps_monthly_country_filepath)
    # compute the rolling sum over three month period. Rolling sum works
    # backwards, i.e. value for month 3 is sum of month 1 till 3. So
    # month==1 is NDJ season
    ds_season = (
        ds.rolling(time=seas_len, min_periods=seas_len)
        .sum()
        .dropna(dim="time", how="all")
    )
    # define the years that are used to define the climatology. We use
    # 1982-2010 since this is also the period used by IRI's seasonal
    # forecasts see
    # https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/
    ds_season_climate = ds_season.sel(
        time=ds_season.time.dt.year.isin(range(1982, 2011))
    )
    # compute the thresholds for the lower tercile, i.e. below average,
    # per season since we computed a rolling sum, each month represents
    # a season
    ds_season_climate_quantile = ds_season_climate.groupby(
        ds_season_climate.time.dt.month
    ).quantile(0.33)
    # save below tercile boundaries
    ds_season_climate_quantile.to_netcdf(
        chirps_seaonal_tercile_bounds_country_filepath
    )
    # determine the raster cells that have below-average precipitation,
    # other cells are set to -666
    list_ds_seass = []
    for s in np.unique(ds_season.time.dt.month):
        ds_seas_sel = ds_season.sel(time=ds_season.time.dt.month == s)
        # keep original values of cells that are either nan or have
        # below average precipitation, all others are set to -666
        ds_seas_below = ds_seas_sel.where(
            (ds_seas_sel.isnull())
            | (ds_seas_sel <= ds_season_climate_quantile.sel(month=s)),
            -666,
        )
        list_ds_seass.append(ds_seas_below)
    ds_season_below = xr.concat(list_ds_seass, dim="time")
    ds_season_below.to_netcdf(chirps_seasonal_lowertercile_country_filepath)
    return chirps_seasonal_lowertercile_country_filepath


def get_chirps_data_daily(config, year, resolution="25", download=False):
    """
    Load CHIRP's NetCDF file as xarray dataset Args: config (Config):
    config for the drought indicator year (str or int): year for which
    the data should be loaded in YYYY format resolution (str):
    resolution of the data to be downloaded. Can be 25 or 05 download
    (bool): if True, download data

    Returns: icpac_ds (xarray dataset): dataset continaing the
        information in the netcdf file transform (numpy array): affine
        transformation of the dataset based on its CRS
    """

    if download:
        download_chirps_daily(config, year, resolution)

    chirps_filepath_crs = os.path.join(
        config.GLOBAL_DIR,
        config.CHIRPS_DIR,
        config.CHIRPS_NC_FILENAME_CRS.format(year=year, resolution=resolution),
    )
    # TODO: would prefer rioxarray but crashes when clipping
    ds = xr.load_dataset(chirps_filepath_crs)
    # ds = rioxarray.open_rasterio(chirps_filepath_crs)
    ds = ds.rename(
        {
            config.CHIRPS_LON: config.LONGITUDE,
            config.CHIRPS_LAT: config.LATITUDE,
        }
    )

    with rasterio.open(chirps_filepath_crs) as src:
        transform = src.transform

    return ds, transform


def get_chirps_data_monthly(
    config,
    country_iso3: str,
    download: bool = True,
    process: bool = True,
    use_cache: bool = True,
):
    if download:
        download_chirps_monthly(config, use_cache)
    if process:
        clip_chirps_monthly_bounds(
            config=config,
            country_iso3=country_iso3,
            use_cache=use_cache,
        )
