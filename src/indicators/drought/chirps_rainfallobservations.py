import logging
import os
import sys
import urllib
from calendar import month_name
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import geopandas as gpd
import rasterio
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp

logger = logging.getLogger(__name__)

_SEASONAL_TERCILE_FILENAME = "{country_iso3}_chirps_seasonal_terciles.nc"
_SEASONAL_TERCILE_BOUNDS_FILENAME = (
    "{country_iso3}_chirps_seasonal_terciles_bounds.nc"
)


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


# leaving this function for now, since it is being called
# by some notebooks
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

    return chirps_seasonal_lowertercile_country_filepath


def get_filepath_seasonal_tercile_raster(country_iso3: str, config: Config):
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

    chirps_seasonal_tercile_filepath = (
        chirps_seasonal_country_dir
        / _SEASONAL_TERCILE_FILENAME.format(country_iso3=country_iso3)
    )

    chirps_seasonal_tercile_bounds_filepath = (
        chirps_seasonal_country_dir
        / _SEASONAL_TERCILE_BOUNDS_FILENAME.format(country_iso3=country_iso3)
    )

    return (
        chirps_seasonal_tercile_filepath,
        chirps_seasonal_tercile_bounds_filepath,
    )


def compute_seasonal_tercile_raster(
    config,
    country_iso3: str,
    use_cache: bool = True,
    climatology_min_year: int = 1982,
    climatology_max_year: int = 2011,
):
    # number of months that is considered a season
    seas_len = 3
    end_month_season_mapping = {
        m: "".join(
            [month_name[(m - i) % 12 + 1][0] for i in range(seas_len, 0, -1)]
        )
        for m in range(1, 13)
    }

    chirps_monthly_country_filepath = get_filepath_chirps_monthly(
        country_iso3, config
    )
    (
        chirps_seasonal_tercile_filepath,
        chirps_seasonal_tercile_bounds_filepath,
    ) = get_filepath_seasonal_tercile_raster(country_iso3, config)

    if use_cache and chirps_seasonal_tercile_filepath.exists():
        logger.debug(
            f"{chirps_seasonal_tercile_filepath} already exists"
            " and cache is set to True, skipping"
        )
        return chirps_seasonal_tercile_filepath

    Path(chirps_seasonal_tercile_filepath.parent).mkdir(
        parents=True, exist_ok=True
    )
    logger.debug("Computing lower tercile values...")
    # for some unknown reason the rolling sum takes very long to
    # compute when using rioxarray so sticking to xarray for now
    da = xr.load_dataset(chirps_monthly_country_filepath).precip
    # compute the rolling sum over three month period. Rolling sum works
    # backwards, i.e. value for month 3 is sum of month 1 till 3. So
    # month==1 is NDJ season
    da_season = (
        da.rolling(time=seas_len, min_periods=seas_len)
        .sum()
        .dropna(dim="time", how="all")
    )
    da_season = da_season.assign_coords(
        season=(
            "time",
            [
                end_month_season_mapping[end_month]
                for end_month in da_season.time.dt.month.values
            ],
        )
    )
    # define the years that are used to define the climatology. We use
    # 1982-2010 since this is also the period used by IRI's seasonal
    # forecasts see
    # https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/
    da_season_climate = da_season.sel(
        time=da_season.time.dt.year.isin(
            range(climatology_min_year, climatology_max_year)
        )
    )

    # compute the thresholds for the lower and upper tercile per season
    da_season_climate_quantile = _compute_tercile_bounds(
        da_season_climate.groupby("season")
    )

    # save tercile boundaries
    da_season_climate_quantile.to_netcdf(
        chirps_seasonal_tercile_bounds_filepath
    )
    da_bn, da_no, da_an = _compute_tercile_category(
        da_season, da_season_climate_quantile
    )
    da_season_terc = da_season.to_dataset(name="precip").assign(
        {
            "below_normal": da_bn.drop("quantile"),
            "normal": da_no,
            "above_normal": da_an.drop("quantile"),
        }
    )

    da_season_terc.to_netcdf(chirps_seasonal_tercile_filepath)
    return chirps_seasonal_tercile_filepath


def _compute_tercile_bounds(da):
    """compute lower and upper tercile bounds."""
    # rename quantiles such that don't have to select on floats later on
    return da.quantile([1 / 3, 2 / 3], skipna=True).assign_coords(
        {"quantile": ["lower_bound", "upper_bound"]}
    )


def _compute_tercile_category(da, da_quantile):
    da_bn = xr.where(
        da <= da_quantile.sel(quantile="lower_bound"), True, False
    )
    da_an = xr.where(
        da >= da_quantile.sel(quantile="upper_bound"), True, False
    )
    da_no = xr.where(
        (da > da_quantile.sel(quantile="lower_bound"))
        & (da < da_quantile.sel(quantile="upper_bound")),
        True,
        False,
    )
    return da_bn, da_no, da_an


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


def clip_chirps_daily(
    iso3,
    config,
    resolution,
) -> Path:
    """
    Combine the daily CHIRPS data.

    Data is downloaded per dyear, combine to one file.

    Returns
    -------
    Path to processed NetCDF file

    """
    parameters = config.parameters(iso3)
    adm0_bound_path = (
        Path(config.DATA_DIR)
        / config.PUBLIC_DIR
        / config.RAW_DIR
        / iso3
        / config.SHAPEFILE_DIR
        / parameters["path_admin0_shp"]
    )
    # get path structure with publication date as wildcard
    raw_path = _get_raw_path_daily(config, year=None, resolution=resolution)
    filepath_list = list(raw_path.parents[0].glob(raw_path.name))
    # output_filepath = _get_processed_path_daily()
    # output_filepath.parent.mkdir(exist_ok=True, parents=True)
    gdf_adm0 = gpd.read_file(adm0_bound_path)
    output_path = _get_processed_path_country_daily(
        iso3=iso3, config=config, year=None, resolution=resolution
    )
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    with xr.open_mfdataset(filepath_list) as ds:
        ds_country = (
            ds.rio.write_crs("EPSG:4326")
            .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            .rio.clip(gdf_adm0["geometry"], all_touched=True)
        )
        ds_country.attrs["included_files"] = [f.stem for f in filepath_list]
        ds_country.to_netcdf(output_path)
    return output_path


def load_chirps_daily_clipped(
    iso3, config, resolution, year: Union[int, None] = None
):
    ds = xr.load_dataset(
        _get_processed_path_country_daily(
            iso3=iso3, config=config, resolution=resolution, year=year
        )
    )
    return ds.rio.write_crs("EPSG:4326", inplace=True)


def _get_raw_path_daily(config, year: Union[int, None], resolution: int):
    """Get the path to the raw api data for a given `date_forecast`."""
    chirps_dir = Path(config.GLOBAL_DIR) / config.CHIRPS_DIR
    if year is None:
        year_str = "*"
    else:
        year_str = year
    chirps_filepath = os.path.join(
        chirps_dir,
        config.CHIRPS_NC_FILENAME_RAW.format(
            year=year_str, resolution=resolution
        ),
    )
    return chirps_dir / chirps_filepath


def _get_processed_path_country_daily(
    iso3, config, year: Union[int, None], resolution: int
):
    """Get the path to the raw api data for a given `date_forecast`."""
    chirps_dir = (
        Path(config.DATA_DIR)
        / config.PUBLIC_DIR
        / config.PROCESSED_DIR
        / iso3
        / "chirps"
        / "daily"
    )
    chirps_filepath = f"{iso3}_chirps_daily"
    if year is not None:
        chirps_filepath += f"_{year}"
    chirps_filepath += f"_p{resolution}.nc"
    return chirps_dir / chirps_filepath


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
