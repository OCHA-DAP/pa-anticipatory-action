import logging
import os
from pathlib import Path

import rasterio

# TODO: understand rioxarray vs xarray rioxarray seems to do a better
# job at correctly reading the data than xarray related to coordinates
# etc. at the same time rioxarray is still under a lot of development so
# not always working as stably as xarray.. so currently using both and
# hopefully in the future can do most with rioxarray
import rioxarray
import xarray as xr

from src.utils_general.raster_manipulation import (
    change_longitude_range,
    fix_calendar,
    invert_latlon,
)
from src.utils_general.utils import download_ftp

logger = logging.getLogger(__name__)


def download_nmme(config, date):
    """
    Download the NMME for date from their ftp server
    Args:
        config (Config): config for the drought indicator
        date (str): date of publication in YYYYMM format
    """
    NMME_dir = os.path.join(config.GLOBAL_DIR, config.NMME_DIR)
    Path(NMME_dir).mkdir(parents=True, exist_ok=True)
    NMME_filepath = os.path.join(
        NMME_dir, config.NMME_NC_FILENAME_RAW.format(date=date)
    )
    # assuming the files don't get updated so only download if doesn't
    # exist yet
    if not os.path.exists(NMME_filepath):
        download_ftp(
            config.NMME_FTP_URL_SEASONAL.format(date=date), NMME_filepath
        )

        # TODO: explore if can also open with rioxarray instead of
        # xarray. Getting an error with current settings
        nmme_ds = xr.open_dataset(NMME_filepath, decode_times=False)

        # generally nmme coordinates are not inverted, but this is a
        # double check
        nmme_ds = invert_latlon(nmme_ds)
        nmme_ds = change_longitude_range(nmme_ds)
        nmme_ds = fix_calendar(nmme_ds, timevar="target")
        nmme_ds = fix_calendar(nmme_ds, timevar="initial_time")
        nmme_ds = xr.decode_cf(nmme_ds)
        NMME_filepath_crs = os.path.join(
            NMME_dir,
            config.NMME_NC_FILENAME_CRS.format(
                date=date, tercile=config.LOWERTERCILE
            ),
        )

        # strange things happen when just overwriting the file, so
        # delete it first if it already exists
        if os.path.exists(NMME_filepath_crs):
            os.remove(NMME_filepath_crs)
        # crs is only saved correctly when saving one var. Don't
        # understand why exactly but hence, only save one tercile
        nmme_ds[config.NMME_LOWERTERCILE].rio.write_crs("EPSG:4326").to_netcdf(
            NMME_filepath_crs
        )


def get_nmme_data(config, date, download=False):
    """
    Read the NMME NetCDF data as xarray dataset Args: config (Config):
    config for the drought indicator date (str): date of publication in
    YYYYMM format download (bool): if True, download data

    Returns: nmme_ds (xarray dataset): dataset continaing the
        information in the netcdf file transform (numpy array): affine
        transformation of the dataset based on its CRS
    """
    if download:
        download_nmme(config, date)
    NMME_filepath = os.path.join(
        config.GLOBAL_DIR,
        config.NMME_DIR,
        config.NMME_NC_FILENAME_CRS.format(
            date=date, tercile=config.LOWERTERCILE
        ),
    )

    nmme_ds = rioxarray.open_rasterio(NMME_filepath)
    # nmme's data comes in fractions while other sources come in
    # percentages, so convert to percentages
    nmme_ds[config.NMME_LOWERTERCILE] = nmme_ds[config.NMME_LOWERTERCILE] * 100
    nmme_ds = nmme_ds.rename(
        {
            config.NMME_LON: config.LONGITUDE,
            config.NMME_LAT: config.LATITUDE,
            config.NMME_LOWERTERCILE: config.LOWERTERCILE,
        }
    )

    with rasterio.open(NMME_filepath) as src:
        transform = src.transform

    return nmme_ds, transform
