import logging
import os
from datetime import datetime
from pathlib import Path

import requests
import xarray as xr

from src.indicators.drought.config import Config
from src.utils_general.raster_manipulation import (
    change_longitude_range,
    fix_calendar,
    invert_latlon,
)

logger = logging.getLogger(__name__)


def download_iri(
    iri_auth: str, output_path: str, url: str, chunk_size: int = 128
):
    """
    Download the IRI seasonal tercile forecast as NetCDF file from the
    url as given in config.IRI_URL Saves the file to the path as defined
    in config
    :param iri_auth: iri key for authentication. An account is
    needed to get this key config. For an account this key might
    be changed over time, so might need to update it regularly
    :param config: config for the drought indicator
    :param chunk_size: number of bytes to download at once
    """
    # strange things happen when just overwriting the file, so delete it
    # first if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # have to authenticate by using a cookie
    cookies = {
        "__dlauth_id": iri_auth,
    }
    # TODO fix/understand missing certificate verification warning For
    # now leaving it as it is since it is a trustable site and we
    # couldn't figure how to improve it
    logger.info("Downloading IRI NetCDF file. This might take some time")
    response = requests.get(url, cookies=cookies, verify=False)
    with open(output_path, "wb") as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def clean_iri_data(
    input_path: str,
    output_path: str,
    config: Config(),
):
    """
    clean up some flaws of IRI's format and write the cleaned
    data to a new file
    :param input_path: path to netcdf containing the raw data
    :param output_path: path to which the cleaned data should be written
    :param config: config for the drought indicator
    """

    # rioxarray cannot open more than 3D datasets so use xarray
    # the nc contains two bands, prob and C.Still not sure what C is
    # used for but couldn't discover useful information in it and will
    # give an error if trying to read both (cause C is also a variable
    # in prob) the date format is formatted as months since 1960. In
    # principle xarray can convert this type of data to datetime, but
    # due to a wrong naming of the calendar variable it cannot do this
    # automatically Thus first load with decode_times=False and then
    # change the calendar variable and decode the months later
    iri_ds = xr.load_dataset(
        input_path, decode_times=False, drop_variables="C"
    )

    # often IRI latitude is flipped so check for that and invert if needed
    iri_ds = invert_latlon(
        iri_ds, lon_coord=config.IRI_LON, lat_coord=config.IRI_LAT
    )
    iri_ds = change_longitude_range(iri_ds, lon_coord=config.IRI_LON)

    # fix dates
    iri_ds = fix_calendar(iri_ds, timevar="F")
    iri_ds = xr.decode_cf(iri_ds)

    # rename to standardized lon and lat name
    iri_ds = iri_ds.rename(
        {config.IRI_LON: config.LONGITUDE, config.IRI_LAT: config.LATITUDE}
    )

    if os.path.exists(output_path):
        os.remove(output_path)
    # The Coordinate Reference System (CRS) is EPSG:4326 but this isn't
    # included in the attributes, so add
    iri_ds.rio.write_crs(config.IRI_CRS, inplace=True)
    iri_ds.to_netcdf(output_path)


def get_iri_data(
    config: Config(),
    download: bool = False,
):
    """
    Load IRI's NetCDF as a xarray dataset.
    The data is separated by tercile, i.e. a probability
    per tercile is included
    :param config: config for the drought indicator
    :param download: if True, download data
    :return: dataset containing the information in the netcdf file
    transform (numpy array): affine
        transformation of the dataset based on its CRS
    """
    iri_dir = Path(config.GLOBAL_DIR) / config.IRI_DIR
    Path(iri_dir).mkdir(parents=True, exist_ok=True)
    iri_filepath_raw_tercile = iri_dir / config.IRI_NC_FILENAME_RAW
    iri_filepath_clean_tercile = (
        Path(config.GLOBAL_DIR) / config.IRI_DIR / config.IRI_NC_FILENAME_CLEAN
    )
    if download:
        # need a key, assumed to be saved as an env variable with name IRI_AUTH
        iri_auth = os.getenv("IRI_AUTH")
        if not iri_auth:
            logger.error(
                "No authentication file found. Needs the environment variable"
                " 'IRI_AUTH'"
            )
        download_iri(iri_auth, iri_filepath_raw_tercile, url=config.IRI_URL)
        clean_iri_data(
            iri_filepath_raw_tercile, iri_filepath_clean_tercile, config
        )

    iri_ds = xr.load_dataset(iri_filepath_clean_tercile)

    return iri_ds


def get_iri_data_dominant(
    config: Config(),
    download: bool = False,
):
    """
    Load IRI's NetCDF where the dominant tercile is indicated
    Negative values indicate below average as dominant
    Positive values above average
    :param config: config for the drought indicator
    :param download: if True, download data
    :return: dataset containing the information in the netcdf file
    transform (numpy array): affine
        transformation of the dataset based on its CRS
    """
    iri_dir = Path(config.GLOBAL_DIR) / config.IRI_DIR
    Path(iri_dir).mkdir(parents=True, exist_ok=True)
    today_year = datetime.now().year
    iri_filepath_raw_dominant = iri_dir / f"iri_2017{today_year}_dominant.nc"
    iri_filepath_clean_dominant = (
        iri_dir / f"iri_2017{today_year}_dominant_clean.nc"
    )

    url_dominant = (
        "https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/"
        ".NMME_Seasonal_Forecast/.Precipitation_ELR/.dominant/data.nc"
    )
    if download:
        # need a key, assumed to be saved as an env variable with name IRI_AUTH
        iri_auth = os.getenv("IRI_AUTH")
        if not iri_auth:
            logger.error(
                "No authentication file found. Needs the environment variable"
                " 'IRI_AUTH'"
            )
        download_iri(iri_auth, iri_filepath_raw_dominant, url=url_dominant)
        clean_iri_data(
            iri_filepath_raw_dominant, iri_filepath_clean_dominant, config
        )

    iri_ds = xr.load_dataset(iri_filepath_clean_dominant)

    return iri_ds
