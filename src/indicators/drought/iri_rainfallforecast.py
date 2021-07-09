import requests
import os
import logging
from pathlib import Path
import rasterio
import xarray as xr
import rioxarray  # noqa: F401

from src.utils_general.raster_manipulation import (
    invert_latlon,
    change_longitude_range,
    fix_calendar,
)

logger = logging.getLogger(__name__)


def download_iri(iri_auth, config, chunk_size=128):
    """
    Download the IRI seasonal tercile forecast as NetCDF file from the
    url as given in config.IRI_URL Saves the file to the path as defined
    in config Args: iri_auth: iri key for authentication. An account is
    needed to get this key config (Config): config for the drought
    indicator chunk_size (int): number of bytes to download at once
    """
    # TODO: it would be nicer to download with opendap instead of
    # requests, since then the file doesn't even have to be saved and is
    # hopefully faster. But haven't been able to figure out how to do
    # that with cookie authentication
    IRI_dir = os.path.join(config.GLOBAL_DIR, config.IRI_DIR)
    Path(IRI_dir).mkdir(parents=True, exist_ok=True)
    IRI_filepath = os.path.join(IRI_dir, config.IRI_NC_FILENAME_RAW)
    # strange things happen when just overwriting the file, so delete it
    # first if it already exists
    if os.path.exists(IRI_filepath):
        os.remove(IRI_filepath)

    # have to authenticate by using a cookie
    cookies = {
        "__dlauth_id": iri_auth,
    }
    # TODO fix/understand missing certificate verification warning For
    # now leaving it as it is since it is a trustable site and we
    # couldn't figure how to improve it
    logger.info("Downloading IRI NetCDF file. This might take some time")
    response = requests.get(config.IRI_URL, cookies=cookies, verify=False)
    with open(IRI_filepath, "wb") as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    # TODO: explore if can also open with rioxarray instead of xarray.
    # Getting an error with current settings
    iri_ds = xr.open_dataset(
        IRI_filepath, decode_times=False, drop_variables="C"
    )
    IRI_filepath_crs = os.path.join(IRI_dir, config.IRI_NC_FILENAME_CRS)

    if os.path.exists(IRI_filepath_crs):
        os.remove(IRI_filepath_crs)

    # invert_latlon assumes lon and lat to be the names of the coordinates
    iri_ds = iri_ds.rename(
        {config.IRI_LON: config.LONGITUDE, config.IRI_LAT: config.LATITUDE}
    )
    # often IRI latitude is flipped so check for that and invert if needed
    iri_ds = invert_latlon(iri_ds)
    iri_ds = change_longitude_range(iri_ds)
    # The iri data is in EPSG:4326 but this isn't included in the
    # filedata (at least from experience) This Coordinate Reference
    # System (CRS) information is later on needed, so add it to the file
    iri_ds.rio.set_spatial_dims(
        x_dim=config.LONGITUDE, y_dim=config.LATITUDE
    ).rio.write_crs("EPSG:4326").to_netcdf(IRI_filepath_crs)


def get_iri_data(config, download=False):
    """
    Load IRI's NetCDF as a xarray dataset Args: config (Config): config
    for the drought indicator download (bool): if True, download data

    Returns: iri_ds (xarray dataset): dataset continaing the information
        in the netcdf file transform (numpy array): affine
        transformation of the dataset based on its CRS
    """
    if download:
        # need a key, assumed to be saved as an env variable with name IRI_AUTH
        iri_auth = os.getenv("IRI_AUTH")
        if not iri_auth:
            logger.error(
                "No authentication file found. Needs the environment variable"
                " 'IRI_AUTH'"
            )
        download_iri(iri_auth, config)
    IRI_filepath = os.path.join(
        config.GLOBAL_DIR, config.IRI_DIR, config.IRI_NC_FILENAME_CRS
    )
    # the nc contains two bands, prob and C. Still not sure what C is
    # used for but couldn't discover useful information in it and will
    # give an error if trying to read both (cause C is also a variable
    # in prob) the date format is formatted as months since 1960. In
    # principle xarray can convert this type of data to datetime, but
    # due to a wrong naming of the calendar variable it cannot do this
    # automatically Thus first load with decode_times=False and then
    # change the calendar variable and decode the months
    iri_ds = xr.open_dataset(
        IRI_filepath, decode_times=False, drop_variables="C"
    )
    iri_ds = fix_calendar(iri_ds, timevar="F")
    iri_ds = xr.decode_cf(iri_ds)

    # TODO: understand rasterio warnings "CPLE_AppDefined in No UNIDATA
    # NC_GLOBAL:Conventions attribute" and "CPLE_AppDefined in No 1D
    # variable is indexed by dimension C" Conventions attribute is
    # caused by the fact that rasterio is expecting a conventions
    # instead of Conventions. Shouldn't impact performance no 1D
    # variable is indexed by C I am not entirely sure what is meant
    with rasterio.open(IRI_filepath) as src:
        transform = src.transform

    return iri_ds, transform
