import os
import logging
import sys
from pathlib import Path
import rasterio
import rioxarray
import shutil

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.drought.config import Config
from utils_general.utils import auth_googleapi, download_gdrive, unzip

logger = logging.getLogger(__name__)

def download_icpac(config):
    """
    Download the ICPAC data from the centre for humdata's drive.
    A secret key is needed for this which should be saved as environmental variable with the name GAPI_AUTH
    Args:
        config (Config): config for the drought indicator
    """
    #TODO: would be nicer to directly download from their ftp but couldn't figure out yet how (something with certificates)
    gclient = auth_googleapi()
    gzip_output_file=os.path.join(config.DROUGHTDATA_DIR, f'{config.ICPAC_DIR}.zip')
    if os.path.exists(os.path.join(config.DROUGHTDATA_DIR, config.ICPAC_DIR)):
        shutil.rmtree(os.path.join(config.DROUGHTDATA_DIR, config.ICPAC_DIR))
    download_gdrive(gclient, config.ICPAC_GDRIVE_ZIPID, gzip_output_file)
    unzip(gzip_output_file,config.DROUGHTDATA_DIR)
    os.remove(gzip_output_file)
    for path in Path(os.path.join(config.DROUGHTDATA_DIR,config.ICPAC_DIR)).rglob(config.ICPAC_PROBFORECAST_REGEX_RAW):
        #opening with rioxarray better than xarray, with xarray gets some lat lon inversion
        icpac_ds = rioxarray.open_rasterio(path)
        #TODO: selection of below is needed to save crs correctly but don't understand why
        icpac_sel = icpac_ds[config.ICPAC_LOWERTERCILE]
        path_crs = f"{str(path)[:-3]}_crs.nc"
        if os.path.exists(path_crs):
            os.remove(path_crs)
        icpac_sel.rio.write_crs("EPSG:4326").to_netcdf(path_crs)


def get_icpac_data(config,pubyear,pubmonth,download=False):
    """
    Load ICPAC's NetCDF file as xarray dataset
    Args:
        config (Config): config for the drought indicator
        pubyear (str): year forecast is published in YYYY format
        pubmonth (str): month forecast is published as abbreviation, e.g. Nov
        download (bool): if True, download data

    Returns:
        icpac_ds (xarray dataset): dataset continaing the information in the netcdf file
        transform (numpy array): affine transformation of the dataset based on its CRS
    """
    if download:
        download_icpac(config)
    #TODO: check if better way to find the file
    try:
        for path in Path(os.path.join(config.DROUGHTDATA_DIR,config.ICPAC_DIR)).rglob(config.ICPAC_PROBFORECAST_REGEX_CRS.format(month=pubmonth,year=pubyear)):

            #rioxarray reads the icpac data correctly while xarray somehow messes up stuff but still not sure what exactly goes wrong there
            icpac_ds = rioxarray.open_rasterio(path,masked=True).squeeze()
            icpac_ds = icpac_ds.rename({config.ICPAC_LON: config.LONGITUDE, config.ICPAC_LAT: config.LATITUDE, config.ICPAC_LOWERTERCILE:config.LOWERTERCILE})

            #assume all transforms of the different files are the same, so just select the one that is read the latest
            with rasterio.open(path) as src:
                transform = src.transform
        return icpac_ds, transform
    except UnboundLocalError:
        logger.error(f"ICPAC forecast with regex {config.ICPAC_PROBFORECAST_REGEX_CRS.format(month=pubmonth,year=pubyear)} not found")
