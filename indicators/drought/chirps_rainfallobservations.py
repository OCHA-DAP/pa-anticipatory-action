from pathlib import Path
import os
import sys
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import urllib
import rasterio


path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from utils_general.utils import download_ftp
from indicators.drought.config import Config
from utils_general.plotting import plot_raster_boundaries_clip, plot_histogram


def chirps_plot_alldates(ds,adm1_path,config,predef_bins=None):
    #just testing function, has to be optimized
    df_bound=gpd.read_file(adm1_path)
    ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    # comprehend dataset to list of datasets to give as input to plotting function
    #compute histogram of all historical values
    #quite some nans, but this is due that ds_clip is a rectangle where any cells outside the country boundaries are set to nan
    ds_clip_array=ds_clip[config.CHIRPS_VARNAME].values.flatten()
    fig_histog=plot_histogram(ds_clip_array,xlabel="Precipitation (mm)")

    ds_list = [ds_clip.sel(time=p) for p in ds_clip["time"]]

    #create list of titles for subplots
    #TODO: find better way to format title list regardless of x being datetime object
    # ds_list_date_str=[pd.to_datetime(str(x)).strftime("%Y-%m-%d") for x in ds_clip["time"].values]
    ds_list_date_str=[x for x in ds_clip["time"].values]
    if predef_bins is None:
        predef_bins=np.linspace(ds_clip.precip.min(), ds_clip.precip.max(), 10)
    fig_clip = plot_raster_boundaries_clip(ds_list, adm1_path, title_list=ds_list_date_str, forec_val=config.CHIRPS_VARNAME, colp_num=3, figsize=(90,60),labelsize=40,predef_bins=predef_bins) #figszie=18*colp_num,6*rows

    return fig_histog,fig_clip

def download_chirps(config,year,resolution="25"):
    """
        Download the CHIRPS data for year from their ftp server
        Args:
            config (Config): config for the drought indicator
            year (str or int): year for which the data should be downloaded in YYYY format
            resolution (str): resolution of the data to be downloaded. Can be 25 or 05
        """
    chirps_dir = os.path.join(config.DROUGHTDATA_DIR, config.CHIRPS_DIR)
    Path(chirps_dir).mkdir(parents=True, exist_ok=True)
    chirps_filepath = os.path.join(chirps_dir, config.CHIRPS_NC_FILENAME_RAW.format(year=year,resolution=resolution))
    # TODO: decide if only download if file doesn't exist. Not sure if ever gets updated
    today=datetime.now()
    #often data is uploaded at a later date, so update those files that are in the current year and in the previous year if at the start of new year
    year_update=(today-timedelta(days=60)).year
    if not os.path.exists(chirps_filepath) or int(year)>=year_update:
        try:
            if os.path.exists(chirps_filepath):
                os.remove(chirps_filepath)
            download_ftp(config.CHIRPS_FTP_URL_GLOBAL_DAILY.format(year=year,resolution=resolution), chirps_filepath)
            # ds=rioxarray.open_rasterio(chirps_filepath)
            ds=xr.open_dataset(chirps_filepath)
            chirps_filepath_crs = os.path.join(chirps_dir, config.CHIRPS_NC_FILENAME_CRS.format(year=year,resolution=resolution))
            if os.path.exists(chirps_filepath_crs):
                os.remove(chirps_filepath_crs)
            ds.rio.write_crs("EPSG:4326").to_netcdf(chirps_filepath_crs)
        except urllib.error.HTTPError as e:
            logging.error(f"{e}. Date might be later than last reported datapoint. URL:{config.CHIRPS_FTP_URL_GLOBAL_DAILY.format(year=year,resolution=resolution)}")

def get_chirps_data(config, year, resolution="25", download=False):
    """
    Load CHIRP's NetCDF file as xarray dataset
    Args:
        config (Config): config for the drought indicator
        year (str or int): year for which the data should be loaded in YYYY format
        resolution (str): resolution of the data to be downloaded. Can be 25 or 05
        download (bool): if True, download data

    Returns:
        icpac_ds (xarray dataset): dataset continaing the information in the netcdf file
        transform (numpy array): affine transformation of the dataset based on its CRS
    """

    if download:
        download_chirps(config,year,resolution)

    chirps_filepath_crs = os.path.join(config.DROUGHTDATA_DIR, config.CHIRPS_DIR,config.CHIRPS_NC_FILENAME_CRS.format(year=year, resolution=resolution))
    #TODO: would prefer rioxarray but crashes when clipping
    ds = xr.open_dataset(chirps_filepath_crs)
    # ds = rioxarray.open_rasterio(chirps_filepath_crs)
    ds=ds.rename({config.CHIRPS_LON: config.LONGITUDE, config.CHIRPS_LAT: config.LATITUDE})

    with rasterio.open(chirps_filepath_crs) as src:
        transform = src.transform

    return ds, transform