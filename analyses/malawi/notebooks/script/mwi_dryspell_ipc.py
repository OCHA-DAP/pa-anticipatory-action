#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.enums import Resampling
import matplotlib.colors as mcolors
import xarray as xr
import cftime
import math
import rioxarray
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib as mpl


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
print(path_mod)
sys.path.append(path_mod)
from src.indicators.food_insecurity.config import Config
from src.utils_general.utils import download_ftp,download_url
from src.utils_general.raster_manipulation import fix_calendar, invert_latlon, change_longitude_range
from src.utils_general.plotting import plot_raster_boundaries_clip


# #### Set config values

country="malawi"
admin_level=2
suffix=""
config=Config()
parameters = config.parameters(country)
country_dir = os.path.join(config.DIR_PATH, "analyses", country)
country_data_raw_dir = os.path.join(config.DATA_DIR, 'raw', country)
country_data_processed_dir = os.path.join(config.DATA_DIR, 'processed', country)
country_data_exploration_dir = os.path.join(config.DATA_DIR,"exploration",country)
drought_data_exploration_dir= os.path.join(config.DATA_DIR, "exploration",  'drought')
cams_data_dir=os.path.join(drought_data_exploration_dir,"CAMS_OPI")
cams_tercile_path=os.path.join(cams_data_dir,"CAMS_tercile.nc")
chirps_monthly_dir=os.path.join(drought_data_exploration_dir,"CHIRPS")
chirps_monthly_path=os.path.join(chirps_monthly_dir,"chirps_global_monthly.nc")


adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


fewsnet_dir = os.path.join(country_data_processed_dir, config.FEWSWORLDPOP_PROCESSED_DIR)
fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix)


globalipc_dir=os.path.join(country_data_processed_dir, config.GLOBALIPC_PROCESSED_DIR)
globalipc_path=os.path.join(globalipc_dir,f"{country}_globalipc_admin{admin_level}{suffix}.csv")


df_fn=pd.read_csv(os.path.join(fewsnet_dir,fewsnet_filename),index_col=False)
df_fn["source"]="FewsNet"


df_fn.head()


df_gipc=pd.read_csv(globalipc_path)
df_gipc["source"]="GlobalIPC"


df_gipc.head()


df_ipc=pd.concat([df_fn,df_gipc])


df_ipc.head()




