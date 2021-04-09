#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from pathlib import Path
import os
import sys
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import numpy as np
import rasterio


path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
print(path_mod)
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.plotting import plot_raster_boundaries, plot_timeseries
from src.indicators.drought.utils import parse_args
# from utils_general.utils import config_logger
from src.indicators.drought.chirps_rainfallobservations import get_chirps_data,chirps_plot_alldates


#rast vs antact
# xarray 0.16.1 vs 0.16.2
# pandas 1.1.3 vs 1.0.5
#rasterio 1.1.7 vs 1.1.8


config = Config()
country = "malawi"
parameters = config.parameters(country)
country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
country_data_raw_dir = os.path.join(config.DATA_DIR,config.RAW_DIR,country)
adm1_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])


#assuming data is already downloaded, else add download=True
ds,transform = get_chirps_data(config, 2020)


year=2020


ds_3011=ds.sel(time=f"{year}-11-30").squeeze()


#check that data overlaps correctly with country boundaries
fig_bound=plot_raster_boundaries(ds_3011, country, parameters, config,forec_val="precip")


#plot histogram and example of data for a month
ds_nov=ds.sel(time=slice(f"{year}-11-01",f"{year}-11-30"))
fig_histo,fig_dates=chirps_plot_alldates(ds_nov, adm1_path, config)


# ### CHIRPS rolling sum

#have to implement to concat different years
# years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
years=[2020]


chirps_filepath = os.path.join(config.DROUGHTDATA_DIR, config.CHIRPS_DIR,config.CHIRPS_NC_FILENAME_RAW.format(year=2020, resolution="25"))


ds=rioxarray.open_rasterio(chirps_filepath)


xr.show_versions()


ds=xr.open_dataset(os.path.join(config.DROUGHTDATA_DIR, config.CHIRPS_DIR,config.CHIRPS_NC_FILENAME_RAW.format(year=2020, resolution="25")))
ds2=xr.open_dataset(os.path.join(config.DROUGHTDATA_DIR, config.CHIRPS_DIR,config.CHIRPS_NC_FILENAME_RAW.format(year=2019, resolution="25")))


ds


ds2


for i in years:
    ds,transform = get_chirps_data(config, i,resolution="25")
    df_bound = gpd.read_file(adm1_path)
    #clip global to malawi to speed up calculating rolling sum
    ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    # rolling sum of 14 days. Uses right side of the window, i.e. sum of last 14 days
    ds_roll=ds_clip.rolling(time=14).sum()


ds_clip


ds_roll


#check for negative values. Somehow with a wrong combination of package versions this might occur
print(np.unique(ds_roll.precip.values.flatten()[~np.isnan(ds_roll.precip.values.flatten())]))


adm1_path=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR,parameters['path_admin1_shp'])
download=False
for i in years:
    ds,transform = get_chirps_data(config, i, download = download)
    df_bound = gpd.read_file(adm1_path)
    #clip global to malawi to speed up calculating rolling sum
    ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.clip(
        df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    # rolling sum of 14 days
    ds_roll=ds_clip.rolling(time=14).sum()
    print(ds_roll)


df_roll=ds_roll.to_dataframe().drop("spatial_ref",axis=1).reset_index()


df_roll.sort_values("precip")


df_roll.dropna().sort_values("precip")


#set to 1 if a cell received less than 2mm over 14 days
df_roll["dryspell"]=np.where(df_roll["precip"]<=2,1,0)


#NaN for dates where no rolling sum available, i.e. first 13 days of the year
df_roll.groupby("time").sum(min_count=1).sort_values("dryspell")


# ### Some random testing

df_roll[df_roll['time']=="2020-01-01"]


from datetime import timedelta


df=ds_clip.to_dataframe().reset_index()


end_date=pd.to_datetime("2020-03-10")
df[(df['time'] > end_date-timedelta(days=14)) & (df['time'] <= end_date) & (df["lat"]==-13.875) & (df["lon"]==34.875)]


df[(df['time'] > end_date-timedelta(days=14)) & (df['time'] <= end_date) & (df["lat"]==-13.875) & (df["lon"]==34.875)].rolling(14).sum()


end_date=pd.to_datetime("2020-03-14")
df[(df['time'] > end_date-timedelta(days=14)) & (df['time'] <= end_date) & (df["lat"]==-10.125) & (df["lon"]==34.125)]


#old negative values with wrong packages for reference
df_roll.sort_values("precip")




