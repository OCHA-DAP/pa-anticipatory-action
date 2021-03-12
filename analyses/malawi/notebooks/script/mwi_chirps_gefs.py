#!/usr/bin/env python
# coding: utf-8

# ### Analysis of CHIRPS-GEFS forecasts and observed dry spells
# **Note: this notebook needs some cleaning up**
# 
# Goal is to see if CHIRPS-GEFS provides a signal for the occurence of a dry spell in the coming 15 days   
# We do this by looking at historical CHIRPS-GEFS data, and classifying the occurence of a dry spell according to this forecast per admin 2   
# This information is thereafter coupled to historically observed dry spells   
# 
# From the first analysis we find that there is not much correlation between forecasted and observed dry spells. However, we need to further understand if the analysis was computed correclty, and if so, what the cause of these differences is. 
# 
# 
# Questions
# - Defined a dry spell being forecasted if cell with the maximum value that touches the admin2 is forecasted to receive maximum 2mm of rain. Is this a good definition?
# - Is it logical to look at the correlation of a dry spell starting at the first date of the forecast, or should we look more broadly at the occurence of a dry spell during the forecast period?
# - Any other methodologies to combine binary geospatial timeseries?
# - [Given deadline, lets wait but important for future] The size of the CHIRPS-GEFS data is now 23.4GB... Could we do something to make that smaller/not have to save it locally?      
#     Climateserv might be an option, includes chirps-gefs data and can select per country and has an API. However, not clear what exactly the data is that you can download (how many days forecast) and API is hard to understand, and not sure if includes latest data. Someone using the API [here](https://github.com/Servir-Mekong/ClimateSERV_CHIRPS-GEFS/blob/master/Get_CHIRPS_GEFSv1/bin/ClimateServ_CHIPS-GEFS.py)
# 
# To do:
# - Understand why observed and forecasted dry spells don't match
# - Get a bit better understanding of the CHIRPS-GEFS methodology
# - Make sure the date of the filename is the first date of the forecast --> yes it is
# - Adjust rainy season definnitions based on definition of dry spell (max or mean)
# 
# 
# Limitations:
# - Part of 2020 data missing
# - Not all data 2000-2010 has been downloaded
# - 

# ### set general variables and functions

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.enums import Resampling
import matplotlib
import matplotlib.colors as mcolors
import xarray as xr
import cftime
import math
import rioxarray
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib as mpl
import datetime
from datetime import timedelta
import re
import seaborn as sns


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
# print(path_mod)
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp,download_url
from src.utils_general.raster_manipulation import fix_calendar, invert_latlon, change_longitude_range
from src.utils_general.plotting import plot_raster_boundaries_clip


# #### Set config values

country="malawi"
config=Config()
parameters = config.parameters(country)
country_dir = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
country_data_raw_dir = os.path.join(config.DATA_DIR,config.RAW_DIR,country)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PROCESSED_DIR,country)
country_data_exploration_dir = os.path.join(config.DATA_DIR,"exploration",country)
drought_data_exploration_dir= os.path.join(config.DATA_DIR, "exploration",  'drought')
cams_data_dir=os.path.join(drought_data_exploration_dir,"CAMS_OPI")
cams_tercile_path=os.path.join(cams_data_dir,"CAMS_tercile.nc")
chirps_monthly_dir=os.path.join(drought_data_exploration_dir,"CHIRPS")
chirps_monthly_path=os.path.join(chirps_monthly_dir,"chirps_global_monthly.nc")


chirpsgefs_dir = os.path.join(config.DROUGHTDATA_DIR,"chirps_gefs")


adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


# #### Rainy season
# Compute a datetimeindex with all dates across all rainy seasons

#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])


df_rain


#set the onset and cessation date for the seasons with them missing (meaning there was no dry spell data from start/till end of the season)
df_rain_filled=df_rain.copy()
df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))


#get min onset and max cessation for each season across all admin2's
df_rain_seas=df_rain_filled.groupby("season_approx",as_index=False).agg({'onset_date': np.min,"cessation_date":np.max})


df_rain_seas


all_dates=pd.Index([])
for i in df_rain_seas.season_approx.unique():
    seas_range=pd.date_range(df_rain_seas[df_rain_seas.season_approx==i].onset_date.values[0],df_rain_seas[df_rain_seas.season_approx==i].cessation_date.values[0])
    all_dates=all_dates.union(seas_range)


all_dates


# ### Download CHIRPS-GEFS Africa data
# We focus on the 15 day forecast, which is released every day. We define a dry spell occuring during that period if the cell with the maximum value within an admin2 is forecasted to receive less than 2mm of rainfall
# 
# We are focussing on the Africa data, since global data gets massive. Nevertheless, even for Africa it gets massive. 

#ftp url, where year and the start_date are variable
#start_date is the day of the year for which the forecast starts
chirpsgefs_ftp_url_africa_15day="https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/15day/Africa/precip_mean/data.{year}.{start_day}.tif"


#part of 2020 data is missing. Might be available with this URL, but uncertain what the difference is. Mailed Pete Peterson on 02-03
#https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip/15day/Africa/precip_mean/


def download_chirpsgefs(date,output_dir,chirpsgefs_ftp_url,days=""):
    """
    Download the chirps-gefs africa 15 day forecast for the given year and day of the year
    Currently in expiremntation style code
    days: number of days forecast predicts ahead. When using 15 day, set to empty string
    """
    
    year=date.year
    day_year=str(date.timetuple().tm_yday).zfill(3)
    date_str=date.strftime("%Y%m%d")
    chirpsgefs_filepath = os.path.join(chirpsgefs_dir, f"chirpsgefs{days}_africa_{date_str}.tif")
    if not os.path.exists(chirpsgefs_filepath):
        print(date_str)
        print(chirpsgefs_ftp_url.format(year=year,start_day=day_year))
        try:
            download_ftp(chirpsgefs_ftp_url.format(year=year,start_day=day_year), chirpsgefs_filepath)
        except Exception as e: 
            print(f'CHIRPS-GEFS data not available for {date}')
            print(e)


all_dates_2010=pd.to_datetime(all_dates)[pd.to_datetime(all_dates).year>=2010]


# only needed if not downloaded yet
#download all the data
for d in all_dates_2010:
    download_chirpsgefs(d,chirpsgefs_dir,chirpsgefs_ftp_url_africa_15day)


#redownload broken files
for d in ["2010-05-07","2011-05-17","2012-02-29","2012-05-17","2013-05-17","2015-05-17","2016-02-29","2016-05-17","2017-05-17","2018-05-17","2019-05-17","2000-10-23","2002-09-09","2007-04-30"]:
    date=pd.to_datetime(d)
    download_chirpsgefs(date,chirpsgefs_dir)


def ds_maxcell(ds,raster_transform,date,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df=gpd.read_file(adm_path)
    df["max_cell_touched"] = pd.DataFrame(
    zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, all_touched=True,nodata=np.nan))["max"]
    df["min_cell_touched"] = pd.DataFrame(
    zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, all_touched=True,nodata=np.nan))["min"]
    df["max_cell"] = pd.DataFrame(
    zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["max"]
    df["min_cell"] = pd.DataFrame(
    zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["min"]
    df["mean_cell"] = pd.DataFrame(
    zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["mean"]
    df["date"]=pd.to_datetime(date)
    #define dryspell occuring if the cell touching the region with the maximum values is expected to receive less than 2mm of rain
#     df["dryspell"]=np.where(df["max_cell_touched"]<=2,1,0)
    df["date_forec_end"]=df["date"]+timedelta(days=14)
       
    return df


#only needed if new data added, else can load df_hist from file
#for each forecast date, compute the occurence of a dry spell and convert to a dataframe
df_list=[]
for filename in os.listdir(chirpsgefs_dir):
    if filename.endswith(".tif"):
        date=pd.to_datetime(re.split("[.\_]+", filename)[-2],format="%Y%m%d")
        if date in all_dates_2010:
            try:
                rds = rioxarray.open_rasterio(os.path.join(chirpsgefs_dir,filename))
                df_date=ds_maxcell(rds.sel(band=1),rds.rio.transform(),date,adm2_bound_path)
                df_list.append(df_date)
            except Exception as e:
                print(e)
                print(filename)
                print(date)
df_hist_all=pd.concat(df_list)


#TODO: decide if wanna use same start and end of rainy season for all adm2's or do according to original data
#probs wanna do according to the adm2 data since there might be large differences
#which means we have to remove the dates currently in df_hist that are outside the rainy season for adm2s


len(df_hist_all)


len(df_hist_all)


# df_hist_all=df_hist.copy()


df_rain_filled


#remove the adm2-date entries outside the rainy season for that specific adm2
#before we included all forecasts within the min start of the rainy season and max end across the whole country
list_hist_rain_adm2=[]
for a in df_hist_all.ADM2_EN.unique():
    dates_adm2=pd.Index([])
    for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
        seas_range=pd.date_range(df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].onset_date.values[0],df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].cessation_date.values[0])
        dates_adm2=dates_adm2.union(seas_range)
    list_hist_rain_adm2.append(df_hist_all[(df_hist_all.ADM2_EN==a)&(df_hist_all.date.isin(dates_adm2))])
df_hist_rain_adm2=pd.concat(list_hist_rain_adm2)


df_hist_rain_adm2


df_hist_rain_adm2.date.dt.month.unique()


# #REMOVE: only for testing
# df_hist=df_hist_rain_adm2.copy()


#computation of df_hist can take rather long, so also save it and load it if no new info is added
hist_path=os.path.join(country_data_exploration_dir,"chirpsgefs","mwi_chirpsgefs_rainyseas_maxcell_test.csv")
# df_hist_sel=df_hist_rain_adm2[["date","date_forec_end","ADM1_EN","ADM2_EN","max_cell_touched","min_cell_touched","max_cell","min_cell"]]
# df_hist_sel.to_csv(hist_path,index=False)
df_hist_rain_adm2.drop("geometry",axis=1).to_csv(hist_path,index=False)
df_hist=pd.read_csv(hist_path)
df_hist["date"]=pd.to_datetime(df_hist["date"])
df_hist["date_forec_end"]=pd.to_datetime(df_hist["date_forec_end"])


df_hist


df_hist[df_hist.dryspell==1]


#that is a huge number of dry spells... Expecting they mainly occur in the last months of the rainy season (May/June) but should check thatd
sns.displot(df_hist,x="max_cell_touched")


#check how distribution changes when using min cell touched instead of max --> even more sensitive
sns.displot(df_hist,x="min_cell_touched")


print("number of adm2-date combination with min cell touched <=2mm",len(df_hist[df_hist.min_cell_touched<=2]))


print("number of adm2-date combination with max cell touched <=2mm",len(df_hist[df_hist.max_cell_touched<=2]))


print("number of dates with at least one adm2 with max cell touched <=2mm",len(df_hist[df_hist.max_cell_touched<=2].date.unique()))


# print("number of dates with at least one adm2 with max cell touched <=2mm",len(df_hist[df_hist.max_cell<=2].date.unique()))


# ### Load ds with dates within rainy seas

ds_list=[]
for d in [rainy_date for rainy_date in all_dates if rainy_date.year>=2010]:
    d_str=pd.to_datetime(d).strftime("%Y%m%d")
    filename=f"chirpsgefs_africa_{d_str}.tif"
    try:
        rds=rioxarray.open_rasterio(os.path.join(chirpsgefs_dir,filename))
        rds=rds.assign_coords({"time":pd.to_datetime(d)})
        rds=rds.sel(band=1)
        ds_list.append(rds)
    except Exception as e:
            print(e)
            print(filename)
            print(d_str)


ds_drys=xr.concat(ds_list,dim="time")


ds_drys=ds_drys.sortby("time")


ds_drys.to_netcdf(country_data_processed_dir,"chirps_gefs","mwi_chirpsgefs_15day.nc")


ds_drys


# ### CHIRPS GEFS 5 day




df_ds=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","dry_spells_during_rainy_season_list_2000_2020_mean.csv"))
df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])


#ftp url, where year and the start_date are variable
#start_date is the day of the year for which the forecast starts
chirpsgefs_ftp_url_africa_5day="https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/05day/Africa/precip_mean/data.{year}.{start_day}.tif"


# only needed if not downloaded yet
#download all the data
for d in df_ds.dry_spell_first_date.unique():
    download_chirpsgefs(pd.to_datetime(d),chirpsgefs_dir,chirpsgefs_ftp_url_africa_5day,days="_5day")


ds_list=[]
for d in df_ds.dry_spell_first_date.unique():
    d_str=pd.to_datetime(d).strftime("%Y%m%d")
    filename=f"chirpsgefs_5day_africa_{d_str}.tif"
    rds=rioxarray.open_rasterio(os.path.join(chirpsgefs_dir,filename))
    rds=rds.assign_coords({"time":pd.to_datetime(d)})
    rds=rds.sel(band=1)
    ds_list.append(rds)


ds_drys=xr.concat(ds_list,dim="time")


ds_drys=ds_drys.sortby("time")

