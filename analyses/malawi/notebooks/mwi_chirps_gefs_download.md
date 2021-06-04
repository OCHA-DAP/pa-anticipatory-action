---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: antact_orig
    language: python
    name: antact_orig
---

<!-- #region -->
### Analysis of CHIRPS-GEFS forecasts and observed dry spells

This notebook downloads the CHIRPS-GEFS 15 day forecasts and computes statistics per ADMIN2 for Malawi. The notebook `mwi_chirpsgefs_corr_dryspells.ipynb` uses this output to understand the correlation between CHIRPS-GEFS and historically observed dry spells. 

#### CHIRPS-GEFS
[CHIRPS-GEFS](https://chc.ucsb.edu/data/chirps-gefs) is the bias-corrected version of [GEFS](https://www.noaa.gov/media-release/noaa-upgrades-global-ensemble-forecast-system). GEFS is the Global Ensemble Forecast System from NOAA. CHIRPS observational data is used for the bias-correction. The forecast is published each day for the whole world with a 0.05 resolution. It is relatively new, started in 2018, and while it seems a respected source, little research articles exist around it. However, GEFS is a well-established source. Forecasts are available starting from 2000

Future: 
The size of the CHIRPS-GEFS data is now 40GB... Could we do something to make that smaller/not have to save it locally?      
    - ClimateServ includes CHIRPS-GEFS data and has an [API](https://github.com/Servir-Mekong/ClimateSERV_CHIRPS-GEFS/blob/master/Get_CHIRPS_GEFSv1/bin/ClimateServ_CHIPS-GEFS.py). Documentatio is limited but seems it is not the 15 day forecast though.. 

Data limitations:
- No CHIRPS-GEFS data is available from 01-01-2020 till 05-10-2020. This data is available from the [older version of the model](https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip/15day/Africa/precip_mean/), but our contact at CHC recommended to not use this


Assumptions
- The grid cell size is small enough to only look at cells with their centre within the region, not those touching
<!-- #endregion -->

### set general variables and functions

```python
%load_ext autoreload
%autoreload 2
```

```python
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
```

```python
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
```

#### Set config values

```python
#adm level to aggregate the raster cells to 
adm_level="adm1"#"adm2" #adm1

if adm_level=="adm1":
    adm_col="ADM1_EN"
if adm_level=="adm2":
    adm_col="ADM2_EN"
```

```python
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
```

```python
chirpsgefs_dir = os.path.join(config.DROUGHTDATA_DIR,"chirps_gefs")
```

```python
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])

#define which to use based on adm_level
if adm_level=="adm1":
    adm_bound_path=adm1_bound_path
elif adm_level=="adm2":
    adm_bound_path=adm2_bound_path
```

#### Rainy season
Compute a datetimeindex with all dates across all rainy seasons. Only data for these dates will be downloaded, to prevent the data from becoming even more massive

```python
#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean_back.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])
```

```python
df_rain.head()
```

```python
# #surprised 2020 earlies start is 01-12..
# df_rain[df_rain.onset_date.dt.year==2020]
```

```python
#set the onset and cessation date for the seasons with those dates missing (meaning there was no dry spell data from start/till end of the season)
df_rain_filled=df_rain.copy()
#remove entries where there is no onset and no cessation date. this happens for some adm2's in 2020
df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
#if onset date or cessation date is missing, set it to Nov 1/Jul1 to make sure all data of that year is downloaded
df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))
```

```python
#get min onset and max cessation for each season across all admin2's
df_rain_seas=df_rain_filled.groupby("season_approx",as_index=False).agg({'onset_date': np.min,"cessation_date":np.max})
```

```python
df_rain_seas.head()
```

```python
#create a daterange index including all dates within rainy seasons
all_dates=pd.Index([])
for i in df_rain_seas.season_approx.unique():
    seas_range=pd.date_range(df_rain_seas[df_rain_seas.season_approx==i].onset_date.values[0],df_rain_seas[df_rain_seas.season_approx==i].cessation_date.values[0])
    all_dates=all_dates.union(seas_range)
```

```python
all_dates
```

### Download CHIRPS-GEFS Africa data
We focus on the 15 day forecast, which is released every day.

We are focussing on the Africa data, since global data gets massive. Nevertheless, even for Africa it gets massive. 

```python
#ftp url, where year and the start_date are variable
#start_date is the day of the year for which the forecast starts
chirpsgefs_ftp_url_africa_15day="https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/15day/Africa/precip_mean/data.{year}.{start_day}.tif"
```

```python
#part of 2020 data is missing. Might be available with this URL, but uncertain what the difference is. Mailed Pete Peterson on 02-03
#https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip/15day/Africa/precip_mean/
```

```python
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
```

```python
# # only needed if not downloaded yet
# #download all the data
# for d in all_dates:
#     download_chirpsgefs(d,chirpsgefs_dir,chirpsgefs_ftp_url_africa_15day)
```

```python
def ds_stats_adm(ds, raster_transform, date, adm_path,ds_thresh_list=[2,4,5,10,15,20,25,30,35,40,45,50]):
    # compute statistics on level in adm_path for all dates in ds
    df = gpd.read_file(adm_path)
    # df["max_cell_touched"] = pd.DataFrame(
    #     zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, all_touched=True, nodata=np.nan))["max"]
    # df["min_cell_touched"] = pd.DataFrame(
    #     zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, all_touched=True, nodata=np.nan))["min"]
    df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["max"]
    df["min_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["min"]
    df["mean_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["mean"]

    for thres in ds_thresh_list:
        # compute the percentage of the admin area that has cells below the threshold
        # set all values with below average rainfall to 1 and others to 0
        forecast_binary = np.where(ds.values <= thres, 1, 0)
        # compute number of cells in admin region (sum) and number of cells in admin region with below average rainfall (count)
        bin_zonal = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],
                        nodata=np.nan))
        df[f'perc_se{thres}'] = bin_zonal['sum'] / bin_zonal['count'] * 100

    df["date"] = pd.to_datetime(date)

    df["date_forec_end"] = df["date"] + timedelta(days=14)

    return df
```

```python jupyter={"outputs_hidden": true} tags=[]
#this takes some time to compute, couple of hours at max
df_list=[]
#load the tif file for each date and compute the statistics
for d in all_dates: 
    d_str=pd.to_datetime(d).strftime("%Y%m%d")
    filename=f"chirpsgefs_africa_{d_str}.tif"
    try:
        rds=rioxarray.open_rasterio(os.path.join(chirpsgefs_dir,filename))
        df_date = ds_stats_adm(rds.sel(band=1), rds.rio.transform(), d_str, adm_bound_path)
        df_list.append(df_date)
    except Exception as e:
            print(e)
            print(filename)
            print(d_str)
df_hist_all=pd.concat(df_list)
```

```python
#remove the adm-date entries outside the rainy season for that specific adm
#before we included all forecasts within the min start of the rainy season and max end across the whole country
list_hist_rain_adm=[]
if adm_level=="adm1":
    gdf_adm2=gpd.read_file(adm2_bound_path)
    df_rain_filled_merged=df_rain_filled.merge(gdf_adm2[["ADM2_EN","ADM1_EN"]],on="ADM2_EN")
    df_rain_filled_adm=df_rain_filled_merged.groupby(["ADM1_EN","season_approx"],as_index=False).agg({'onset_date': np.min,"cessation_date":np.max})
elif adm_level=="adm2":
    df_rain_filled_adm=df_rain_filled.copy()
    
for a in df_hist_all[adm_col].unique():
    dates_adm=pd.Index([])
    for i in df_rain_filled_adm[df_rain_filled_adm[adm_col]==a].season_approx.unique():
        seas_range=pd.date_range(df_rain_filled_adm[(df_rain_filled_adm[adm_col]==a)&(df_rain_filled_adm.season_approx==i)].onset_date.values[0],df_rain_filled_adm[(df_rain_filled_adm[adm_col]==a)&(df_rain_filled_adm.season_approx==i)].cessation_date.values[0])
        dates_adm=dates_adm.union(seas_range)
    list_hist_rain_adm.append(df_hist_all[(df_hist_all[adm_col]==a)&(df_hist_all.date.isin(dates_adm))])
df_hist_rain_adm=pd.concat(list_hist_rain_adm)
```

```python
# #remove the adm2-date entries outside the rainy season for that specific adm2
# #before we included all forecasts within the min start of the rainy season and max end across the whole country
# list_hist_rain_adm2=[]
# for a in df_hist_all.ADM2_EN.unique():
#     dates_adm2=pd.Index([])
#     for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
#         seas_range=pd.date_range(df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].onset_date.values[0],df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].cessation_date.values[0])
#         dates_adm2=dates_adm2.union(seas_range)
#     list_hist_rain_adm2.append(df_hist_all[(df_hist_all.ADM2_EN==a)&(df_hist_all.date.isin(dates_adm2))])
# df_hist_rain_adm2=pd.concat(list_hist_rain_adm2)
```

```python
# #save file
# hist_path=os.path.join(country_data_exploration_dir,"chirpsgefs",f"mwi_chirpsgefs_rainyseas_stats_mean_back_{adm_level}.csv")
# df_hist_rain_adm.drop("geometry",axis=1).to_csv(hist_path,index=False)
```
