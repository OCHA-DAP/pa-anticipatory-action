#!/usr/bin/env python
# coding: utf-8

# ### The correlation of observed precipitation data with dry spells
# This notebook explores the correlation between observed precipitation and dry spells. 
# We transform the observed precipitation to correspond with the format that precipitation **forecasts** are published in.   
# We do this since the goal of the analysis is to see if, given perfect forecasting skill, there is information in the forecasted quantities for forecasting dryspells. 
# 
# We explore the following precipitation-related variables:
# - Lower tercile (i.e. below-average) monthly precipitation
# - Lower tercile 3-monthly (i.e. seasonal) precipitation
# - [TODO: ENSO state]    
# 
# 
# We prefer to use as many readily available products as we can. However, we could only find historical monthly terciles (using CAMS-OPI data) and not historical seasonal terciles. We have therefore computed these seasonal terciles ourselves. As source we use CHIRPS data as this is also the source used to compute the dry spells, we have experience working with this source, has a high resolugion, and generally speaking good performance. 
# 
# Since the code to compute monthly terciles is almost the same as seasonal, we also propose to use the CHIRPS methodology to compute the monthly terciles instead of using the readily available CAMS-OPI terciles. 

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
import matplotlib.colors as mcolors
import xarray as xr
import cftime
import math
import rioxarray
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib as mpl
import seaborn as sns


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
print(path_mod)
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
country_data_exploration_dir = os.path.join(config.DATA_DIR,"exploration",country)
drought_data_exploration_dir= os.path.join(config.DATA_DIR, "exploration",  'drought')
cams_data_dir=os.path.join(drought_data_exploration_dir,"CAMS_OPI")
cams_tercile_path=os.path.join(cams_data_dir,"CAMS_tercile.nc")
chirps_monthly_dir=os.path.join(drought_data_exploration_dir,"CHIRPS")
chirps_monthly_path=os.path.join(chirps_monthly_dir,"chirps_global_monthly.nc")


adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


# ### CHIRPS seasonal
# Optimally we would use data that is already processed to indicate the percentile per season and year. However, we couldn't find this data so we processed it ourselves using CHIRPS data.     
# The CHIRPS data is provided as monthly sum per 5x5 degree raster cell. This is global data, and we clip it to Malawi.     
# From this data we first compute the total precipitation per 3 month period. We then compute the climatological lower tercile where we use the period 1981-2010 to define the climatology. Thereafter, we combine the two to get a yes/no if whether any cell within an admin2 region had below average rainfall. 
# 
# We use below-average and lower tercile interchangily in the comments but they refer to the same
# 
# Questions:
# - Is there a source I am missing that has readily processed seasonal percentile data?
# - Does this methodology make sense?
# - Is it valid to say there was below average rainfall in an admin2 if any cell touching that region had below average rainfall?
# - Is it essential to include a dry mask?

df_bound = gpd.read_file(adm1_bound_path)


chirps_monthly_mwi_path=os.path.join(chirps_monthly_dir,"chirps_mwi_monthly.nc")


#date to make plots for to test values. To be sure this is consistent across the different plots
test_date=cftime.DatetimeGregorian(2020, 1, 1, 0, 0, 0, 0)
test_date_dtime="2020-1-1"


# # Only need to run if you want to update the data
# # The file is around 7GB so takes a while
# url_chirpsmonthly="https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
# download_ftp(url_chirpsmonthly,chirps_monthly_path)
# # clipping to MWI can take some time, so clip and save to a new file
# ds=xr.open_dataset(chirps_monthly_path)
# ds_clip = ds.rio.set_spatial_dims(x_dim="longitude",y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
# ds_clip.to_netcdf(chirps_monthly_mwi_path)


#don't fully understand, but need masked=True to read nan values as nan correctly
ds_clip=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)


ds_clip=ds_clip.rename({"x":"lon","y":"lat"})


ds_clip.rio.transform()


#this loads the dataset completely
#default is to do so called "lazy loading", which causes aggregations on a larger dataset to be very slow. See http://xarray.pydata.org/en/stable/io.html#netcdf
#however, this .load() takes a bit of time, about half an hour
ds_clip.load()


#show the data for each month of 2020, clipped to MWI
g=ds_clip.sel(time=ds_clip.time.dt.year.isin([2020])).precip.plot(
    col="time",
    col_wrap=6,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation (mm)"
    },
    cmap="YlOrRd",
)

df_bound = gpd.read_file(adm2_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")


#should already sorted by date, but just to be sure before applying rolling
ds_clip=ds_clip.sortby("time")


#compute the rolling sum over three month period. Rolling sum works backwards, i.e. value for month 3 is sum of month 1 till 3
seas_len=3
ds_season=ds_clip.rolling(time=seas_len,min_periods=seas_len).sum().dropna(dim="time",how="all")


ds_season


#check dense spot for other dates


#plot the three month sum for NDJ 2019/2020
fig=plot_raster_boundaries_clip([ds_season.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))


#adjust climatological period - experiment


#define the years that are used to define the climatology. We use 1981-2010 since this is also the period used by IRI's seasonal forecasts
ds_season_climate=ds_season.sel(time=ds_season.time.dt.year.isin(range(1981,2011)))


ds_season_climate


#compute the thresholds for the lower tercile, i.e. below average, per season
#since we computed a rolling sum, each month represents a season
ds_season_climate_quantile=ds_season_climate.groupby(ds_season_climate.time.dt.month).quantile(0.33)


ds_season_climate_quantile


#hotspot check


#plot the below average boundaries for the NDJ season
fig=plot_raster_boundaries_clip([ds_season_climate_quantile.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(month=1)],adm1_bound_path,colp_num=1,forec_val="precip",predef_bins=np.linspace(0,1000,10),cmap="YlOrRd")


sns.distplot(ds_season_climate.sel(time=ds_season_climate.time.dt.month==1).precip.values.flatten())


sns.distplot(ds_season_climate_quantile.sel(month=1).precip.values.flatten())


# possibly change -999


#determine the raster cells that have below-average precipitation, other cells are set to -999
list_ds_seass=[]
for s in np.unique(ds_season.time.dt.month):
    ds_seas_sel=ds_season.sel(time=ds_season.time.dt.month==s)
    #keep original values of cells that are either nan or have below average precipitation, all others are set to -999
    ds_seas_below=ds_seas_sel.where((ds_seas_sel.precip.isnull())|(ds_seas_sel.precip<=ds_season_climate_quantile.sel(month=s).precip),-999)#,drop=True)
    list_ds_seass.append(ds_seas_below)
ds_season_below=xr.concat(list_ds_seass,dim="time")        


ds_season_below


# #can be used to inspect the values that are below average
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_season_below.sel(time=test_date)["precip"].values.flatten()[~np.isnan(ds_season_below.sel(time=test_date)["precip"].values.flatten())]))


#fraction of values with below average value. Should be around 0.33
ds_season_below_notnan=ds_season_below.precip.values[~np.isnan(ds_season_below.precip.values)]
np.count_nonzero(ds_season_below_notnan!=-999)/np.count_nonzero(ds_season_below_notnan)


#check occurence below average per raster cell
#compute #times each category
#geographical autocorrelation


#plot the cells that had below-average rainfall in NDJ of 2019/2020
fig=plot_raster_boundaries_clip([ds_season_below.where(ds_season_below.precip!=-999).rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))


#usual center point


#resolution forecasts--> interpolate (or vectorize)
#Do lots of CRS checks! 
#test different thresholds percentage area
#forecast: weighted average or >%area above threshold


def alldates_statistics_seasonal(ds,raster_transform,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    for date in ds.time.values:
        df=gpd.read_file(adm_path)
        ds_date=ds.sel(time=date)
        
        # compute the percentage of the admin area that has below average rainfall
        #set all values with below average rainfall to 1 and others to 0
        forecast_binary = np.where(ds_date.precip.values!=-999, 1, 0)
        #compute number of cells in admin region (sum) and number of cells in admin region with below average rainfall (count)
        bin_zonal = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],nodata=np.nan))
        df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
        
        #same but then also including cells that only touch the admin region, i.e. don't have their cell within that region
        bin_zonal_touched = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, all_touched=True, stats=['count', 'sum'],nodata=np.nan))
        df['perc_threshold_touched'] = bin_zonal_touched['sum'] / bin_zonal_touched['count'] * 100
        
        #return the value of the cell with the highest value within the admin region. 
        #In our case if this isn't -999 it indicates that at least one cell has below average rainfall
        df["max_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, all_touched=True,nodata=np.nan))["max"]
        df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan))["max"]
        
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
  
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")
    #all admins that are not -999, have at least one cell with below average rainfall
    df_hist[f"below_average_touched"]=np.where(df_hist["max_cell_touched"]!=-999,1,0)
    df_hist[f"below_average_max"]=np.where(df_hist["max_cell"]!=-999,1,0)
        
    df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")
    df_hist['date_month']=df_hist.date.dt.to_period("M")
        
    return df_hist


ds_season_below.rio.crs





ds_season_below.rio.transform()


ds_season_below.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.transform()


#compute whether the maximum cell that touches an admin2 region has below average rainfall for each month since 2010
#have to write the crs for the transform to be correct!
df_belowavg_seas=alldates_statistics_seasonal(ds_season_below.sel(time=ds_season_below.time.dt.year.isin(range(2010,2021))),ds_season_below.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.transform(),adm2_bound_path)


# #quality check
# #check that all max touched values in df_belowavg_seas are also present in ds_season_below
# #this is to make sure all goes correctly with zonal_stats regarding crs's since it has happened that things go wrong there without insight why
# display(df_belowavg_seas[df_belowavg_seas.date==test_date_dtime])
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_season_below.sel(time=test_date)["precip"].values.flatten()[~np.isnan(ds_season_below.sel(time=test_date)["precip"].values.flatten())]))


#percentage of area with below average rain
#TODO: decide which threshold to use
#could argue for 50%, i.e. majority
#could also argue for 30% as this results in around 1/3 of the occurences the area being below average, which is the definition of terciles. 
#Don't know if large difference between adm regions
#Note: only including cells that have their centre within the region, do think this makes sense especially with the high resolution we got
for i in np.linspace(0,100,11):
    print(f"Fraction of adm2-date combinations where >={i}% of area received bel. avg. rain:","{:.2f}".format(len(df_belowavg_seas[df_belowavg_seas["perc_threshold"]>=i])/len(df_belowavg_seas)))


#This is when assigning the whole adm2 region as below average when any cell touching has below average rainfall 
#--> logical that it is a larger fraction than the climatological 0.33
print("fraction of adm2s with at least one cell with below average rain:",len(df_belowavg_seas[df_belowavg_seas["below_average_max"]==1])/len(df_belowavg_seas))


#for now using 0.30 as threshold
df_belowavg_seas["below_average"]=np.where(df_belowavg_seas["perc_threshold"]>=30,1,0)


#Hmm surprised that there is such a large difference across months.. Would want them to all be around 0.33?
#compute fraction of values having below average per month
df_bavg_month=df_belowavg_seas.groupby(df_belowavg_seas.date.dt.month).sum()
df_bavg_month['fract_below']=df_bavg_month["below_average"]/df_belowavg_seas.groupby(df_belowavg_seas.date.dt.month).count()["below_average"]
df_bavg_month[["fract_below"]]


#compute fraction of values having below average per year
df_bavg_year=df_belowavg_seas.groupby(df_belowavg_seas.date.dt.year).sum()
df_bavg_year['fract_below']=df_bavg_year["below_average"]/df_belowavg_seas.groupby(df_belowavg_seas.date.dt.year).count()["below_average"]
df_bavg_year[["fract_below"]]


#seems the deviation from 0.33 is not crazy large for any adm2 and not clear dependency on the size of the adm2
#compute fraction of values having below average per ADMIN2
df_bavg_adm2=df_belowavg_seas.groupby("ADM2_EN").sum()
df_bavg_adm2["fract_below"]=df_bavg_adm2["below_average"]/df_belowavg_seas.groupby("ADM2_EN").count()["below_average"]
df_bavg_adm2[["fract_below","Shape_Area"]]


#Currently not being used
# #mapping of month to season. Computed by rolling sum, i.e. month indicates last month of season
# seasons_rolling={3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND",1:"NDJ",2:"DJF"}


# ### Observed dryspells and correlation with seasonal below-average rainfall
# Process the observed dryspell list as outputed by `malawi/scripts/mwi_chirps_dry_spell_detection.R` and correlate the occurence of a dry spell with below-average seasonal rainfall
# 
# As first analysis we are focussing on the sole occurence of a dry spell per admin2. This can be extended to e.g. duration, number of dry spells, and geographical spread

country_data_processed_dir = os.path.join(config.DATA_DIR,config.PROCESSED_DIR,country)


# df_ds=pd.read_csv(os.path.join(country_data_exploration_dir,"dryspells","mwi_dry_spells_list.csv")) #"../Data/transformed/mwi_dry_spells_list.csv")
df_ds=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","dry_spells_during_rainy_season_list.csv")) 


df_ds


df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
df_ds["ds_fd_m"]=df_ds.dry_spell_first_date.dt.to_period("M")


#for now only want to know if a dry spell occured in a given month, so drop those that have several dry spells confirmed within a month
df_ds_drymonth=df_ds.drop_duplicates(["ADM2_EN","ds_fd_m"]).groupby(["ds_fd_m","ADM2_EN"],as_index=False).agg("count")[["ds_fd_m","ADM2_EN","dry_spell_first_date"]] #["ADM2_EN"]


#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed
df_ds_drymonth_alldates=df_ds_drymonth.merge(df_belowavg_seas[["ADM2_EN","date_month"]],how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_ds_drymonth_alldates.dry_spell_first_date=df_ds_drymonth_alldates.dry_spell_first_date.replace(np.nan,0)


#compute the rolling sum of months having a dry spell per admin2
s_ds_dryseas=df_ds_drymonth_alldates.sort_values("date_month").set_index("date_month").groupby('ADM2_EN')['dry_spell_first_date'].rolling(3).sum()
#convert series to dataframe
df_ds_dryseas=pd.DataFrame(s_ds_dryseas).reset_index()


df_ds_dryseas


#supposed length of df_ds_dryseas
#TODO: check discrepancy
11*12*len(df_ds.ADM2_EN.unique())


#merge the dry spells with the info if a month had below average rainfall
#merge on outer such that all dates present in one of the two are included
df_comb_seas=df_ds_dryseas.merge(df_belowavg_seas,how="outer",on=["date_month","ADM2_EN"])


#remove dates where dry_spell_confirmation is nan, i.e. where rolling sum could not be computed for (first dates)
df_comb_seas=df_comb_seas[df_comb_seas.dry_spell_first_date.notna()]


#set the occurence of a dry spell to true if in at least one of the months of the season (=3 months) a dry spell occured
df_comb_seas["dry_spell"]=np.where(df_comb_seas.dry_spell_first_date>=1,1,0)


df_comb_seas.head()


from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

y_target =    df_comb_seas["dry_spell"]
y_predicted = df_comb_seas["below_average"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during season")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
plt.show()


#TODO: define this based on rainy_seasons output R script
df_comb_seas_rainyseas=df_comb_seas[df_comb_seas.date_month.dt.month.isin([11,12,1,2,3,4])]


from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

y_target =    df_comb_seas_rainyseas["dry_spell"]
y_predicted = df_comb_seas_rainyseas["below_average"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during season")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
plt.show()


# ### CHIRPS monthly

#TODO: make sure is aligned with newest seasonal version


#date to make plots for to test values. To be sure this is consistent across the different plots
test_month_date=cftime.DatetimeGregorian(2020, 1, 1, 0, 0, 0, 0)
test_month_date_dtime="2020-1-1"


#plot the three month sum for NDJ 2019/2020
fig=plot_raster_boundaries_clip([ds_clip.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_month_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))


#define the years that are used to define the climatology. We use 1981-2010 since this is also the period used by IRI's seasonal forecasts
ds_month_climate=ds_clip.sel(time=ds_clip.time.dt.year.isin(range(1981,2011)))


#compute the thresholds for the lower tercile, i.e. below average, per season
#since we computed a rolling sum, each month represents a season
ds_month_climate_quantile=ds_month_climate.groupby(ds_month_climate.time.dt.month).quantile(0.33,skipna=True)


sns.distplot(ds_month_climate.sel(time=ds_month_climate.time.dt.month==1).precip.values.flatten())


sns.distplot(ds_month_climate_quantile.sel(month=1).precip.values.flatten())


ds_month_climate_quantile


#plot the below average boundaries for the NDJ season
fig=plot_raster_boundaries_clip([ds_month_climate_quantile.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(month=1)],adm1_bound_path,colp_num=1,forec_val="precip",predef_bins=np.linspace(0,1000,10),cmap="YlOrRd")


#determine the raster cells that have below-average precipitation, other cells are set to -999
list_ds_months=[]
for s in np.unique(ds_clip.time.dt.month):
    ds_month_sel=ds_clip.sel(time=ds_clip.time.dt.month==s)
    #drop removes the dates with all nan values, i.e. no below average, but we dont want that
    ds_month_below=ds_month_sel.where((ds_month_sel.precip.isnull()) | (ds_month_sel.precip<=ds_month_climate_quantile.sel(month=s).precip) ,-999)
    list_ds_months.append(ds_month_below)
ds_month_below=xr.concat(list_ds_months,dim="time")        


sns.distplot(ds_month_below.precip.values.flatten())


with np.printoptions(threshold=np.inf):
    print(np.unique(ds_month_below.sel(time=test_month_date)["precip"].values.flatten()[~np.isnan(ds_month_below.sel(time=test_month_date)["precip"].values.flatten())]))


ds_month_below_notnan=ds_month_below.precip.values[~np.isnan(ds_month_below.precip.values)]
np.count_nonzero(ds_month_below_notnan!=-999)/np.count_nonzero(ds_month_below_notnan)


#not correct anymore cause using -999
#plot the cells that had below-average rainfall in DJF of 2019/2020
fig=plot_raster_boundaries_clip([ds_month_below.where(ds_month_below.precip!=-999).rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_month_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))


def alldates_statistics_monthly(ds,raster_transform,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    for date in ds.time.values:
        df=gpd.read_file(adm_path)
        ds_date=ds.sel(time=date)
        df["max_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, all_touched=True,nodata=np.nan))["max"]
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
        
         # calculate the percentage of the area within an geographical area that has a value larger than threshold
        forecast_binary = np.where(ds_date.precip.values!=-999, 1, 0)
        bin_zonal = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],nodata=np.nan))
        df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
        bin_zonal_touched = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, all_touched=True, stats=['count', 'sum'], nodata=np.nan))
        df['perc_threshold_touched'] = bin_zonal_touched['sum'] / bin_zonal_touched['count'] * 100
  
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")
    #all cells that are not nan, have below average rainfall
    for m in ["max_cell_touched"]:
        df_hist[f"below_average"]=np.where(df_hist[m]!=-999,1,0)

        
   
        
    df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")
    df_hist['date_month']=df_hist.date.dt.to_period("M")
        
    return df_hist


ds_month_below.rio.transform()


ds_month_below


#compute whether the maximum cell that touches an admin2 region has below average rainfall for each month since 2010
df_belowavg_month=alldates_statistics_monthly(ds_month_below.sel(time=ds_month_below.time.dt.year.isin(range(2010,2021))),ds_month_below.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.transform(),adm2_bound_path)


# #quality check
# #check that all max touched values in df_belowavg_month are also present in ds_month_below
# #this is to make sure all goes correctly with zonal_stats regarding crs's since it has happened that things go wrong there without insight why
# display(df_belowavg_month[df_belowavg_month.date==test_date_dtime])
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_month_below.sel(time=test_month_date)["precip"].values.flatten()[~np.isnan(ds_month_below.sel(time=test_month_date)["precip"].values.flatten())]))


#AAARRRHHH this should be around 0.33... Nope it shouldnt since we are taking the max cell touched --> logical higher than 0.33.. However earlier on it should
#Plus problematic for correlations if this is an overestimation. Or well this depends again how we classify an adm2 with ofrecasts, and would argue for doing that according to largest area
print("fraction of observations with below average value:",len(df_belowavg_month[df_belowavg_month["below_average"]==1])/len(df_belowavg_month))


#Hmm strange that there is such a large difference across months.. Should in theory all be 0.33?
#compute fraction of values having below average per month
df_bavg_month=df_belowavg_month.groupby(df_belowavg_month.date.dt.month).sum()
df_bavg_month['fract_below']=df_bavg_month["below_average"]/df_belowavg_month.groupby(df_belowavg_month.date.dt.month).count()["below_average"]
df_bavg_month[["fract_below"]]


#Hmm would have expected more deviation per year, e.g. I thought 2011 was very little rainfall year?
#compute fraction of values having below average per month
df_bavg_year=df_belowavg_month.groupby(df_belowavg_month.date.dt.year).sum()
df_bavg_year['fract_below']=df_bavg_year["below_average"]/df_belowavg_month.groupby(df_belowavg_month.date.dt.year).count()["below_average"]
df_bavg_year[["fract_below"]]


#compute fraction of values having below average per ADMIN2
df_bavg_adm2=df_belowavg_month.groupby("ADM2_EN").sum()
df_bavg_adm2["fract_below"]=df_bavg_adm2["below_average"]/df_belowavg_month.groupby("ADM2_EN").count()["below_average"]
df_bavg_adm2[["fract_below"]]


# ### Quality checks

# #### Compare computed 3 month sum to that reported directly by CHIRPS
# AAAAHHH There are large differences and I don't understand why... 
# - The 3monthly data for Global and Africa don't correspond at all, also not when inspecting it in QGIS
# - The 3monthly Global data clipped to MWI is not in the same range as the 3 month rolling sum
# - The 3monthly Africa data clipped to MWI is in the same range, but doesn't equal the 3 month rolling sum
# 
# 
# Leaving it as it is for now and summing the monthly data to 3 months ourselves since this is an easier to use format. However should at some point understand the differences.. 




ds_chirps_3month=rioxarray.open_rasterio("../Data/chirps.3monthly.africa2019.111201.tiff",masked=True)
# ds_chirps_3month=rioxarray.open_rasterio("../Data/chirps.3monthly.africa2019.091011.tiff",masked=True)
# ds_chirps_3month=rioxarray.open_rasterio("../Data/chirps.3monthly.global2019.111201.tiff",masked=True)
# ds_chirps_3month=rioxarray.open_rasterio("../Data/chirps.3monthly.global2019.091011.tiff",masked=True)


ds_chirps_3month=ds_chirps_3month.where(ds_chirps_3month.values!=-9999)


ds_chirps_3month_mwi = ds_chirps_3month.rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)


#3 monthly mean from chirps ftp
ds_chirps_3month_mwi.mean()


#computed mean from our framework
ds_season.sel(time=cftime.DatetimeGregorian(2020, 1, 1, 0, 0, 0, 0)).precip.mean()
# ds_season.sel(time=cftime.DatetimeGregorian(2019, 11, 1, 0, 0, 0, 0)).precip.mean()


# ### IRI CAMS-OPI observed monthly terciles
# **First wanted to use this source for computation of historical below avearge, but now we also got CHIRPS which has a higher resolution and is the ssame source as we used for computing the dry spells, I think it makes more sense to use that data**
# 
# The percentile data is downloaded from [IRI's Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Precipitation/Percentiles.html?bbox=bb%3A-20%3A-40%3A55%3A40%3Abb&T=Dec%202020)   
# The percentiles are computed from the CAMS-OPI data. The period of 1981-2010 is used to define the climatology for each calendar month of the year. A dry mask is included for regions that receive less than 10 mm/month of precipitation on average in the 1981-2010 climatology for the month of the year.    
# 
# Somehow the resolution is very low, while the original CAMS-OPI dataset has a higher resolution. I don't know what the reason of this is though for the exploration it should be good enough
# 

# #would be great to do with opendap but couldn't find the right url.. 
# obs_tercile_url="http://iridl.ldeo.columbia.edu/ds:/SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.mean/.prcp/prcp_percentiles/dods"
# obs_tercile_url="http://iridl.ldeo.columbia.edu/ds:/SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.mean/.prcp/dup/T/%28Jan%201981%29%28Dec%202010%29RANGE/T/12/splitstreamgrid%5BT2%5D0.0/0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/1.0/1.0/replacebypercentile%5Bpercentile%5Dboundedtablefunction/startcolormap/DATA/0/1/RANGE/white/sienna/sienna/0/VALUE/sienna/0.1/bandmax/tan/RGBdup/0.2/bandmax/white/white/0.8/bandmax/PaleGreen/RGBdup/0.9/bandmax/DarkGreen/RGBdup/1.0/bandmax/RGBdup/endcolormap//name//prcp_percentiles/def//long_name/%28Precipitation%20Percentiles%20%28brown%20below%2020th%20and%20green%20above%2080th%29%29def/SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.mean/.prcp/T/%28Jan%201981%29%28Dec%202010%29RANGE/T/%28days%20since%201960-01-01%29streamgridunitconvert/T/differential_mul/T/%28months%20since%201960-01-01%29streamgridunitconvert//units/%28mm/month%29def/yearly-climatology/10/flaglt%5BT%5DregridAverage//name//dry_mask/def/:ds/prcp_percentiles/dods"
# xr.open_dataset(obs_tercile_url)


# #only need to run if you want to update the data
# obs_tercile_netcdf_url="http://iridl.ldeo.columbia.edu/ds:/SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.mean/.prcp/dup/T/%28Jan%201981%29%28Dec%202010%29RANGE/T/12/splitstreamgrid%5BT2%5D0.0/0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/1.0/1.0/replacebypercentile%5Bpercentile%5Dboundedtablefunction/startcolormap/DATA/0/1/RANGE/white/sienna/sienna/0/VALUE/sienna/0.1/bandmax/tan/RGBdup/0.2/bandmax/white/white/0.8/bandmax/PaleGreen/RGBdup/0.9/bandmax/DarkGreen/RGBdup/1.0/bandmax/RGBdup/endcolormap//name//prcp_percentiles/def//long_name/%28Precipitation%20Percentiles%20%28brown%20below%2020th%20and%20green%20above%2080th%29%29def/SOURCES/.NOAA/.NCEP/.CPC/.CAMS_OPI/.v0208/.mean/.prcp/T/%28Jan%201981%29%28Dec%202010%29RANGE/T/%28days%20since%201960-01-01%29streamgridunitconvert/T/differential_mul/T/%28months%20since%201960-01-01%29streamgridunitconvert//units/%28mm/month%29def/yearly-climatology/10/flaglt%5BT%5DregridAverage//name//dry_mask/def/:ds/prcp_percentiles/data.nc"
# download_url(obs_tercile_netcdf_url,cams_tercile_path)


terc_ds=xr.open_dataset(cams_tercile_path,decode_times=False)
terc_ds=terc_ds.rename({"X":"lon","Y":"lat"})
terc_ds=fix_calendar(terc_ds,timevar="T")
terc_ds = xr.decode_cf(terc_ds)
terc_ds = invert_latlon(terc_ds)

# AAARGHH apparently for this data I shouldn't adjust the longitude range, while for the rainfall forecasts it seemed essential
# and I have no clue what might cause this difference. Possibly also connected to the shapefile??
#This is essential for the rasterstats.zonal_stats, and I still don't fully understand what the format is that it needs
#But if it is not correct, it overlays the shapefile with another part of the world and thus the results can be completely off..
# terc_ds = change_longitude_range(terc_ds)


#check again with this new methodology of computing the transform of the ds


transform=terc_ds.rio.transform()


transform


terc_ds


#check that all percentile values are in the expected range, i.e. 0-1
np.unique(terc_ds.prcp_percentiles.values.flatten()[~np.isnan(terc_ds.prcp_percentiles.values.flatten())])


#plot the globe for one of the dates
plt.imshow(terc_ds.sel(T=cftime.Datetime360Day(2020, 12, 16, 0, 0, 0, 0))["prcp_percentiles"].values)


#clip to country
df_bound = gpd.read_file(adm1_bound_path)
ds_clip = terc_ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)


ds_clip


#fraction of values with percentile <= 0.33, i.e. below average value. Should be around 0.33
ds_clip_notnan=ds_clip.prcp_percentiles.values[~np.isnan(ds_clip.prcp_percentiles.values)]
np.count_nonzero(ds_clip_notnan<=0.33)/np.count_nonzero(ds_clip_notnan)


#show the data for each month of 2020, clipped to MWI
g=ds_clip.sel(T=ds_clip.T.dt.year.isin([2020])).prcp_percentiles.plot(
    col="T",
    col_wrap=3,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation percentile (mm)"
    },
)

df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")


def alldates_statistics(ds,raster_transform,prob_threshold,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    print(f"Number of forecasts: {len(ds.T)}")
    for date in ds.T.values:
        df=gpd.read_file(adm_path)
        ds_date=ds.sel(T=date)
        df["min_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["prcp_percentiles"].values, affine=raster_transform, all_touched=True,nodata=-999))["min"]
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")

    for m in ["min_cell_touched"]:
        df_hist[f"{m}_se{prob_threshold}"]=np.where(df_hist[m]<=prob_threshold,1,0)
  
    df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")

    return df_hist


#compute whether the minimum cell that touches an admin1 region has a percentile of 0.33 or below for each month since 2010
df_terc=alldates_statistics(terc_ds.sel(T=terc_ds.T.dt.year.isin(range(2010,2021))),terc_ds.rio.transform(),0.33,adm2_bound_path)
#remove the day from the date as it is monthly data and easier for further processing
df_terc["date_month"]=df_terc.date.dt.to_period("M")


# `rasterstats.zonal_stats` overlays the raster data with the shapefile that I still don't fully understand. If e.g. the range of longitude coordinates is not as `zonal_stats` it expects, it overlays the shapefile with another part of the world.   
# Thus, a double check that the `min_cell_touched` values in `df_terc` are one of the actual raw values in `ds_clip` on the same date

ds_clip.sel(T=cftime.Datetime360Day(2020, 12, 16, 0, 0, 0, 0))["prcp_percentiles"].values


df_terc[df_terc.date=="2020-12-16"]


#due to the methodology, this is larger than the fraction of raster cells with below average value
print(f"fraction of adm2s with below average value:",len(df_terc[df_terc["min_cell_touched_se0.33"]==1])/len(df_terc))


#Hmm strange that there is such a large difference across months.. Should in theory all be 0.33?
#compute fraction of values having below average per month
df_terc_month=df_terc.groupby(df_terc.date.dt.month).sum()
df_terc_month['fract_below']=df_terc_month["min_cell_touched_se0.33"]/df_terc.groupby(df_terc.date.dt.month).count()["min_cell_touched_se0.33"]#/(len(df_terc.ADM2_EN.unique())*len(df_terc.date.dt.year.unique()))
df_terc_month[["fract_below"]]


#Hmm would have expected more deviation per year, e.g. I thought 2011 was very little rainfall year?
#compute fraction of values having below average per month
df_terc_year=df_terc.groupby(df_terc.date.dt.year).sum()
df_terc_year['fract_below']=df_terc_year["min_cell_touched_se0.33"]/df_terc.groupby(df_terc.date.dt.year).count()["min_cell_touched_se0.33"]#/(len(df_terc.ADM2_EN.unique())*len(df_terc.date.dt.year.unique()))
df_terc_year[["fract_below"]]


#compute fraction of values having below average per ADMIN2
df_terc_adm2=df_terc.groupby("ADM2_EN").sum()
df_terc_adm2["fract_below"]=df_terc_adm2["min_cell_touched_se0.33"]/df_terc.groupby("ADM2_EN").count()["min_cell_touched_se0.33"]
df_terc_adm2[["fract_below"]]


# ### Observed dryspells and correlation with below average monthly rainfall
# **note: the list of dry spells used here is preliminary, thus the correlations will likely change but the processing should be the same**
# Process the observed dryspell list as outputed by `malawi/scripts/mwi_chirps_dry_spell_detection.R` and correlate the occurence of a dry spell with below-average monthly and seasonal rainfall
# 
# As first analysis we are focussing on the sole occurence of a dry spell per admin2. This can be extended to e.g. duration, number of dry spells, and geographical spread
# 
# Questions
# - Does it make sense to use the datae of dry spell confirmation, or more logical to use the start date?

df_ds=pd.read_csv(os.path.join(country_data_exploration_dir,"dryspells","mwi_dry_spells_list.csv")) #"../Data/transformed/mwi_dry_spells_list.csv")


df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_confirmation"]=pd.to_datetime(df_ds["dry_spell_confirmation"])
df_ds["ds_conf_m"]=df_ds.dry_spell_confirmation.dt.to_period("M")


#for now only want to know if a dry spell occured in a given month, so drop those that have several dry spells confirmed within a month
df_ds_drymonth=df_ds.drop_duplicates(["ADM2_EN","ds_conf_m"]).groupby(["ds_conf_m","ADM2_EN"],as_index=False).agg("count")[["ds_conf_m","ADM2_EN","dry_spell_confirmation"]] #["ADM2_EN"]


df_ds_drymonth


#merge the dry spells with the info if a month had below average rainfall
#merge on outer such that all dates present in one of the two are included
df_comb=df_ds_drymonth.merge(df_terc,how="outer",left_on=["ds_conf_m","ADM2_EN"],right_on=["date_month","ADM2_EN"])


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_comb.dry_spell_confirmation=df_comb.dry_spell_confirmation.replace(np.nan,0)


#contigency matrix rainfall and dry spells for all months
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

y_target =    df_comb["dry_spell_confirmation"]
y_predicted = df_comb["min_cell_touched_se0.33"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during month")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
plt.show()


#This will eventually be defined by output from R script but for now making rough selection of months generally considered rainy season
df_comb_rainyseas=df_comb[df_comb.date_month.dt.month.isin([11,12,1,2,3,4])]


#contigency matrix rainfall and dry spells for rainy season months
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

y_target =    df_comb_rainyseas["dry_spell_confirmation"]
y_predicted = df_comb_rainyseas["min_cell_touched_se0.33"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during month")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
plt.show()

