#!/usr/bin/env python
# coding: utf-8

# ### The correlation of observed precipitation data with dry spells
# This notebook explores the correlation between observed precipitation and dry spells. 
# The goal of the analysis is to see if, given perfect forecasting skill, there is information in the forecasted quantities for forecasting dryspells.
# This notebook focuses on the observation of below average seasonal and monthly precipitation
# For the seasonal and monthly precipitation as well as for the dry spells, CHIRPS is used as data source. The occurence of seasonal and monthly below average precipitation is computed in `mwi_obsprecip.ipynb`, and the dry spells are computed in `mwi_chirps_dry_spell_detection.R`
# 
# As first analysis we are focussing on the sole occurence of a dry spell per admin2. This can be extended to e.g. duration, number of dry spells, and geographical spread

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
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PROCESSED_DIR,country)
country_data_exploration_dir = os.path.join(config.DATA_DIR,"exploration",country)
drought_data_exploration_dir= os.path.join(config.DATA_DIR, "exploration",  'drought')
cams_data_dir=os.path.join(drought_data_exploration_dir,"CAMS_OPI")
cams_tercile_path=os.path.join(cams_data_dir,"CAMS_tercile.nc")
chirps_monthly_dir=os.path.join(drought_data_exploration_dir,"CHIRPS")
chirps_monthly_path=os.path.join(chirps_monthly_dir,"chirps_global_monthly.nc")


adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


# #### Load dry spell data

df_ds=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","dry_spells_during_rainy_season_list_2000_2020.csv")) 


df_ds


df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
df_ds["ds_fd_m"]=df_ds.dry_spell_first_date.dt.to_period("M")


#for now only want to know if a dry spell occured in a given month, so drop those that have several dry spells confirmed within a month
df_ds_drymonth=df_ds.drop_duplicates(["ADM2_EN","ds_fd_m"]).groupby(["ds_fd_m","ADM2_EN"],as_index=False).agg("count")[["ds_fd_m","ADM2_EN","dry_spell_first_date"]] #["ADM2_EN"]


# #### Load historical seasonal below average rainfall
# And remove seasons outside the rainy season
# For now, only including the seasons that are completely within the rainy season

df_belowavg_seas=pd.read_csv(os.path.join(country_data_processed_dir,"observed_belowavg_precip","chirps_seasonal_below_average_precipitation.csv"))
df_belowavg_seas.date_month=pd.to_datetime(df_belowavg_seas.date_month).dt.to_period("M")


#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])


df_rain[df_rain.ADM2_EN=="Balaka"]


#set the onset and cessation date for the seasons with them missing (meaning there was no dry spell data from start/till end of the season)
df_rain_filled=df_rain.copy()
df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))


df_rain_filled["onset_month"]=df_rain_filled["onset_date"].dt.to_period("M")
df_rain_filled["cessation_month"]=df_rain_filled["cessation_date"].dt.to_period("M")


df_rain_filled[df_rain_filled.ADM2_EN=="Balaka"]


#df_belowavg_seas only includes data from 2000, so the 1999 entries are not included
#remove the adm2-date entries outside the rainy season for that specific adm2
#before we included all forecasts within the min start of the rainy season and max end across the whole country
# total_days=0
list_hist_rain_adm2=[]
for a in df_rain_filled.ADM2_EN.unique():
    dates_adm2=pd.Index([])
    for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
#         onset_month_adm2=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].onset_date.values[0].to_period("M")
#         cessation_month_adm2=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].cessation_date.values[0].to_period("M")
        df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
#         df_rain_adm2_seas.onset_month.values[0]
#         print(df_rain_adm2_seas.onset_date.values[0])
#         print(df_rain_adm2_seas.cessation_date.values[0])
        seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
#         print(seas_range)
        dates_adm2=dates_adm2.union(seas_range) #[2:]
#     dates_adm2
#         total_days+=len(dates_adm2)
    list_hist_rain_adm2.append(df_belowavg_seas[(df_belowavg_seas.ADM2_EN==a)&(df_belowavg_seas.date_month.isin(dates_adm2))])
df_belowavg_seas_rain=pd.concat(list_hist_rain_adm2)


# ### Correlation of dry spells with Seasonal below average rainfall

# TODO:
# - check if no dates go missing from dry spell/seas data when merging
# - TODO exclude non-rainy season dates


#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed
# df_ds_drymonth_alldates=df_ds_drymonth.merge(df_belowavg_seas_rain[["ADM2_EN","date_month"]],how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])
df_ds_drymonth_rain=df_ds_drymonth.merge(df_belowavg_seas_rain[["ADM2_EN","date_month"]],how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_ds_drymonth_rain.dry_spell_first_date=df_ds_drymonth_rain.dry_spell_first_date.replace(np.nan,0)


#fill the data frame to include all months, also outside the rainy season --> this enables us to take the rolling sum 
#(else e.g. the rolling sum for Nov might include May-June-Nov)
df_ds_drymonth_alldates=df_ds_drymonth_rain.sort_values("date_month").set_index("date_month").groupby('ADM2_EN').resample('M').sum().drop("ADM2_EN",axis=1).reset_index()#['dry_spell_first_date'].rolling(3).sum()


df_ds_drymonth_alldates[df_ds_drymonth_alldates.ADM2_EN=="Balaka"].date_month.unique()


# (df_ds_drymonth_rain.reset_index(level=0)
#         .groupby('ADM2_EN')['dry_spell_first_date']
#         .apply(lambda x: x.asfreq('M'))
#         .reset_index())


# #compute the number of months of a season during which a dry spell occured
# #equals to nan if not all three months are within the season (based on the df_belowavg_seas_rain)
# #never occured that all 3 months have a dry spell..
# df_ds_drymonth_alldates["dry_spell_seas"]=df_ds_drymonth_alldates['dry_spell_first_date'].rolling(3).sum()


#compute the rolling sum of months having a dry spell per admin2
s_ds_dryseas=df_ds_drymonth_alldates.sort_values("date_month").set_index("date_month").groupby('ADM2_EN')['dry_spell_first_date'].rolling(3).sum()
#convert series to dataframe
df_ds_dryseas=pd.DataFrame(s_ds_dryseas).reset_index().sort_values(["ADM2_EN","date_month"])


df_ds_dryseas


len(df_ds_drymonth_alldates[df_ds_drymonth_alldates.dry_spell_first_date!=0])


#TODO!!! Something going wrong when merging with outer, get nans in below average. Understand why and fix!!!
#merge the dry spells with the info if a month had below average rainfall
#merge on right such that only the dates within the rainy season are included, df_ds_drymonth_alldates also includes all other months
df_comb_seas=df_ds_dryseas.merge(df_belowavg_seas_rain,how="right",on=["date_month","ADM2_EN"])


df_comb_seas


#remove dates where dry_spell_confirmation is nan, i.e. where rolling sum could not be computed for (first dates)
df_comb_seas=df_comb_seas[df_comb_seas.dry_spell_first_date.notna()]


#set the occurence of a dry spell to true if in at least one of the months of the season (=3 months) a dry spell occured
df_comb_seas["dry_spell"]=np.where(df_comb_seas.dry_spell_first_date>=1,1,0)
#never occurs in 3 consecutive months there is a dry spell in the same adm2
# df_comb_seas["dry_spell"]=np.where(df_comb_seas.dry_spell_first_date==2,1,0)


len(df_comb_seas[df_comb_seas.dry_spell==1])


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


from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
for m in df_comb_seas.ADM1_EN.unique():
    y_target =    df_comb_seas.loc[df_comb_seas.ADM1_EN==m,"dry_spell"]
    y_predicted = df_comb_seas.loc[df_comb_seas.ADM1_EN==m,"below_average"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
#     print(cm)

    fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
    ax.set_ylabel("Dry spell in ADMIN2 during season")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
    ax.set_title(m)
    plt.show()


#mapping of month to season. Computed by rolling sum, i.e. month indicates last month of season
seasons_rolling={3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND",1:"NDJ",2:"DJF"}


from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
for m in df_comb_seas.sort_values(by="date_month").date_month.dt.month.unique():
    y_target =    df_comb_seas.loc[df_comb_seas.date_month.dt.month==m,"dry_spell"]
    y_predicted = df_comb_seas.loc[df_comb_seas.date_month.dt.month==m,"below_average"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
#     print(cm)

    fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
    ax.set_ylabel("Dry spell in ADMIN2 during season")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
    ax.set_title(seasons_rolling[m])
    plt.show()


# #### Seasonal dry spells correlation number adm2's

df_comb_seas.groupby("date_month")["dry_spell","below_average"].sum().sort_values(by="dry_spell")


df_comb_seas.groupby("date_month")["dry_spell","below_average"].sum().sort_values(by="below_average")


df_numadm=df_comb_seas.groupby("date_month")["dry_spell","below_average"].sum()


df_numadm


sns.regplot(data = df_numadm, x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2)


sns.regplot(data = df_numadm, x = 'dry_spell', y = 'below_average', fit_reg = True,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2,)


df_nummonthadm=df_comb_seas.groupby(["date_month","ADM1_EN"],as_index=False)[["dry_spell","below_average"]].sum()


for a in df_nummonthadm.ADM1_EN.unique():
    g=sns.regplot(data = df_nummonthadm[df_nummonthadm.ADM1_EN==a], x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2)
    g.axes.set_title(a)
    plt.show()


# ### Observed dryspells and correlation with below average monthly rainfall
# **note: the list of dry spells used here is preliminary, thus the correlations will likely change but the processing should be the same**
# Process the observed dryspell list as outputed by `malawi/scripts/mwi_chirps_dry_spell_detection.R` and correlate the occurence of a dry spell with below-average monthly and seasonal rainfall
# 
# As first analysis we are focussing on the sole occurence of a dry spell per admin2. This can be extended to e.g. duration, number of dry spells, and geographical spread
# 
# Questions
# - Does it make sense to use the datae of dry spell confirmation, or more logical to use the start date?

df_ds_drymonth


df_belowavg_month=pd.read_csv(os.path.join(country_data_processed_dir,"observed_belowavg_precip","chirps_monthly_below_average_precipitation.csv"))
df_belowavg_month.date_month=pd.to_datetime(df_belowavg_month.date_month).dt.to_period("M")


#df_belowavg_seas only includes data from 2000, so the 1999 entries are not included
#remove the adm2-date entries outside the rainy season for that specific adm2
#before we included all forecasts within the min start of the rainy season and max end across the whole country
# total_days=0
list_hist_rain_adm2=[]
for a in df_rain_filled.ADM2_EN.unique():
    dates_adm2=pd.Index([])
    for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
        df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
        seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
        dates_adm2=dates_adm2.union(seas_range)
    list_hist_rain_adm2.append(df_belowavg_month[(df_belowavg_month.ADM2_EN==a)&(df_belowavg_seas.date_month.isin(dates_adm2))])
df_belowavg_month_rain=pd.concat(list_hist_rain_adm2)


#merge the dry spells with the info if a month had below average rainfall
#merge on outer such that all dates present in one of the two are included
df_comb=df_ds_drymonth.merge(df_belowavg_month_rain,how="outer",left_on=["ds_fd_m","ADM2_EN"],right_on=["date_month","ADM2_EN"])


df_comb.head()


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_comb.dry_spell_first_date=df_comb.dry_spell_first_date.replace(np.nan,0)


#contigency matrix rainfall and dry spells for all months
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

y_target =    df_comb["dry_spell_first_date"]
y_predicted = df_comb["below_average"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during month")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
plt.show()

