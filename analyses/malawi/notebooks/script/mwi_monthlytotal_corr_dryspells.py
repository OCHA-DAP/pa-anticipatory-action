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
import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar


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

df_ds=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","dry_spells_during_rainy_season_list_2000_2020_mean_back.csv")) 
df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
df_ds["ds_fd_m"]=df_ds.dry_spell_first_date.dt.to_period("M")


df_ds


#compute if start of dryspell per month-ADM2
#for now only want to know if a dry spell occured in a given month, so drop those that have several dry spells confirmed within a month
df_ds_drymonth=df_ds.drop_duplicates(["ADM2_EN","ds_fd_m"]).groupby(["ds_fd_m","ADM2_EN"],as_index=False).agg("count")[["ds_fd_m","ADM2_EN","dry_spell_first_date"]]


# ### Total monthly vs dry spells

#dataframe that includes all month-adm2 combinations that experienced a dry spell 
#(if several dry spells started within a month in an adm2, these "duplicates" are dropped)
df_ds_drymonth.head()


df_total_month=pd.read_csv(os.path.join(country_data_processed_dir,"observed_belowavg_precip","chirps_monthly_total_precipitation.csv"))
#remove day part of date (day doesnt indicate anything with this data and easier for merge)
df_total_month.date_month=pd.to_datetime(df_total_month.date_month).dt.to_period("M")


#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])


#set the onset and cessation date for the seasons where these are missing 
#(meaning there was no dry spell data from start/till end of the season)
df_rain_filled=df_rain.copy()
df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))


df_rain_filled["onset_month"]=df_rain_filled["onset_date"].dt.to_period("M")
df_rain_filled["cessation_month"]=df_rain_filled["cessation_date"].dt.to_period("M")


#remove the adm2-date entries outside the rainy season for that specific adm2
#df_belowavg_month only includes data from 2000, so the 1999 entries are not included
list_hist_rain_adm2=[]
for a in df_rain_filled.ADM2_EN.unique():
    dates_adm2=pd.Index([])
    for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
        df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
        seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
        dates_adm2=dates_adm2.union(seas_range)
    list_hist_rain_adm2.append(df_total_month[(df_total_month.ADM2_EN==a)&(df_total_month.date_month.isin(dates_adm2))])
df_total_month_rain=pd.concat(list_hist_rain_adm2)


#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_comb_month=df_ds_drymonth.merge(df_total_month_rain,how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_comb_month.dry_spell_first_date=df_comb_month.dry_spell_first_date.replace(np.nan,0)
#date becomes binary
df_comb_month.rename(columns={"dry_spell_first_date":"dry_spell"},inplace=True)


df_comb_month["month"]=df_comb_month.date_month.dt.month


df_comb_month_labels=df_comb_month.replace({"dry_spell":{0:"No",1:"Yes"}})
cg_stats=["mean_cell","max_cell","min_cell","perc_threshold"]
num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_comb_month_labels,x=s,ax=ax,stat="count",kde=True,hue="dry_spell",palette=["#18998F","#FCE0DE"])
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Total monthly precipitation (mm)")


len(df_comb_month)


df_comb_month_sel=df_comb_month[df_comb_month.month.isin(df_comb_month[df_comb_month.dry_spell==1].month.unique())]


len(df_comb_month_sel)


df_comb_month_sel_labels=df_comb_month_sel.replace({"dry_spell":{0:"No",1:"Yes"}})
cg_stats=df_comb_month_sel.month.unique()#["mean_cell","max_cell","min_cell"]#,"perc_threshold"]
num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,10))
for i, s in enumerate(cg_stats):
    df_comb_month_sel_labels_s=df_comb_month_sel_labels[df_comb_month_sel_labels.month==s]
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_comb_month_sel_labels_s.sort_values("dry_spell",ascending=False),x="mean_cell",ax=ax,stat="count",kde=True,hue="dry_spell",palette=["#18998F","#FCE0DE"])
    ax.set_title(calendar.month_name[s])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Total monthly precipitation (mm)")


#check if difference per season
colp_num=3
num_plots=len(df_comb_month_sel.date_month.dt.month.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,m in enumerate(df_comb_month_sel.sort_values(by="date_month").date_month.dt.month.unique()):
    y_target =    df_comb_month_sel.loc[df_comb_month_sel.date_month.dt.month==m,"dry_spell"]
    y_predicted = y_predicted=np.where( df_comb_month_sel.loc[df_comb_month_sel.date_month.dt.month==m,"mean_cell"]<=100,1,0)
    
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    print(f"hit rate {calendar.month_name[m]}: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
    print(f"miss rate {calendar.month_name[m]}: {round(fp/(tp+fp)*100,1)}% ({fp}/{tp+fp})")
    ax = fig.add_subplot(rows,colp_num,i+1)
    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax)
    ax.set_ylabel("Dry spell in ADMIN2 during month")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
    ax.set_title(calendar.month_name[m])


df_ds.groupby("start_month").count()


df_feb=df_comb_month_sel[df_comb_month_sel.month==2]


y_target =  df_feb.dry_spell
threshold_list=np.arange(df_feb.mean_cell.min() - df_feb.mean_cell.min()%10,df_feb.mean_cell.max() - df_feb.mean_cell.max()%10,10)
df_pr=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted = np.where(df_feb.mean_cell<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
#     print(f"hit rate {t}mm: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
#     print(f"miss rate {t}mm: {round(fp/(tp+fp)*100,1)}% ({fp}/{tn+fp})")
    df_pr.loc[t,["hit rate","miss rate"]]=tp/(tp+fn)*100,fp/(tp+fp)*100


#plot timeseries for one adm2
#can see that trends are similair, but there are large differences, especially for the points of our interested, i.e. when observed <=2mm
from matplotlib.ticker import StrMethodFormatter
fig,ax=plt.subplots()
df_pr=df_pr.reset_index()
df_pr.plot(x="threshold",y="hit rate" ,figsize=(16, 8), color='red',legend=True,ax=ax,label="hit rate")
df_pr.plot(x="threshold",y="miss rate" ,figsize=(16, 8), color='#86bf91',legend=True,ax=ax,label="miss rate")

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Hit and miss rate for monthly precipitation during February")


y_target =  df_comb_month_sel.dry_spell
threshold_list=np.arange(df_comb_month_sel.mean_cell.min() - df_comb_month_sel.mean_cell.min()%10,df_comb_month_sel.mean_cell.max() - df_comb_month_sel.mean_cell.max()%10,10)
df_pr=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted = np.where(df_comb_month_sel.mean_cell<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
#     print(f"hit rate {t}mm: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
#     print(f"miss rate {t}mm: {round(fp/(tp+fp)*100,1)}% ({fp}/{tn+fp})")
    df_pr.loc[t,["hit_rate","miss_rate"]]=tp/(tp+fn)*100,fp/(tp+fp)*100
    #this doesn't really work to compute optimum threshold
#     df_pr.loc[t,"hit_miss"]=df_pr.loc[t,"hit_rate"]+(100-df_pr.loc[t,"miss_rate"])
    if t==100 or t==200:
        print(f"hit rate {t}mm: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
        print(f"miss rate {t}mm: {round(fp/(tp+fp)*100,1)}% ({fp}/{tp+fp})")
        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True)


df_pr.loc[100]


#### df_pr[df_pr.hit_miss==df_pr.hit_miss.max()]


#plot timeseries for one adm2
#can see that trends are similair, but there are large differences, especially for the points of our interested, i.e. when observed <=2mm
from matplotlib.ticker import StrMethodFormatter
fig,ax=plt.subplots()
df_pr=df_pr.reset_index()
df_pr.plot(x="threshold",y="hit rate" ,figsize=(16, 8), color='red',legend=True,ax=ax,label="hit rate")
df_pr.plot(x="threshold",y="miss rate" ,figsize=(16, 8), color='#86bf91',legend=True,ax=ax,label="miss rate")

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Hit and miss rate for monthly precipitation")


t=100
df_feb["below_t"]=np.where(df_feb.mean_cell<=t,1,0)
df_numadm=df_feb.groupby("date_month").agg({"dry_spell":"sum","below_t":"sum"})#,"dry_spell":"count"})


df_numadm=df_comb_month_sel[df_comb_month_sel.mean_cell<=100].groupby("date_month").agg({"dry_spell":"sum","ADM2_EN":"count"})#,"dry_spell":"count"})


df_ds_res=df_ds.reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_daterange=df_ds_res[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_ds_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_ds_daterange["dryspell_obs"]=1


import itertools


df_daterange=pd.DataFrame(list(itertools.product(pd.date_range("2000-01-01","2020-12-31",freq="D"),df_ds.ADM2_EN.unique())),columns=['date','ADM2_EN'])


df_daterange_ds=df_daterange.merge(df_ds_daterange,on=["date","ADM2_EN"],how="left")


df_daterange_ds[df_daterange_ds.dryspell_obs==1]


pd.to_datetime(df_comb_month.date)


t=100
df_comb_month_be=df_comb_month[df_comb_month.mean_cell<=t]
df_comb_month_be["first_date"]=pd.to_datetime(df_comb_month_be.date)
df_comb_month_be["last_date"]=df_comb_month_be.date_month.dt.to_timestamp("M")
df_comb_month_be_res=df_comb_month_be.reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_comb_month_be_res[['first_date', 'last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_precip_daterange=df_comb_month_be_res[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_precip_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_precip_daterange["precip_below"]=1


df_daterange_comb=df_daterange_ds.merge(df_precip_daterange,on=["date","ADM2_EN"],how="left")


df_daterange_comb.dryspell_obs=df_daterange_comb.dryspell_obs.replace(np.nan,0)
df_daterange_comb.precip_below=df_daterange_comb.precip_below.replace(np.nan,0)


def label_ds(row):
    if row["dryspell_obs"]==1 and row["precip_below"]==1:
        return 3
    elif row["dryspell_obs"]==1:
        return 2
    elif row["precip_below"]==1:
        return 1
    else:
        return 0


#encode dry spells and whether it was none, only observed, only forecasted, or both
df_daterange_comb["dryspell_match"]=df_daterange_comb.apply(lambda row:label_ds(row),axis=1)


df_adm2=gpd.read_file(adm2_bound_path)
df_daterange_comb=df_daterange_comb.merge(df_adm2[["ADM2_EN","ADM2_PCODE"]])
df_daterange_comb.rename(columns={"ADM2_PCODE":"pcode"},inplace=True)


df_daterange_comb.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"monthly_dryspellobs_th100.csv"))


df_numadm[df_numadm.dry_spell==1]


sns.regplot(data = df_numadm, x = 'dry_spell', y = 'ADM2_EN', fit_reg = False,
            scatter_kws = {'alpha' : 1/3},x_jitter = 0.2, y_jitter = 0.2)


# ### Seasonal

df_total_seas=pd.read_csv(os.path.join(country_data_processed_dir,"observed_belowavg_precip","chirps_seasonal_total_precipitation.csv"))
#remove day part of date (day doesnt indicate anything with this data and easier for merge)
df_total_seas.date_month=pd.to_datetime(df_total_seas.date_month).dt.to_period("M")


# #remove the adm2-date entries outside the rainy season for that specific adm2
# #df_total_seas only includes data from 2000, so the 1999 entries are not included
# list_hist_rain_adm2=[]
# for a in df_rain_filled.ADM2_EN.unique():
#     dates_adm2=pd.Index([])
#     for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
#         df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
#         seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
#         dates_adm2=dates_adm2.union(seas_range)
#     list_hist_rain_adm2.append(df_total_seas[(df_total_seas.ADM2_EN==a)&(df_total_seas.date_month.isin(dates_adm2))])
# df_total_seas_rain=pd.concat(list_hist_rain_adm2)


df_total_seas_rain=df_total_seas.copy()


#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_drymonth_rain=df_ds_drymonth.merge(df_total_seas_rain[["ADM2_EN","date_month"]],how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_ds_drymonth_rain.dry_spell_first_date=df_ds_drymonth_rain.dry_spell_first_date.replace(np.nan,0)


#fill the data frame to include all months, also outside the rainy season --> this enables us to take the rolling sum 
#(else e.g. the rolling sum for Nov might include May-June-Nov)
df_ds_drymonth_alldates=df_ds_drymonth_rain.sort_values("date_month").set_index("date_month").groupby('ADM2_EN').resample('M').sum().drop("ADM2_EN",axis=1).reset_index()


#compute the rolling sum of months having a dry spell per admin2
s_ds_dryseas=df_ds_drymonth_alldates.sort_values("date_month").set_index("date_month").groupby('ADM2_EN')['dry_spell_first_date'].rolling(3).sum()
#convert series to dataframe
df_ds_dryseas=pd.DataFrame(s_ds_dryseas).reset_index().sort_values(["ADM2_EN","date_month"])
df_ds_dryseas.rename(columns={"dry_spell_first_date":"num_dry_spell_seas"},inplace=True)


#merge the dry spells with the info if a month had below average rainfall
#merge on right such that only the dates within the rainy season are included, df_ds_dryseas also includes all other months
df_comb_seas=df_ds_dryseas.merge(df_total_seas_rain,how="right",on=["date_month","ADM2_EN"])


#remove dates where dry_spell_confirmation is nan, i.e. where rolling sum could not be computed for (first dates)
df_comb_seas=df_comb_seas[df_comb_seas.num_dry_spell_seas.notna()]


#set the occurence of a dry spell to true if in at least one of the months of the season (=3 months) a dry spell occured
df_comb_seas["dry_spell"]=np.where(df_comb_seas.num_dry_spell_seas>=1,1,0)


df_comb_seas["month"]=df_comb_seas.date_month.dt.month


df_comb_seas[(df_comb_seas.dry_spell==1)&(df_comb_seas.date_month.dt.month==4)]


df_comb_seas_labels=df_comb_seas.replace({"dry_spell":{0:"No",1:"Yes"}})
cg_stats=["mean_cell","max_cell","min_cell"]#,"perc_threshold"]
num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,10))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_comb_seas_labels.sort_values("dry_spell",ascending=False),x=s,ax=ax,stat="count",kde=True,hue="dry_spell",palette=["#18998F","#FCE0DE"])
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Total seasonal precipitation (mm)")


df_comb_seas_sel=df_comb_seas[df_comb_seas.month.isin(df_comb_seas[df_comb_seas.dry_spell==1].month.unique())]


df_comb_seas_sel[(df_comb_seas_sel.dry_spell==1)&(df_comb_seas_sel.date_month.dt.month==4)]


threshold_list=[300,400,600,800,1000]
df_pr=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted=np.where(df_comb_seas_sel["mean_cell"]<=t,1,0)
    y_target =    df_comb_seas_sel["dry_spell"]
#y_predicted = df_comb_seas["precip_se200"]
    print(confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted))
    print(confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted).flatten())
    df_pr.loc[t,["tn","fp","fn","tp"]]=confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted).flatten()


seasons_rolling={3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",12:"SON",12:"OND",1:"NDJ",2:"DJF"}


df_comb_seas_sel["season_abbr"]=df_comb_seas_sel.month.map(seasons_rolling)


df_comb_seas_sel[(df_comb_seas_sel.dry_spell==1)&(df_comb_seas_sel.season_abbr=="FMA")]


df_comb_seas_sel_labels=df_comb_seas_sel.replace({"dry_spell":{0:"No",1:"Yes"}})
cg_stats=df_comb_seas_sel.season_abbr.unique()#["mean_cell","max_cell","min_cell"]#,"perc_threshold"]
num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,10))
for i, s in enumerate(cg_stats):
    df_comb_seas_sel_labels_s=df_comb_seas_sel_labels[df_comb_seas_sel_labels.season_abbr==s]
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_comb_seas_sel_labels_s.sort_values("dry_spell",ascending=False),x="mean_cell",ax=ax,stat="count",kde=True,hue="dry_spell",palette=["#18998F","#FCE0DE"])
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Total seasonal precipitation (mm)")


df_ds["year"]=df_ds.dry_spell_first_date.dt.year


df_ds.groupby("season_approx").count()


df_comb_seas_sel[(df_comb_seas_sel.season_abbr=="FMA")&(df_comb_seas_sel.mean_cell<=250)].groupby("date_month").agg({"dry_spell":["sum","count"]})


df_comb_seas_sel[(df_comb_seas_sel.season_abbr=="FMA")&(df_comb_seas_sel.dry_spell==1)&(df_comb_seas_sel.mean_cell>250)]#.groupby("date_month").agg({"dry_spell":["sum","count"]})


33/(19+33)


19/(19+13)


#check if difference per season
colp_num=3
num_plots=len(df_comb_seas_sel.season_abbr.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,m in enumerate(df_comb_seas_sel.season_abbr.unique()):
    y_target =    df_comb_seas_sel.loc[df_comb_seas_sel.season_abbr==m,"dry_spell"]
    y_predicted = np.where( df_comb_seas_sel.loc[df_comb_seas_sel.season_abbr==m,"mean_cell"]<=250,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    ax = fig.add_subplot(rows,colp_num,i+1)
    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax)
    ax.set_ylabel("Dry spell in ADMIN2 during season")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
    ax.set_title(m)


# df_pr.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"seasonal_confusionmatrix_thresholds.csv"))


df_pr


15/(151+15)


df_pr["ratio_hit"]=df_pr["tp"]/(df_pr["tp"]+df_pr["fp"])
df_pr["ratio_miss"]=df_pr["fn"]/(df_pr["fn"]+df_pr["tp"])


per seas
num with and without dry spells


y_predicted=np.where(df_comb_seas_sel["mean_cell"]<=500,1,0)
y_target =    df_comb_seas_sel["dry_spell"]
tn,fp,fn,tp=confusion_matrix(y_target=y_target, 
                  y_predicted=y_predicted).flatten()





df_comb_seas_sel




