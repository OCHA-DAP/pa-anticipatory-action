#!/usr/bin/env python
# coding: utf-8

# ### Observed dryspells and correlation with forecasted dry spells
# Process the observed dryspell list as outputed by `malawi/scripts/mwi_chirps_dry_spell_detection.R` and correlate the occurence of a dry spell with the forecasted CHIRPS-GEFS dry spell
# 
# As first analysis we are focussing on the sole occurence of a dry spell per admin2. This can be extended to e.g. duration, number of dry spells, and geographical spread
# 
# For simplicity we are only focusing on forecasting the first 15 days of a dry spell, i.e. the start date of the dry spell corresponds with the start date of the forecast   
# This might be too simple, e.g. might want to forecast if a dry spell that already started will persist.
# 
# For now this means that if the start_date of the forecast occurs in the dry spell list, we classify it is a true positive
# 
# Assumptions
# - We want to forecast the start of a dry spell, not if a dry spell persists
# - The grid cell size is small enough to only look at cells with their centre within the region, not those touching

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
import calendar


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
# print(path_mod)
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp,download_url
from src.utils_general.raster_manipulation import fix_calendar, invert_latlon, change_longitude_range
from src.utils_general.plotting import plot_raster_boundaries_clip,plot_spatial_columns


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


# #### Load CHIRPS-GEFS data

hist_path=os.path.join(country_data_exploration_dir,"chirpsgefs","mwi_chirpsgefs_rainyseas_stats_mean.csv")
df_chirpsgefs=pd.read_csv(hist_path)
df_chirpsgefs["date"]=pd.to_datetime(df_chirpsgefs["date"])#.dt.to_period("d")
df_chirpsgefs["date_forec_end"]=pd.to_datetime(df_chirpsgefs["date_forec_end"])#.dt.to_period("d")


#05-03-2021: chirpsgefs data before 2010 isn't yet complete--> focus on data from 2010 for now to circumvent biases
df_chirpsgefs=df_chirpsgefs[df_chirpsgefs.date.dt.year>=2010]


#most dates are already removed in prepocessing notebook, but since dfeinition of rainy season changed, recomputing them


# #path to data start and end rainy season
# df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean.csv"))
# df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])#.dt.to_period('d')
# df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])#.dt.to_period('d')


# #set the onset and cessation date for the seasons where these are missing 
# #(meaning there was no dry spell data from start/till end of the season)
# df_rain_filled=df_rain.copy()
# df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
# df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
# df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))


# #remove the adm2-date entries outside the rainy season for that specific adm2
# #df_belowavg_seas only includes data from 2000, so the 1999 entries are not included
# list_hist_rain_adm2=[]
# for a in df_rain_filled.ADM2_EN.unique():
#     dates_adm2=pd.Index([])
#     for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
#         df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
#         seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="D")
#         dates_adm2=dates_adm2.union(seas_range)
#     list_hist_rain_adm2.append(df_chirpsgefs[(df_chirpsgefs.ADM2_EN==a)&(df_chirpsgefs.date.isin(dates_adm2))])
# df_chirpsgefs=pd.concat(list_hist_rain_adm2)


cg_stats=["max_cell","mean_cell","min_cell","perc_se2","perc_se10"]


len(df_chirpsgefs)


len(df_chirpsgefs)


len(df_chirpsgefs.date.unique())


df_chirpsgefs.date.dt.month.unique()


len(df_chirpsgefs.ADM2_EN.unique())


df_chirpsgefs


num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_chirpsgefs,x=s,ax=ax)
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# print("number of adm2-date combination with max cell <=2mm:",len(df_chirpsgefs[df_chirpsgefs.max_cell<=2]))
# print("number of dates with at least one adm2 with max cell <=2mm:",len(df_chirpsgefs[df_chirpsgefs.max_cell<=2].date.unique()))
# print("number of adm2-date combination with mean cell <=2mm:",len(df_chirpsgefs[df_chirpsgefs.mean_cell<=2]))
# print("number of dates with at least one adm2 with mean cell <=2mm:",len(df_chirpsgefs[df_chirpsgefs.mean_cell<=2].date.unique()))


# #### Load observed dry spells

#different methodologies of computing the observed dry spells 
#set the one being used, dryspell_obs_suffix determines which file to load
dryspell_obs_stat="mean" #"max"
if dryspell_obs_stat=="max":
    dryspell_obs_suffix=""
else:
    dryspell_obs_suffix=f"_{dryspell_obs_stat}"


df_ds=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells",f"dry_spells_during_rainy_season_list_2000_2020{dryspell_obs_suffix}.csv"))
df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
df_ds["year"]=df_ds.dry_spell_first_date.dt.year


df_ds


df_ds["dry_spell_extension"]=df_ds.dry_spell_duration-14
print(df_ds.dry_spell_extension.sum())


sns.histplot(df_ds,x="dry_spell_duration")


#number of historically observed dry spells
#this is waay less than the predicted..
len(df_ds)


len(df_ds[df_ds.dry_spell_first_date.dt.year>=2010])


#chirpsgefs 2020 data is not complete, so these might be removed
len(df_ds[df_ds.dry_spell_first_date.dt.year==2020])


df_ds_adm2=df_ds.groupby(["ADM2_EN"],as_index=False).count()
df_bound_adm2=gpd.read_file(adm2_bound_path)
gdf_ds_adm2=df_bound_adm2.merge(df_ds_adm2)#.merge(df_bound_adm2)

fig=plot_spatial_columns(gdf_ds_adm2, ["pcode"], title="Number of dry spells", predef_bins=None,cmap='YlOrRd',colp_num=1)


#check correlation with size of area and number of dry spells
g=sns.regplot(data=gdf_ds_adm2,y="pcode",x="Shape_Area",scatter_kws = {'alpha' : 1/3},fit_reg=False)
ax=g.axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("Number of dry spells")


#compute years with a dry spell compared to total number of dry spells per admin2
# df_ds_adm2_year=df_ds.drop_duplicates(["year","ADM2_EN"]).groupby("ADM2_EN",as_index=False).count()
# gdf_ds_adm2_year=df_bound_adm2.merge(df_ds_adm2_year)
# fig=plot_spatial_columns(gdf_ds_adm2_year, ["pcode"], title="Number of dry spells", predef_bins=None,cmap='YlOrRd',colp_num=1)


# ### Observed dryspells and correlation with forecasted dry spells

#combine chirpsgefs and observed dryspells data
#merge on right to include all adm2-dates present in chirpsgefs
#df_chirpsgefs, only includes the dates per adm2 that were in a rainy season
df_comb=df_ds.merge(df_chirpsgefs[["ADM2_EN","date","date_forec_end"]+cg_stats],how="right",left_on=["dry_spell_first_date","ADM2_EN"],right_on=["date","ADM2_EN"])


df_comb


#nan = there was a forecast but no observed dry spell--> set occurence of dry spell to zero
#again, only looking at if any dry spell started on that date, not whether it persisted
df_comb["dryspell_obs"]=np.where(df_comb.dry_spell_first_date.notna(),1,0)


# Aaah somehow dry spells went missing when merging on the forecast list
#reason not all data till 2010 is included yet and part of data 2020 is missing..
#TODO: check how (might have to do with missing data in 2020..)
len(df_comb[df_comb.dryspell_obs==1])


num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_comb,x=s,ax=ax,stat="density",common_norm=False,kde=True,hue="dryspell_obs")
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_comb,x=s,ax=ax,stat="count",kde=True,hue="dryspell_obs")
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# #### Plot raw raster values
# Of CHIRPS-GEFS for dates where there is an observed dry spell

#load the raster data
ds_list=[]
for d in df_comb[df_comb.dryspell_obs==1].date.unique():
    d_str=pd.to_datetime(d).strftime("%Y%m%d")
    filename=f"chirpsgefs_africa_{d_str}.tif"
    rds=rioxarray.open_rasterio(os.path.join(chirpsgefs_dir,filename))
    rds=rds.assign_coords({"time":pd.to_datetime(d)})
    rds=rds.sel(band=1)
    ds_list.append(rds)

ds_drys=xr.concat(ds_list,dim="time")

ds_drys=ds_drys.sortby("time")


ds_list[0]


#plot the rasters. Plot per adm2
ds_drys_clip = ds_drys.rio.clip(df_bound_adm2.geometry.apply(mapping), df_bound_adm2.crs, all_touched=True)
bins=np.arange(0,101,10)

# ds_drys_clip.values[~np.isnan(ds_drys_clip.values)].max()

df_comb_ds=df_comb[df_comb.dryspell_obs==1]
for a in df_comb_ds.ADM2_EN.unique():
    print(a)
    df_bound_sel_adm=df_bound_adm2[df_bound_adm2.ADM2_EN==a]
    ds_drys_clip_adm = ds_drys.rio.clip(df_bound_sel_adm.geometry.apply(mapping), df_bound_sel_adm.crs, all_touched=True)
    ds_drys_clip_adm_dates=ds_drys_clip_adm.sel(time=ds_drys_clip_adm.time.isin(df_comb_ds[df_comb_ds.ADM2_EN==a].date.unique()))
    #cannot make the facetgrid if only one occurence. For now leave them out since just exploration, but for completeness should somehow include them
    if len(ds_drys_clip_adm_dates.time)>1:
        g=ds_drys_clip_adm_dates.plot(
        col="time",
        col_wrap=6,
        levels=bins,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "label":"Precipitation (mm)",
            "ticks": bins
        },
        cmap="YlOrRd",
    )

        for ax in g.axes.flat:
            df_bound_sel_adm.boundary.plot(linewidth=1, ax=ax, color="red")
            ax.axis("off")
        g.fig.suptitle(f"{a}")# {df_comb_ds[df_comb_ds.ADM2_EN==a].sort_values(by='date').dryspell_forec.values}")


# ### Discrepancy forecasted and observed rainfall

#read historically observed 14 day rolling sum for all dates (so not only those with dry spells)
df_histobs=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","data_mean_values_long.csv"))
df_histobs.date=pd.to_datetime(df_histobs.date)

#add start of the rolling sum 
df_histobs["date_start"]=df_histobs.date-timedelta(days=14)


#add adm2 name
df_histobs=df_histobs.merge(df_bound_adm2[["ADM1_EN","ADM2_EN","ADM2_PCODE"]],left_on="pcode",right_on="ADM2_PCODE")


#merge forecast and observed
#only include dates within the rainy season
df_histformerg=df_histobs.merge(df_chirpsgefs,how="right",left_on=["date_start","ADM2_EN"],right_on=["date","ADM2_EN"],suffixes=("obs","forec"))


df_histformerg["diff_precip"]=df_histformerg["rollsum_15d"]-df_histformerg["mean_cell"]
df_histformerg["diff_forecobs"]=df_histformerg["mean_cell"]-df_histformerg["rollsum_15d"]


g=sns.regplot(data=df_histformerg,y="diff_forecobs",x="rollsum_15d",scatter_kws = {'alpha' : 1/3},fit_reg=False)
ax=g.axes
ax.set_xlabel("Observed 15 day sum (mm)")
ax.set_ylabel("Forecasted 15 day sum - Observed 15 day sum")
ax.set_title("Discrepancy observed and forecasted values in Malawi per admin2")
ax.axhline(0, ls='--',color="red",label="obs=forec")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(0,ax.get_xlim()[1])
# ax.plot(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),np.linspace(-ax.get_xlim()[0],-ax.get_xlim()[1],50),ls="--",label="forec=0")
plt.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_CHIRPSGEFS_CHIRPS.png"))


bins = np.arange(0,df_histformerg.rollsum_15d.max(),10)
group = df_histformerg.groupby(pd.cut(df_histformerg.rollsum_15d, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.mean()
g=sns.jointplot(data=df_histformerg,y="diff_forecobs",x="rollsum_15d", kind="hex",height=16,joint_kws={ 'bins':'log'})
# ax=g.axes
g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
g.ax_joint.plot(plot_centers,plot_values,color="green",label="mean")
g.ax_joint.legend()
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
# ax.set_xlabel("Observed 15 day sum (mm)")
# ax.set_ylabel("Forecasted 15 day sum - Observed 15 day sum")
# ax.set_title("Discrepancy observed and forecasted values in Malawi per admin2")
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_density.png"))


len(df_histformerg[df_histformerg.rollsum_15d<=2])


len(df_histformerg[df_histformerg.rollsum_15d<=2])


len(df_histformerg)


df_histformerg.columns


df_histformerg.groupby("during_rainy_season_bin").sum()


df_histformerg[df_histformerg.during_rainy_season_bin==0].dateobs.dt.month.min()


df_histformerg[df_histformerg.during_rainy_season_bin==0][["dateobs","ADM2_EN"]]


len(df_histformerg[df_histformerg.mean_cell<=2])


df_sel=df_histformerg[df_histformerg.rollsum_15d<=2].sort_values("rollsum_15d")
bins = np.arange(0,df_sel.rollsum_15d.max()+2,0.2)
group = df_sel.groupby(pd.cut(df_sel.rollsum_15d, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.mean()
g=sns.jointplot(data=df_sel,y="diff_forecobs",x="rollsum_15d", kind="hex",height=16,joint_kws={ 'bins':'log'})
# ax=g.axes
g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
# g.ax_joint.plot(df_sel.rollsum_15d,df_sel.diff_forecobs,alpha=0.3)
g.ax_joint.plot(plot_centers,plot_values,color="green",label="mean")#,legend=True)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# ax.set_xlabel("Observed 15 day sum (mm)")
# ax.set_ylabel("Forecasted 15 day sum - Observed 15 day sum")
# ax.set_title("Discrepancy observed and forecasted values in Malawi per admin2")
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_density2mm.png"))


# g=sns.regplot(data=df_histformerg,y="diff_precip",x="rollsum_15d",scatter_kws = {'alpha' : 1/3},fit_reg=False)
# ax=g.axes
# ax.set_xlabel("Observed 15 day sum (mm)")
# ax.set_ylabel("Observed 15 day sum - Forecasted 15 day sum")
# ax.set_title("Discrepancy observed and forecasted values")
# ax.axhline(0, ls='--',color="red",label="Obs=Forec")
# print(ax.get_xlim())
# ax.set_xlim(0,ax.get_xlim()[1])
# ax.plot(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),ls="--",label="Forec=0")
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.legend()


# ### Correlation different thresholds

#compute the contigency table when using mean cell for different thresholds
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
t_list=[2,4,6,8,10]
num_plots = len(t_list)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,t in enumerate(t_list):
    y_target =    df_comb["dryspell_obs"]
    y_predicted = np.where(df_comb["mean_cell"]<=t,1,0)
    ax = fig.add_subplot(rows,colp_num,i+1)
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)

    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
    ax.set_ylabel("Observed dry spell in ADMIN2")
    ax.set_xlabel("Forecasted dry spell in ADMIN2")
    ax.set_title(f"mean cell<={t}mm")


#set definition of forecasted dry spell as max cell having not more than 2 mm of rains in 15 days period
df_comb["dryspell_forec"]=np.where(df_comb.mean_cell<=2,1,0)


#number of adm2s with dryspell for forecasted and observed over time
#the diagonal lines are caused by the last day of a rainy season having a forecasted dry spell
from matplotlib.ticker import StrMethodFormatter
fig,ax=plt.subplots()
df_comb_date=df_comb.groupby("date",as_index=False).sum()
df_comb_date.sort_values(by="date").plot(x="date",y="dryspell_forec" ,figsize=(16, 8), color='red',legend=True,ax=ax)
df_comb_date.sort_values(by="date").plot(x="date",y="dryspell_obs" ,figsize=(16, 8), color='#86bf91',legend=True,ax=ax)

# Set x-axis label
ax.set_xlabel("Start date", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Dry spell", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Observed and forecasted dryspells")


#understand monthly patterns observed vs forecasted
df_comb["month"]=df_comb.date.dt.month


#observed dry spells per month
sns.displot(df_comb[df_comb.dryspell_obs==1],bins=range(df_comb.month.min(),df_comb.month.max()+1),x="month")


#forecasted dry spells per month
sns.displot(df_comb[df_comb.dryspell_forec==1],bins=range(df_comb.month.min(),df_comb.month.max()+1),x="month")


# ### Correlations

# None of the mean cell boundaries really captures the dry spell and do results in lot of false positives.. When using the max, it doesn't get better
# 
# ##### Understand differences patterns observed and forecasted dry spells
# - Dry spell is at least 14 days while forecasted period is 15 days --> can be a reason for underpredicting
# - [WHEN USING MAX TO COMPUTE OBSERVED DRY SPELLS] Definition of observed dry spell is stricter than forecasted --> can be a reason for overpredicting
# 

#sum of dryspells per adm2, doesn't mean they occured on the same dates
#could add fraction obs/forec
df_comb.groupby("ADM2_EN").sum()[["dryspell_forec","dryspell_obs"]]#.date.sort_values().unique()


# df_comb.drop_duplicates("ADM2_EN").groupby("ADM1_EN").count()


#compute the contigency table per month
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

for m in df_comb.sort_values(by="month").month.unique():

    y_target =    df_comb.loc[df_comb.month==m,"dryspell_obs"]
    y_predicted = df_comb.loc[df_comb.month==m,"dryspell_forec"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
#     print(cm)
#     print(calendar.month_name[m])
#     print(f"contigency matrix for {pd.to_datetime(m).strftime('%B')}")
    fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
    ax.set_ylabel("Observed dry spell in ADMIN2")
    ax.set_xlabel("Forecasted dry spell in ADMIN2")
#     print(m)
    ax.set_title(f"contigency matrix for {calendar.month_name[m]}")
    plt.show()


# #### correlation number of admin2s with observed and forecasted dry spell

#compute number of adm2s with dryspell per date
df_numadm=df_comb.groupby("date")[["dryspell_obs","dryspell_forec"]].sum()


df_numadm


#darker shade means higher density of values
#-->not really clear pattern..
sns.regplot(data = df_numadm, x = 'dryspell_obs', y = 'dryspell_forec', fit_reg = False,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2)


df_comb=df_comb.merge(df_bound_adm2[["ADM1_EN","ADM2_EN"]],on="ADM2_EN",how="left")
df_numadm1=df_comb.groupby(["date","ADM1_EN"],as_index=False)[["dryspell_obs","dryspell_forec"]].sum()

for a in df_numadm1.ADM1_EN.unique():
    g=sns.regplot(data = df_numadm1[df_numadm1.ADM1_EN==a], x = 'dryspell_obs', y = 'dryspell_forec', fit_reg = False,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2)
    g.axes.set_title(a)
    plt.show()


#compute the contigency table per date
#Damnn that correlation between obsrved and forecasted is waay less than I expected.. 
#Need to further understand why and if this is really the case or bug somewhere
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

df_comb_date=df_comb.groupby("date",as_index=False).sum()
df_comb_date["dryspell_obs_bool"]=np.where(df_comb_date.dryspell_obs>=1,1,0)
df_comb_date["dryspell_forec_bool"]=np.where(df_comb_date.dryspell_forec>=1,1,0)

y_target =    df_comb_date["dryspell_obs_bool"]
y_predicted = df_comb_date["dryspell_forec_bool"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Observed dry spell in ADMIN2")
ax.set_xlabel("Forecasted dry spell in ADMIN2")
plt.show()


#compute the contigency table per date
#Damnn that correlation between obsrved and forecasted is waay less than I expected.. 
#Need to further understand why and if this is really the case or bug somewhere
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

df_comb["date_month"]=df_comb.date.dt.to_period("M")
df_comb_month=df_comb.groupby("date_month",as_index=False).sum()
df_comb_month["dryspell_obs_bool"]=np.where(df_comb_month.dryspell_obs>=1,1,0)
df_comb_month["dryspell_forec_bool"]=np.where(df_comb_month.dryspell_forec>=1,1,0)

y_target =    df_comb_month["dryspell_obs_bool"]
y_predicted = df_comb_month["dryspell_forec_bool"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Observed dry spell in ADMIN2")
ax.set_xlabel("Forecasted dry spell in ADMIN2")
plt.show()


# ### Correlation 15 day rolling sum

#read historically observed 14 day rolling sum for all dates (so not only those with dry spells)
df_histobs=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","data_mean_values_long.csv"))
df_histobs.date=pd.to_datetime(df_histobs.date)

#add start of the rolling sum 
df_histobs["date_start"]=df_histobs.date-timedelta(days=14)


#add adm2 name
df_histobs=df_histobs.merge(df_bound_adm2[["ADM1_EN","ADM2_EN","ADM2_PCODE"]],left_on="pcode",right_on="ADM2_PCODE")


#merge forecast and observed
#only include dates within the rainy season
df_roll=df_histobs.merge(df_chirpsgefs,how="right",left_on=["date_start","ADM2_EN"],right_on=["date","ADM2_EN"],suffixes=("obs","forec"))


df_roll.columns


df_roll["dryspell_obs"]=np.where(df_roll.rollsum_15d<=2,1,0)


num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_roll,x=s,ax=ax,stat="density",common_norm=False,kde=True,hue="dryspell_obs")
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_roll,x=s,ax=ax,stat="count",kde=True,hue="dryspell_obs")
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


#compute the contigency table when using mean cell for different thresholds
t_list=[2,4,6,8,10]
num_plots = len(t_list)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,t in enumerate(t_list):
    y_target =    df_roll["dryspell_obs"]
    y_predicted = np.where(df_roll["mean_cell"]<=t,1,0)
    ax = fig.add_subplot(rows,colp_num,i+1)
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)

    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
    ax.set_ylabel("Observed 15day sum<=2mm in ADMIN2")
    ax.set_xlabel("Forecasted dry spell in ADMIN2")
    ax.set_title(f"mean cell<={t}mm")


#compute the contigency table when using mean cell for different thresholds
t_list=[2,4,6,8,10]
num_plots = len(t_list)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,t in enumerate(t_list):
    y_target =    df_roll["dryspell_obs"]
    y_predicted = np.where(df_roll["max_cell"]<=t,1,0)
    ax = fig.add_subplot(rows,colp_num,i+1)
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)

    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
    ax.set_ylabel("Observed dry spell in ADMIN2")
    ax.set_xlabel("Forecasted dry spell in ADMIN2")
    ax.set_title(f"max cell<={t}mm")


df_roll["dryspell_forec"]=np.where(df_roll.mean_cell<=2,1,0)


#number of adm2s with dryspell for forecasted and observed over time
#the diagonal lines are caused by the last day of a rainy season having a forecasted dry spell
from matplotlib.ticker import StrMethodFormatter
fig,ax=plt.subplots()
df_roll_date=df_roll.groupby("dateforec",as_index=False).sum()
df_roll_date.sort_values(by="dateforec").plot(x="dateforec",y="dryspell_forec" ,figsize=(16, 8), color='red',legend=True,ax=ax)
df_roll_date.sort_values(by="dateforec").plot(x="dateforec",y="dryspell_obs" ,figsize=(16, 8), color='#86bf91',legend=True,ax=ax)

# Set x-axis label
ax.set_xlabel("Start date", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Dry spell", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Observed and forecasted dryspells")


# ### Overlap dry spells observed and forecasted

df_ds


#create df with all dates that were part of a dry spell per adm2
#only including data since 2010 since this is the chirps data we are currently using
df_ds_2010=df_ds[df_ds.dry_spell_first_date.dt.year>=2010].sort_values(["ADM2_EN","dry_spell_first_date"]).reset_index(drop=True)
#assign an ID to each dry spell, such that we can group by that later on
df_ds_2010["ID_obs"]=range(1,len(df_ds_2010)+1)
#create datetimeindex per row
a = [pd.date_range(*r, freq='D') for r in df_ds_2010[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_daterange=df_ds_2010[["ADM2_EN","ID_obs"]].join(pd.DataFrame(a)).set_index(["ADM2_EN","ID_obs"]).stack().droplevel(-1).reset_index()
df_ds_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell
df_ds_daterange["dryspell_obs"]=1


df_ds_2010[df_ds_2010.ADM2_EN=="Blantyre City"]


df_ds_daterange[df_ds_daterange.ID_obs==5]


len(df_ds_daterange)


df_ds_daterange=df_ds_daterange.drop_duplicates()


#there shouldn't be any duplicates so check if two lengths match
len(df_ds_daterange)


#select all dates where chirps-gefs forecasts dry spell
df_cg_ds=df_chirpsgefs[df_chirpsgefs.mean_cell<=2]


# df_cg_ds.sort_values("date").groupby("ADM2_EN").date.diff().dt.days.ne(1).cumsum()


#create list of dates that were within a forecast that predicted a dry spell
df_cg_ds=df_cg_ds.sort_values(["ADM2_EN","date"]).reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_cg_ds[['date', 'date_forec_end']].values]
df_cg_daterange=df_cg_ds[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_cg_daterange.rename(columns={0:"date"},inplace=True)
df_cg_daterange["dryspell_forec"]=1


df_cg_daterange


len(df_cg_daterange)


#with the forecast it is expected to be a lot of overlap
df_cg_daterange=df_cg_daterange.drop_duplicates()


#still many more dates forecasted to be part of a dry spell than the observed..
len(df_cg_daterange)


len(df_ds_daterange)


#assign ID to each forecasted dry spell
#we assign one ID per any overlapping range
#e.g. if dry spell forecasted on 01-01 and 03-01 (but not 02-01), we would see this as one dry spell from 01-01 till 17-01
df_cg_daterange["ID_forec"]=df_cg_daterange.sort_values(["ADM2_EN","date"]).groupby("ADM2_EN").date.diff().dt.days.ne(1).cumsum()


#create dataframe with all dates, where for each date it is added whether a dry spell was observed and/or forecasted
#include all dates for which we got chirpsgefs data
df_dates=df_chirpsgefs[["date","ADM2_EN"]].sort_values(["ADM2_EN","date"])


#merge the observed dry spells
#merge on left, such to only include dates for which a forecast is available
df_dates=df_dates.merge(df_ds_daterange,how="left",on=["date","ADM2_EN"])
df_dates.dryspell_obs=df_dates.dryspell_obs.replace(np.nan,0)


#AAHH lots of obs dryspell dates went missing...
len(df_dates[df_dates.dryspell_obs==1])


# common = df_ds_daterange.merge(df_dates,on=['date','ADM2_EN'])
# print(common)
# df_ds_daterange[(~df_ds_daterange.date.isin(common.date))&(~df_ds_daterange.ADM2_EN.isin(common.ADM2_EN))]


# df_ds_daterange[["date","ADM2_EN"]].isin(df_dates[["date","ADM2_EN"]])


# df_ds_daterange[["date","ADM2_EN"]][~df_ds_daterange[["date","ADM2_EN"]].isin(df_dates[["date","ADM2_EN"]])].dropna(how="any")


#merge the forecasted dry spells
df_dates=df_dates.merge(df_cg_daterange,how="left",on=["date","ADM2_EN"])
df_dates.dryspell_forec=df_dates.dryspell_forec.replace(np.nan,0)


df_dates


# df_dates[df_dates.dryspell_obs.isnull()]


y_target =    df_dates["dryspell_obs"]
y_predicted = df_dates["dryspell_forec"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
# ax.set_ylabel("Day within observed dry spell")
# ax.set_xlabel("Day within forecasted dry spell")
plt.show()


#some complete dry spells went missing --> is because of 2020 missing data..
df_dates.sort_values("ID_obs").ID_obs.unique()


# df_ds_2010[df_ds_2010.ID_obs.isin([4,6,11,14,36,40])]


df_dates[df_dates.ID_obs==2]


# df_dates.groupby("ID_obs").dryspell_forec.apply(lambda x: np.argmax(x.values))#.ID_forec.first_valid_index()


# df_dates[df_dates.ID_obs==1].ID_forec.first_valid_index()#s.index.get_loc(s.first_valid_index())


df_dsid=df_dates.groupby("ID_obs").sum()


#-1 indicates no overlap
df_dsid["days_late"]=df_dates.groupby("ID_obs").ID_forec.apply(lambda x: x.argmax(skipna=True))#.ID_forec.first_valid_index()
df_dsid["days_late"]=df_dsid["days_late"].replace(-1,np.nan)


df_dsid


round(len(df_dsid[df_dsid.dryspell_forec>0])/len(df_dsid)*100,2)


sns.histplot(df_dsid[df_dsid.dryspell_forec>0],bins=np.arange(0,df_dsid.days_late.max()+2),x="days_late",stat="count",kde=True)


df_dates


#number of adm2s with dryspell for forecasted and observed over time
#the diagonal lines are caused by the last day of a rainy season having a forecasted dry spell
from matplotlib.ticker import StrMethodFormatter
fig,ax=plt.subplots()
df_dates_group=df_dates.groupby("date",as_index=False).sum()
df_dates_group.sort_values(by="date").plot(x="date",y="dryspell_forec" ,figsize=(16, 8), color='red',legend=True,ax=ax)
df_dates_group.sort_values(by="date").plot(x="date",y="dryspell_obs" ,figsize=(16, 8), color='#86bf91',legend=True,ax=ax)

# Set x-axis label
ax.set_xlabel("Date", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Number of ADM2s with dry spell", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Observed and forecasted dryspells")





# dates = pd.DataFrame(pd.date_range(start=df_cg_ds.min().StartDate, 
#                      end=df_cg_ds.max().EndDate), columns=['Date'])
# pd.merge(left=dates, right=df, left_on='Date', right_on='StartDate', 
#          how='outer').fillna(method='ffill')


# melt = df_ds[["ADM2_EN","dry_spell_first_date","dry_spell_last_date"]].melt(id_vars=['ADM2_EN'], value_name='date').drop('variable', axis=1)
# melt['date'] = pd.to_datetime(melt['date'])

# melt = melt.groupby('ADM2_EN').apply(lambda x: x.set_index('date').resample('d').first())\
#            .ffill()\
#            .reset_index(level=1)\
#            .reset_index(drop=True)


# list_overlap=[]
# for a in df_ds.ADM2_EN.unique():
# #     dates_ds_adm2=pd.index([])
#     dates_adm2=pd.Index([])
#     df_adm2=df_cg_ds[df_cg_ds.ADM2_EN==a]
#     for i in df_adm2.date.unique():
#         df_adm2_date=df_adm2[df_adm2.date==i]
# #         print(df_adm2_date.dry_spell_first_date.values[0])
# #         print(df_adm2_date.dry_spell_last_date.values[0])
# #         forec_range=pd.period_range(df_adm2_date.dry_spell_first_date.values[0]-timedelta(days=13),df_adm2_date.dry_spell_last_date.values[0]-timedelta(days=1))
#         forec_range=pd.period_range(df_adm2_date.date.values[0],df_adm2_date.date_forec_end.values[0],freq="D")
#         dates_adm2=dates_adm2.union(forec_range)
# #     print(dates_adm2)
# #     print(df_chirpsgefs.date)
#     df_chirpsgefs["ds_range_forec"]
#     list_overlap.append(df_chirpsgefs[(df_chirpsgefs.ADM2_EN==a)&(df_chirpsgefs.date.isin(dates_adm2))])
# df_chirpsgefs_overlap=pd.concat(list_overlap)
# #         ds_range=pd.period_range(i["dry_spell_first_dat"])


# list_overlap=[]
# for a in df_ds.ADM2_EN.unique():
# #     dates_ds_adm2=pd.index([])
#     dates_adm2=pd.Index([])
#     df_adm2=df_ds[df_ds.ADM2_EN==a]
#     for i in df_adm2.dry_spell_first_date.unique():
#         df_adm2_date=df_adm2[df_adm2.dry_spell_first_date==i]
#         print(df_adm2_date.dry_spell_first_date.values[0])
#         print(df_adm2_date.dry_spell_last_date.values[0])
# #         forec_range=pd.period_range(df_adm2_date.dry_spell_first_date.values[0]-timedelta(days=13),df_adm2_date.dry_spell_last_date.values[0]-timedelta(days=1))
#         forec_range=pd.period_range(df_adm2_date.dry_spell_first_date.values[0],df_adm2_date.dry_spell_last_date.values[0],freq="D")
#         dates_adm2=dates_adm2.union(forec_range)
#     print(dates_adm2)
#     print(df_chirpsgefs.date)
#     list_overlap.append(df_chirpsgefs[(df_chirpsgefs.ADM2_EN==a)&(df_chirpsgefs.date.isin(dates_adm2))])
# df_chirpsgefs_overlap=pd.concat(list_overlap)
# #         ds_range=pd.period_range(i["dry_spell_first_dat"])


# #remove the adm2-date entries outside the rainy season for that specific adm2
# #df_belowavg_seas only includes data from 2000, so the 1999 entries are not included
# list_hist_rain_adm2=[]
# for a in df_rain_filled.ADM2_EN.unique():
#     dates_adm2=pd.Index([])
#     for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
#         df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
#         seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
#         dates_adm2=dates_adm2.union(seas_range)
#     list_hist_rain_adm2.append(df_belowavg_seas[(df_belowavg_seas.ADM2_EN==a)&(df_belowavg_seas.date_month.isin(dates_adm2))])
# df_belowavg_seas_rain=pd.concat(list_hist_rain_adm2)


# ### Visualizing rasters 5 days

#load rasters
ds_list=[]
for d in df_ds.dry_spell_first_date.unique():
    d_str=pd.to_datetime(d).strftime("%Y%m%d")
    filename=f"chirpsgefs_5day_africa_{d_str}.tif"
    try:
        rds=rioxarray.open_rasterio(os.path.join(chirpsgefs_dir,filename))
        rds=rds.assign_coords({"time":pd.to_datetime(d)})
        rds=rds.sel(band=1)
        ds_list.append(rds)
    except:
        print(d_str)


ds_drys=xr.concat(ds_list,dim="time")


ds_drys=ds_drys.sortby("time")


df_comb_ds=df_comb[df_comb.dryspell_obs==1]
for a in df_comb_ds.ADM2_EN.unique():
    print(a)
    df_bound_sel_adm=df_bound_adm2[df_bound_adm2.ADM2_EN==a]
    ds_drys_clip_adm = ds_drys.rio.clip(df_bound_sel_adm.geometry.apply(mapping), df_bound_sel_adm.crs, all_touched=True)
    ds_drys_clip_adm_dates=ds_drys_clip_adm.sel(time=ds_drys_clip_adm.time.isin(df_comb_ds[df_comb_ds.ADM2_EN==a].date.unique()))
    #cannot make the facetgrid if only one occurence. For now leave them out since just exploration, but for completeness should somehow include them
    if len(ds_drys_clip_adm_dates.time)>1:
        g=ds_drys_clip_adm_dates.plot(
        col="time",
        col_wrap=6,
        levels=bins,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "label":"Precipitation (mm)",
            "ticks": bins
        },
        cmap="YlOrRd",
    )

        # df_bound = gpd.read_file(adm1_bound_path)
        for ax in g.axes.flat:
            df_bound_sel_adm.boundary.plot(linewidth=1, ax=ax, color="red")
            ax.axis("off")
        g.fig.suptitle(f"{a} {df_comb_ds[df_comb_ds.ADM2_EN==a].sort_values(by='date').dryspell_forec.values}")


# # ### Old experiments

# #inspect difference min and max cell touched
# #plot data of Balaka for 2011
# fig,ax=plt.subplots()
# df_comb[(df_comb.date.dt.year==2011)&(df_comb.ADM2_EN=="Balaka")].sort_values(by="date").plot(x="date",y="max_cell_touched" ,figsize=(16, 8), color='red',legend=True,ax=ax)
# df_comb[(df_comb.date.dt.year==2011)&(df_comb.ADM2_EN=="Balaka")].sort_values(by="date").plot(x="date",y="min_cell_touched" ,figsize=(16, 8), color='green',legend=True,ax=ax)

# # Set x-axis label
# ax.set_xlabel("Start date", labelpad=20, weight='bold', size=12)

# # Set y-axis label
# ax.set_ylabel("mm of rain", labelpad=20, weight='bold', size=12)

# # Despine
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

# plt.title(f"Forecasted rainfall Balaka 2011")


# # ##### Test how many days should be included

# # #path to data start and end rainy season
# # df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020.csv"))
# # df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
# # df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])


# # #set the onset and cessation date for the seasons with them missing (meaning there was no dry spell data from start/till end of the season)
# # df_rain_filled=df_rain.copy()
# # df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
# # df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))


# # #remove the adm2-date entries outside the rainy season for that specific adm2
# # #before we included all forecasts within the min start of the rainy season and max end across the whole country
# # total_days=0
# # list_hist_rain_adm2=[]
# # for a in df_rain.ADM2_EN.unique():
# #     dates_adm2=pd.Index([])
# #     for i in df_rain_filled.season_approx.unique():
# #         seas_range=pd.date_range(df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].onset_date.values[0],df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)].cessation_date.values[0])
# #         dates_adm2=dates_adm2.union(seas_range)
# #         total_days+=len(dates_adm2)
# # #     list_hist_rain_adm2.append(df_hist_all[(df_hist_all.ADM2_EN==a)&(df_hist_all.date.isin(dates_adm2))])
# # # df_hist_rain_adm2=pd.concat(list_hist_rain_adm2)


# # total_days/32





