---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: antact
    language: python
    name: antact
---

<!-- #region -->
### Observed dryspells and correlation with forecasted dry spells in Malawi
**Note: this analysis was done at the beginning of the dry spell pilot project. Along the way some definitions and ideas changed, which might have to be updated in this notebook as well if revisited later on**    

This notebook explores the correlation of observed dry spells and the 15-day forecast provided by CHIRPS-GEFS. The goal of this analysis is to understand the performance of CHIRPS-GEFS for predicting dry spells in order to judge whether it is a suitable indicator to base anticipatory action for dry spells in Malawi on. 


The observed dry spells are computed in the R script `malawi/scripts/mwi_chirps_dry_spell_detection.R`. That script uses different methodologies to define a dry spell. This notebook assumes one is chosen, which is indicated by the filename of `dry_spells_list_path`
The CHIRPS-GEFS data is downloaded and processed in `src/indicators/drought/chirpsgefs_rainfallforecast.py` 

The format CHIRPS-GEFS is produced is the 15 cumulative sum per raster cell. Several statistics per admin region are computed. 

We focus on the mean value per admin area. In this notebook several thresholds for this mean value are tested, as 1) sometimes forecasts have the tendency to overestimate precipitation and 2) the forecast predicts for 15 days while a dry spell only covers 14 days. .

We are mainly focussing on the overlap between observed dry spell and forecasted dry spell. From the analysis you can see that this performance is already pretty bad. We shortly also explore a more strict definition where the start date of the observed dry spell has to be forecasted, but as could be expected this performance is even worse. 

Data limitations
- No CHIRPS-GEFS data is available from 01-01-2020 till 05-10-2020
<!-- #endregion -->

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math 
from datetime import timedelta
import seaborn as sns
import calendar
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import glob
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.plotting import plot_spatial_columns
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
data_public_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(data_public_dir,config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(data_public_dir,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(data_public_dir,"exploration",country_iso3)
dry_spells_processed_dir=os.path.join(country_data_processed_dir,"dry_spells")
chirpsgefs_processed_dir=os.path.join(country_data_processed_dir,"chirpsgefs")

#we have different methodologies of computing dryspells and rainy season
#this notebook chooses one, which is indicated by the files being used
chirpsgefs_stats_path=os.path.join(country_data_processed_dir,"chirpsgefs","mwi_chirpsgefs_rainyseas_stats_mean_back.csv")
rainy_season_path = os.path.join(dry_spells_processed_dir, "rainy_seasons_detail_2000_2020_mean_back.csv")
chirps_rolling_sum_path=os.path.join(dry_spells_processed_dir,"data_mean_values_long.csv")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```python
ds_meth="mean_2mm"
# ds_meth="consecutive_days_2mm"
# ds_meth="consecutive_days_4mm"
```

```python
adm_col="ADM2_EN"
```

```python
#we have different methodologies of computing dryspells and rainy season
#this notebook chooses one, which is indicated by the files being used
if ds_meth=="mean_2mm":
    dry_spells_list_path=os.path.join(dry_spells_processed_dir,f"dry_spells_during_rainy_season_list_2000_2020_mean_back.csv")
elif ds_meth=="consecutive_days_2mm":
    dry_spells_list_path=os.path.join(dry_spells_processed_dir,f"daily_mean_dry_spells_details_2mm_2000_2020.csv")
elif ds_meth=="consecutive_days_4mm":
    dry_spells_list_path=os.path.join(dry_spells_processed_dir,f"daily_mean_dry_spells_details_2000_2020.csv")
```

#### Load CHIRPS-GEFS data

```python
all_files=glob.glob(os.path.join(chirpsgefs_processed_dir, "mwi_chirpsgefs_stats_adm2_15days_*.csv"))
```

```python tags=[]
#combine files of all dates into one
#takes a minute
df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files)
df_chirpsgefs_all = pd.concat(df_from_each_file, ignore_index=True)
```

```python
df_chirpsgefs_all
```

```python tags=[]
#filter out dates that are outside the rainy season for each admin
onset_col="onset_date"
cessation_col="cessation_date"
rainy_adm_col="ADM2_EN"

df_rain = pd.read_csv(rainy_season_path, parse_dates=[onset_col, cessation_col])
 #remove entries where there is no onset and no cessation date, i.e. no rainy season
df_rain=df_rain[(df_rain.onset_date.notnull())|(df_rain.cessation_date.notnull())]
#if onset date or cessation date is missing, set it to Nov 1/Jul1 to make sure all data of that year is downloaded. This happens if e.g. the rainy season hasn't ended yet
df_rain.loc[df_rain.onset_date.isnull()]=df_rain[df_rain[onset_col].isnull()].assign(onset_date=lambda x: pd.to_datetime(f"{x.season_approx.values[0]}-11-01"))
df_rain.loc[df_rain.cessation_date.isnull()]=df_rain[df_rain[cessation_col].isnull()].assign(cessation_date=lambda x: pd.to_datetime(f"{x.season_approx.values[0]+1}-07-01"))

df_rain_adm = df_rain.groupby([rainy_adm_col, "season_approx"], as_index=False).agg({onset_col: np.min, cessation_col: np.max})
    
list_hist_rain_adm = []
for a in df_chirpsgefs_all[adm_col].unique():
    dates_adm = pd.Index([])
    df_rain_seladm = df_rain_adm[df_rain_adm[rainy_adm_col] == a]
    for i in df_rain_seladm.season_approx.unique():
        df_rain_seladm_seas = df_rain_seladm[df_rain_seladm.season_approx == i]
        seas_range = pd.date_range(df_rain_seladm_seas.onset_date.values[0],df_rain_seladm_seas.cessation_date.values[0])
        dates_adm = dates_adm.union(seas_range)
    list_hist_rain_adm.append(df_chirpsgefs_all[(df_chirpsgefs_all[adm_col] == a) & (df_chirpsgefs_all.date.isin(dates_adm))])
df_chirpsgefs = pd.concat(list_hist_rain_adm)
```

```python
#when initial analysis was done, 12-03 was the last date, so selecting those to keep the numbers the same
df_chirpsgefs=df_chirpsgefs[df_chirpsgefs.date<="2021-03-12"]
```

```python
len(df_chirpsgefs)
```

```python
df_chirpsgefs.head()
```

```python
#included statistics in the file
cg_stats=["max_cell","mean_cell","min_cell","perc_se2","perc_se10"]
```

```python
#less than num unique dates*adm2s cause most adm2s end their rainy season earlier (or start later) and thus those date-adm2 combinations are not included
len(df_chirpsgefs)
```

```python
len(df_chirpsgefs.date.unique())
```

```python
len(df_chirpsgefs.ADM2_EN.unique())
```

```python
df_chirpsgefs.date.dt.month.unique()
```

```python
#plot the distributions of the different statistics
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
```

### Understand performance CHIRPS-GEFS
Compare values with those of CHIRPS observational data

```python
df_bound_adm2=gpd.read_file(adm2_bound_path)
```

```python
#read historically observed 15 day rolling sum for all dates (so not only those with dry spells), derived from CHIRPS
#this sometimes gives a not permitted error --> move the chirps_rolling_sum_path file out of the folder and back in to get it to work (don't know why)
df_histobs=pd.read_csv(chirps_rolling_sum_path)
df_histobs.date=pd.to_datetime(df_histobs.date)

#add start of the rolling sum 
df_histobs["date_start"]=df_histobs.date-timedelta(days=14)

#add adm2 and adm1 name
df_histobs=df_histobs.merge(df_bound_adm2[["ADM1_EN","ADM2_EN","ADM2_PCODE"]],left_on="pcode",right_on="ADM2_PCODE")
```

```python
#merge forecast and observed
#only include dates that have a forecast, i.e. merge on right
#date in df_chirpsgefs is the first date of the forecast
df_histformerg=df_histobs.merge(df_chirpsgefs,how="right",left_on=["date_start","ADM2_EN"],right_on=["date","ADM2_EN"],suffixes=("obs","forec"))
```

```python
df_histformerg["diff_forecobs"]=df_histformerg["mean_cell"]-df_histformerg["rollsum_15d"]
```

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_histformerg,y="diff_forecobs",x="rollsum_15d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_histformerg.rollsum_15d.max()+20,10)
group = df_histformerg.groupby(pd.cut(df_histformerg.rollsum_15d, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python
#plot the observed vs forecast-observed for obs<=2mm
df_sel=df_histformerg[df_histformerg.rollsum_15d<=2].sort_values("rollsum_15d")
g=sns.jointplot(data=df_sel,y="diff_forecobs",x="rollsum_15d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_sel.rollsum_15d.max()+2,0.2)
group = df_sel.groupby(pd.cut(df_sel.rollsum_15d, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_density.png"))
```

```python
#plot the observed vs forecast-observed for obs<=30mm
df_sel=df_histformerg[df_histformerg.rollsum_15d<=30].sort_values("rollsum_15d")
g=sns.jointplot(data=df_sel,y="diff_forecobs",x="rollsum_15d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_sel.rollsum_15d.max()+2,1)
group = df_sel.groupby(pd.cut(df_sel.rollsum_15d, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_density.png"))
```

```python
# #plot values per month
# # is ugly, lots of space, so comment out
# #however, couldnt find a better way cause jointplot is a facetgrid itself so becomes messy

# df_histformerg=df_histformerg.sort_values(by="month")
# g = sns.FacetGrid(df_histformerg, col="month")#, size=len(df_histformerg.month.unique()))
# g.map(sns.jointplot, "rollsum_15d","diff_forecobs",kind="hex",height=10,joint_kws={ 'bins':'log'})#, extent=[0, 50, 0, 10])
```

```python
# #plot values per adm1
# g = sns.FacetGrid(df_histformerg, col="ADM1_PCODE")
# g.map(sns.jointplot, "rollsum_15d","diff_forecobs",kind="hex",height=10,joint_kws={ 'bins':'log'})
```

```python
len(df_histformerg[df_histformerg.rollsum_15d<=2])
```

```python
#decently close number of occurences of below 2mm of obs and forecasted, but apparently not at the same times..
len(df_histformerg[df_histformerg.mean_cell<=2])
```

#### Load observed dry spells

```python
df_ds=pd.read_csv(dry_spells_list_path)
df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
df_ds["year"]=df_ds.dry_spell_first_date.dt.year
```

```python
df_ds.head()
```

```python
#number of historically observed dry spells
len(df_ds)
```

```python
len(df_ds.dry_spell_first_date.unique())
```

```python
#chirpsgefs 2020 data is not complete, so these might be removed
len(df_ds[df_ds.dry_spell_first_date.dt.year==2020])
```

```python
sns.histplot(df_ds,x="dry_spell_duration")
```

```python
#get a feeling for where the dry spells occured
df_ds_adm2=df_ds.groupby(["ADM2_EN"],as_index=False).count()
df_bound_adm2=gpd.read_file(adm2_bound_path)
gdf_ds_adm2=df_bound_adm2.merge(df_ds_adm2,how="left")
fig=plot_spatial_columns(gdf_ds_adm2, ["pcode"], title="Number of dry spells", predef_bins=None,cmap='YlOrRd',colp_num=1)
```

```python
#check correlation with size of area and number of dry spells
g=sns.regplot(data=gdf_ds_adm2,y="pcode",x="Shape_Area",scatter_kws = {'alpha' : 1/3},fit_reg=False)
ax=g.axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("Number of dry spells")
```

### Overlap dry spells observed and forecasted
Solely looking if any part of an observed dry spell overlaps any part of a forecasted dry spell, and the other way around. Not if their start date overlaps
i.e. this is the most loosely defined way to look at overlap

```python
#create df with all dates that were part of a dry spell per adm2
#assign an ID to each dry spell, such that we can group by that later on
df_ds["ID_obs"]=range(1,len(df_ds)+1)
#important to reset the index, since that is what is being joined on
df_ds_res=df_ds.reset_index(drop=True)
#create datetimeindex per row
a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_daterange=df_ds_res[["ADM2_EN","ID_obs"]].join(pd.DataFrame(a)).set_index(["ADM2_EN","ID_obs"]).stack().droplevel(-1).reset_index()
df_ds_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_ds_daterange["dryspell_obs"]=1

```python
df_ds_daterange.head()
```

```python
#total number of dates that were part of a dry spell
len(df_ds_daterange)
```

```python
df_ds_daterange=df_ds_daterange.drop_duplicates()
```

```python
#there shouldn't be any duplicates so check if two lengths match
len(df_ds_daterange)
```

```python
#create dataframe with all dates for which we got chirpsgefs data
df_dates=df_chirpsgefs[["date","ADM2_EN"]].sort_values(["ADM2_EN","date"])
```

```python
#merge the observed dry spells
#merge on left, such to only include dates for which a forecast is available
df_dates=df_dates.merge(df_ds_daterange,how="left",on=["date","ADM2_EN"])
df_dates.dryspell_obs=df_dates.dryspell_obs.replace(np.nan,0)
```

```python
#Quite some obs dry spell dates go missing
#This has two explanations: 
#1) chirps-gefs has missing data in 2020 (87 entries)
#2) a dry spell started in 2005 in Salima before 15-03 and then continued into the dry season. After 15-03 is excluded from chirps-gefs but included in obs ds (21 entries)
print(f"number of dates observed dry spells after merge: {len(df_dates[df_dates.dryspell_obs==1])}")
print(f"original number of dates with observed dry spells: {len(df_ds_daterange)}")
print(f"difference: {len(df_ds_daterange)-len(df_dates[df_dates.dryspell_obs==1])}")
```

```python
len(df_ds.dry_spell_first_date.unique())
```

```python
#this is the forecast of the next 15 days for each day that was part of a dry spell, so not perse super helpful
#e.g. last date of a dry spell is expected to not have a super low forecast
df_chirpsgefs_ds=df_chirpsgefs.merge(df_ds_daterange,how="left",on=["date","ADM2_EN"])
df_chirpsgefs_ds.dryspell_obs=df_chirpsgefs_ds.dryspell_obs.replace(np.nan,0)
num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_chirpsgefs_ds,x=s,ax=ax,stat="density",common_norm=False,kde=True,hue="dryspell_obs")
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
```

```python
num_plots = len(cg_stats)
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,20))
for i, s in enumerate(cg_stats):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.histplot(df_chirpsgefs_ds,x=s,ax=ax,stat="count",kde=True,hue="dryspell_obs")
    ax.set_title(s)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
```

Compute dry spell overlap    
Step by step for 2 mm threshold, and thereafter code repeated in function to compare different thresholds

```python
#select all dates where chirps-gefs forecasts dry spell
df_cg_ds=df_chirpsgefs[df_chirpsgefs.mean_cell<=2]
```

```python
#create list of dates that were within a forecast that predicted a dry spell
df_cg_ds=df_cg_ds.sort_values(["ADM2_EN","date"]).reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_cg_ds[['date', 'date_forec_end']].values]
df_cg_daterange=df_cg_ds[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_cg_daterange.rename(columns={0:"date"},inplace=True)
df_cg_daterange["dryspell_forec"]=1
```

```python
df_cg_daterange.head()
```

```python
len(df_cg_daterange)
```

```python
#with the forecast it is expected to be a lot of overlap
df_cg_daterange=df_cg_daterange.drop_duplicates()
```

```python
#still many more dates forecasted to be part of a dry spell than the observed..
len(df_cg_daterange)
```

```python
len(df_ds_daterange)
```

```python
#assign ID to each forecasted dry spell
#we assign one ID per any overlapping range
#e.g. if dry spell forecasted on 01-01 and 03-01 (but not 02-01), we would see this as one dry spell from 01-01 till 17-01
df_cg_daterange["ID_forec"]=df_cg_daterange.sort_values(["ADM2_EN","date"]).groupby("ADM2_EN").date.diff().dt.days.ne(1).cumsum()
```

```python
#merge the forecasted dry spells
df_dates_comb=df_dates.merge(df_cg_daterange,how="left",on=["date","ADM2_EN"])
df_dates_comb["dryspell_forec"]=df_dates_comb["dryspell_forec"].replace(np.nan,0)
```

```python
#Quite some forecasted dry spell dates go missing
#This is due to the start of the forecast period still being inside the rainy season, while the later dates are not
print(f"number of dates forecasted dry spells after merge: {len(df_dates_comb[df_dates_comb.dryspell_forec==1])}")
print(f"original number of dates with forecasted dry spells: {len(df_cg_daterange)}")
print(f"difference: {len(df_cg_daterange)-len(df_dates_comb[df_dates_comb.dryspell_forec==1])}")
```

```python
#number of dates with observed dry spell overlapping with forecasted
y_target =    df_dates_comb["dryspell_obs"]
y_predicted = df_dates_comb["dryspell_forec"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Day within observed dry spell")
ax.set_xlabel("Day within forecasted dry spell")
plt.show()
```

```python
df_dates_comb["month"]=df_dates_comb.date.dt.month
fig, axes = plt.subplots(1, 2, sharex=True,sharey=True, figsize=(12,6))
sns.histplot(df_dates_comb[df_dates_comb.dryspell_obs==1],ax=axes[0],x="month",bins=range(df_dates_comb.month.min(),df_dates_comb.month.max()+2))
axes[0].set_title("Observed dry spells")
sns.histplot(df_dates_comb[df_dates_comb.dryspell_forec==1],ax=axes[1],x="month",bins=range(df_dates_comb.month.min(),df_dates_comb.month.max()+1))
axes[1].set_title("Forecasted dry spells")
```

```python
#number of dates with observed dry spell overlapping with forecasted per month
num_plots = len(df_dates_comb.month.unique())
colp_num=4
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(30,15))
for i, m in enumerate(df_dates_comb.sort_values(by="month").month.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    y_target =    df_dates_comb.loc[df_dates_comb.month==m,"dryspell_obs"]
    y_predicted = df_dates_comb.loc[df_dates_comb.month==m,"dryspell_forec"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)

    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
    ax.set_ylabel("Day within observed dry spell")
    ax.set_xlabel("Day within forecasted dry spell")
    ax.set_title(f"contigency matrix for {calendar.month_name[m]}")
```

```python
#number of dates with observed dry spell overlapping with forecasted per month
num_plots = len(df_dates_comb.ADM2_EN.unique())
colp_num=4
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,50))
for i, m in enumerate(df_dates_comb.sort_values(by="ADM2_EN").ADM2_EN.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    y_target =    df_dates_comb.loc[df_dates_comb.ADM2_EN==m,"dryspell_obs"]
    y_predicted = df_dates_comb.loc[df_dates_comb.ADM2_EN==m,"dryspell_forec"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)

    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
    ax.set_ylabel("Day within observed dry spell")
    ax.set_xlabel("Day within forecasted dry spell")
    ax.set_title(f"contigency matrix for {m}")
```

```python
#create dataframes with start and end dates of the observed and forecasted dry spells, per ID
df_forec_dates=df_dates_comb.groupby("ID_forec",as_index=False).agg(start_date=("date","min"),end_date=("date","max"))
df_obs_dates=df_dates_comb.groupby("ID_obs",as_index=False).agg(start_date=("date","min"),end_date=("date","max"))
```

```python
df_forec_dates.head()
```

```python
#create dataframe with all the observed dry spells and the overlapping forecasts (if any)
df_dsobsgr=df_dates_comb.groupby("ID_obs").sum()
#becomes meaningless when summing
df_dsobsgr=df_dsobsgr.drop("ID_forec",axis=1)
df_dsobsgr["ID_forec"]=df_dates_comb.groupby("ID_obs").ID_forec.apply(lambda x: x.min(skipna=True))
```

```python
df_dsobsgr.head()
```

```python
df_obsid=df_dsobsgr.reset_index().merge(df_forec_dates[["ID_forec","start_date"]],how="left",on="ID_forec").rename(columns={"start_date":"start_date_forec"})
df_obsid=df_obsid.merge(df_obs_dates[["ID_obs","start_date"]],how="left",on="ID_obs").rename(columns={"start_date":"start_date_obs"})
#how many days late the forecast started forecasting the dry spell (negative if it anticipated)
df_obsid["days_late"]=(df_obsid.start_date_forec-df_obsid.start_date_obs).dt.days
```

```python
df_obsid.sort_values("start_date_obs").head()
```

```python
print(f"The recall (TP/(TP+FN)) is: {round(len(df_obsid[df_obsid.dryspell_forec>0])/len(df_obsid),4)}")
print(f"{len(df_obsid)} dry spells were observed of which {len(df_obsid[~df_obsid.ID_forec.isnull()])} overlapped with forecasted dry spells")
```

```python
#repeat the same excercise, but now to see how many forecasted dryspells overlap observed dry spells
#--> can compute precision from that
df_dsforgr=df_dates_comb.groupby("ID_forec").sum()
df_dsforgr=df_dsforgr.drop("ID_obs",axis=1)
df_dsforgr["ID_obs"]=df_dates_comb.groupby("ID_forec").ID_obs.apply(lambda x: x.min(skipna=True))
df_forid=df_dsforgr.reset_index().merge(df_obs_dates[["ID_obs","start_date"]],how="left",on="ID_obs").rename(columns={"start_date":"start_date_obs"})
df_forid=df_forid.merge(df_forec_dates[["ID_forec","start_date"]],how="left",on="ID_forec").rename(columns={"start_date":"start_date_forec"})
df_forid["days_late"]=(df_forid.start_date_obs-df_forid.start_date_forec).dt.days
```

```python
#the overlap of observed dry spells for precision might not be the same as recall
#this can be caused if several forecasted dry spells fall within one observed dry spell
#however the difference shouldn't be large
print(f"The precision (TP/(TP+FP)) is: {round(len(df_forid[df_forid.dryspell_obs>0])/len(df_forid),4)}")
print(f"{len(df_forid)} dry spells were forecasted of which {len(df_forid[~df_forid.ID_obs.isnull()])} overlapped with observed dry spells")
```

```python
#ID_obs indicates the number of forecasted dry spells that matched observations per month
df_forid["start_month_forec"]=df_forid.start_date_forec.dt.month
df_forid.groupby("start_month_forec").count()
```

```python
#compile all above code into function such that can test different thresholds
def stats_threshold(df_dates,df_chirpsgefs,threshold):
    df_cg_ds=df_chirpsgefs[df_chirpsgefs.mean_cell<=threshold]
    df_cg_ds=df_cg_ds.sort_values(["ADM2_EN","date"]).reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_cg_ds[['date', 'date_forec_end']].values]
    df_cg_daterange=df_cg_ds[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
    df_cg_daterange.rename(columns={0:"date"},inplace=True)
    df_cg_daterange["dryspell_forec"]=1
    df_cg_daterange=df_cg_daterange.drop_duplicates()
    df_cg_daterange["ID_forec"]=df_cg_daterange.sort_values(["ADM2_EN","date"]).groupby("ADM2_EN").date.diff().dt.days.ne(1).cumsum()
    df_dates_comb=df_dates.merge(df_cg_daterange,how="left",on=["date","ADM2_EN"])
    df_dates_comb["dryspell_forec"]=df_dates_comb["dryspell_forec"].replace(np.nan,0)
    df_forec_dates=df_dates_comb.groupby("ID_forec",as_index=False).agg(start_date=("date","min"),end_date=("date","max"))
    df_obs_dates=df_dates_comb.groupby("ID_obs",as_index=False).agg(start_date=("date","min"),end_date=("date","max"))
    df_dsobsgr=df_dates_comb.groupby("ID_obs").sum()
    df_dsobsgr=df_dsobsgr.drop("ID_forec",axis=1)
    df_dsobsgr["ID_forec"]=df_dates_comb.groupby("ID_obs").ID_forec.apply(lambda x: x.min(skipna=True))
    df_obsid=df_dsobsgr.reset_index().merge(df_forec_dates[["ID_forec","start_date"]],how="left",on="ID_forec").rename(columns={"start_date":"start_date_forec"})
    df_obsid=df_obsid.merge(df_obs_dates[["ID_obs","start_date"]],how="left",on="ID_obs").rename(columns={"start_date":"start_date_obs"})
    df_obsid["days_late"]=(df_obsid.start_date_forec-df_obsid.start_date_obs).dt.days
    df_dsforgr=df_dates_comb.groupby("ID_forec").sum()
    df_dsforgr=df_dsforgr.drop("ID_obs",axis=1)
    df_dsforgr["ID_obs"]=df_dates_comb.groupby("ID_forec").ID_obs.apply(lambda x: x.min(skipna=True))
    df_forid=df_dsforgr.reset_index().merge(df_obs_dates[["ID_obs","start_date"]],how="left",on="ID_obs").rename(columns={"start_date":"start_date_obs"})
    df_forid=df_forid.merge(df_forec_dates[["ID_forec","start_date"]],how="left",on="ID_forec").rename(columns={"start_date":"start_date_forec"})
    df_dates_tn=df_dates_comb[(df_dates_comb.dryspell_obs==0)&(df_dates_comb.dryspell_forec==0)]
    pd.options.mode.chained_assignment = None  # default='warn'
    df_dates_tn.loc[:,"ID_tn"]=df_dates_tn.sort_values(["ADM2_EN","date"]).groupby("ADM2_EN").date.diff().dt.days.ne(1).cumsum()
    tn=len(df_dates_tn.ID_tn.unique())

    recall=len(df_obsid[df_obsid.dryspell_forec>0])/len(df_obsid)
    num_obs=len(df_obsid)
    num_obsfor=len(df_obsid[~df_obsid.ID_forec.isnull()])
    precision=len(df_forid[df_forid.dryspell_obs>0])/len(df_forid)
    num_for=len(df_forid)
    num_forobs=len(df_forid[~df_forid.ID_obs.isnull()])
    return recall,precision,num_obs,num_obsfor,num_for,num_forobs,tn
```

```python
threshold_list=[2,5,10,15,20,25]
df_pr=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for threshold in threshold_list:
    recall,precision,num_obs,num_obsfor,num_for,num_forobs,tn=stats_threshold(df_dates,df_chirpsgefs,threshold)
#     df_pr.loc[threshold,["recall","precision","num_obs","num_obsfor","num_for","num_forobs","tn"]]=recall,precision,num_obs,num_obsfor,num_for,num_forobs,tn
    print(f"The recall (TP/(TP+FN)) with {threshold}mm threshold is: {round(recall,4)} ({num_obsfor}/{num_obs})")
    print(f"The precision (TP/(TP+FP))  with {threshold}mm threshold is: {round(precision,4)} ({num_forobs}/{num_for})")
    print(f"The miss rate (FP/(TP+FP))  with {threshold}mm threshold is: {round((num_for-num_forobs)/(num_for),4)} ({num_for-num_forobs}/{num_for})\n")
    cm=np.array([[tn, num_for-num_obsfor],[num_obs-num_obsfor, num_obsfor]])
    fig_cm,ax=plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,class_names=["No","Yes"])
    plt.xlabel("Forecasted dry spell")
    plt.ylabel("Observed dry spell")
    plt.title(f"{threshold}mm threshold")
    plt.show()
    # df_pr.to_csv(os.path.join(country_data_processed_dir,"dry_spells","chirpsgefs",f"chirpsgefs_{ds_meth}_precision_recall_thresholds.csv"))
    fig_cm.tight_layout()
    # fig_cm.savefig(os.path.join(country_data_processed_dir,"plots","dry_spells","chirpsgefs",f"{country_iso3}_cm_chirpsgefs_{ds_meth}_thresh_{threshold}mm.png"))
```

### Experiment extent

```python
def stats_threshold_extent(df_dates,df_chirpsgefs,threshold):
    df_cg_ds=df_chirpsgefs[df_chirpsgefs.mean_cell<=threshold]
    df_cg_ds=df_cg_ds.sort_values(["ADM2_EN","date"]).reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_cg_ds[['date', 'date_forec_end']].values]
    df_cg_daterange=df_cg_ds[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
    df_cg_daterange.rename(columns={0:"date"},inplace=True)
    df_cg_daterange["dryspell_forec"]=1
    df_cg_daterange=df_cg_daterange.drop_duplicates()
    df_cg_daterange["ID_forec"]=df_cg_daterange.sort_values(["ADM2_EN","date"]).groupby("ADM2_EN").date.diff().dt.days.ne(1).cumsum()
    df_dates_comb=df_dates.merge(df_cg_daterange,how="left",on=["date","ADM2_EN"])
    df_dates_comb["dryspell_forec"]=df_dates_comb["dryspell_forec"].replace(np.nan,0)
    df_forec_dates=df_dates_comb.groupby("ID_forec",as_index=False).agg(start_date=("date","min"),end_date=("date","max"))
    df_obs_dates=df_dates_comb.groupby("ID_obs",as_index=False).agg(start_date=("date","min"),end_date=("date","max"))
    df_dsobsgr=df_dates_comb.groupby("ID_obs").sum()
    df_dsobsgr=df_dsobsgr.drop("ID_forec",axis=1)
    df_dsobsgr["ID_forec"]=df_dates_comb.groupby("ID_obs").ID_forec.apply(lambda x: x.min(skipna=True))
    df_obsid=df_dsobsgr.reset_index().merge(df_forec_dates[["ID_forec","start_date"]],how="left",on="ID_forec").rename(columns={"start_date":"start_date_forec"})
    df_obsid=df_obsid.merge(df_obs_dates[["ID_obs","start_date"]],how="left",on="ID_obs").rename(columns={"start_date":"start_date_obs"})
    df_obsid["days_late"]=(df_obsid.start_date_forec-df_obsid.start_date_obs).dt.days
    df_dsforgr=df_dates_comb.groupby("ID_forec").sum()
    df_dsforgr=df_dsforgr.drop("ID_obs",axis=1)
    df_dsforgr["ID_obs"]=df_dates_comb.groupby("ID_forec").ID_obs.apply(lambda x: x.min(skipna=True))
    df_forid=df_dsforgr.reset_index().merge(df_obs_dates[["ID_obs","start_date"]],how="left",on="ID_obs").rename(columns={"start_date":"start_date_obs"})
    df_forid=df_forid.merge(df_forec_dates[["ID_forec","start_date"]],how="left",on="ID_forec").rename(columns={"start_date":"start_date_forec"})
    df_dates_tn=df_dates_comb[(df_dates_comb.dryspell_obs==0)&(df_dates_comb.dryspell_forec==0)]
    return df_dates_comb,df_obsid,df_forid
```

```python
df_dates_comb,df_obsid,df_forid=stats_threshold_extent(df_dates,df_chirpsgefs,25)
```

```python
df_obsid["date_month"]=df_obsid.start_date_obs.dt.to_period("M")
```

```python
df_obsid["forecasted"]=np.where(df_obsid.ID_forec.isnull(),0,1)
```

```python
df_obsid.groupby("date_month").agg({"forecasted":["sum","count"]})#,"forecasted":"count"})
```

```python
df_forid["observed"]=np.where(df_forid.ID_obs.isnull(),0,1)
```

```python
df_forid["date_month"]=df_forid.start_date_forec.dt.to_period("M")
```

```python
df_forid
```

```python
df_fordm=df_forid.groupby("date_month").agg(count=('ID_forec', 'count'),obs_ds=("observed","sum"))
```

### Transform data for heatmap
We got a R script that creates a real nice heatmap showing the dry spells per adm2 for each year. Since we also use this code to analyze observed dry spells, it is nice to keep the layout the same for comparing observed and forecasted.    
Thus, transform the data such that it can be given as input to the R script
On the GDrive you can find the outputted heatmap from the R script that shows the observed and forecasted dry spells clearly over time

```python
#add pcode
df_chirpsgefs_pcode=df_chirpsgefs.merge(df_bound_adm2[["ADM2_EN","ADM2_PCODE"]],on="ADM2_EN",how="left")
#create dataframe with all dates for which we got chirpsgefs data
df_dates_viz=df_chirpsgefs_pcode[["date","ADM2_EN","ADM2_PCODE"]].sort_values(["ADM2_EN","date"])
df_dates_viz.rename(columns={"ADM2_PCODE":"pcode"},inplace=True)
#merge the observed dry spells
#merge on outer instead of left for visualization
df_dates_viz=df_dates_viz.merge(df_ds_daterange,how="outer",on=["date","ADM2_EN"]) #left
df_dates_viz.dryspell_obs=df_dates_viz.dryspell_obs.replace(np.nan,0)
```

```python
#select all dates where chirps-gefs forecasts dry spell
df_cg_ds=df_chirpsgefs[df_chirpsgefs.mean_cell<=threshold]
#create list of dates that were within a forecast that predicted a dry spell
df_cg_ds=df_cg_ds.sort_values(["ADM2_EN","date"]).reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_cg_ds[['date', 'date_forec_end']].values]
df_cg_daterange=df_cg_ds[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_cg_daterange.rename(columns={0:"date"},inplace=True)
df_cg_daterange["dryspell_forec"]=1
#with the forecast it is expected to be a lot of overlap
df_cg_daterange=df_cg_daterange.drop_duplicates()
```

```python
#merge the forecasted dry spells
df_dates_viz_comb=df_dates_viz.merge(df_cg_daterange,how="outer",on=["date","ADM2_EN"]) #left
df_dates_viz_comb["dryspell_forec"]=df_dates_viz_comb["dryspell_forec"].replace(np.nan,0)
```

```python
def label_ds(row):
    if row["dryspell_obs"]==1 and row["dryspell_forec"]==1:
        return 3
    elif row["dryspell_obs"]==1:
        return 2
    elif row["dryspell_forec"]==1:
        return 1
    else:
        return 0
```

```python
#encode dry spells and whether it was none, only observed, only forecasted, or both
df_dates_viz_comb["dryspell_match"]=df_dates_viz_comb.apply(lambda row:label_ds(row),axis=1)
```

```python
#add dates that are not present in df, i.e. outside rainy season
df_dates_viz_filled=df_dates_viz_comb.sort_values('date').set_index(['date']).groupby('pcode').apply(lambda x: x.reindex(pd.date_range(pd.to_datetime('01-01-2000'), pd.to_datetime('31-12-2020'), name='date'),fill_value=0).drop('pcode',axis=1).reset_index()).reset_index().drop("level_1",axis=1)
```

```python
#cause for now we only wanna show till end of 2020 cause no obs dry spells data after that
#could also choose to only do till 2019 cause no chirps-gefs for 2020, but wanted to keep data the same as obs dry spells
df_dates_viz_filled=df_dates_viz_filled[df_dates_viz_filled.date.dt.year<=2020]
```

```python
# df_dates_viz_filled.drop(["ID_obs"],axis=1).to_csv(os.path.join(country_data_processed_dir,"dry_spells","chirpsgefs",f"dryspells_chirpsgefs_dates_viz_{ds_meth}_th{threshold}.csv"))
```

#### Plot raw raster values
Of CHIRPS-GEFS for dates where an observed dry spell started --> goal to better understand the patterns at those dates and if there would be a better way of aggregation. Conclusion that there doesnt seem to be a consistent pattern :( 

```python
#create df where forecast start date is merged with dry spell start date
df_comb=df_ds.merge(df_chirpsgefs[["ADM2_EN","date","date_forec_end"]+cg_stats],how="right",left_on=["dry_spell_first_date","ADM2_EN"],right_on=["date","ADM2_EN"])
df_comb["dryspell_obs"]=np.where(df_comb.dry_spell_first_date.notna(),1,0)
```

```python
# import rioxarray
# import xarray as xr
# from shapely.geometry import mapping
```

```python
# chirpsgefs_ras_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,config.GLOBAL_ISO3,"chirpsgefs")
# #load the raster data
# ds_list=[]
# for d in df_ds.dry_spell_first_date.unique():
#     d_str=pd.to_datetime(d).strftime("%Y%m%d")
#     filename=f"chirpsgefs_africa_15days_{d_str}.tif"
#     try:
#         rds=rioxarray.open_rasterio(os.path.join(chirpsgefs_ras_dir,filename))
#         rds=rds.assign_coords({"time":pd.to_datetime(d)})
#         rds=rds.sel(band=1)
#         ds_list.append(rds)
#     except:
#         print(f"no data for {d}")

# ds_drys=xr.concat(ds_list,dim="time")

# ds_drys=ds_drys.sortby("time")
```

```python
# #plot the rasters. Plot per adm2
# ds_drys_clip = ds_drys.rio.clip(df_bound_adm2.geometry.apply(mapping), all_touched=True)
# bins=np.arange(0,101,10)

# df_comb_ds=df_comb[df_comb.dryspell_obs==1]
# for a in df_comb_ds.ADM2_EN.unique():
#     df_bound_sel_adm=df_bound_adm2[df_bound_adm2.ADM2_EN==a]
#     ds_drys_clip_adm = ds_drys.rio.clip(df_bound_sel_adm.geometry.apply(mapping), all_touched=True)
#     ds_drys_clip_adm_dates=ds_drys_clip_adm.sel(time=ds_drys_clip_adm.time.isin(df_comb_ds[df_comb_ds.ADM2_EN==a].date.unique()))
#     #cannot make the facetgrid if only one occurence. For now leave them out since just exploration, but for completeness should somehow include them
#     if len(ds_drys_clip_adm_dates.time)>1:
#         g=ds_drys_clip_adm_dates.plot(
#         col="time",
#         col_wrap=6,
#         levels=bins,
#         cbar_kwargs={
#             "orientation": "horizontal",
#             "shrink": 0.8,
#             "aspect": 40,
#             "pad": 0.1,
#             "label":"Precipitation (mm)",
#             "ticks": bins
#         },
#         cmap="YlOrRd",
#     )

#         for ax in g.axes.flat:
#             df_bound_sel_adm.boundary.plot(linewidth=1, ax=ax, color="red")
#             ax.axis("off")
#         df_comb_ds_adm=df_comb_ds.sort_values(by=['ADM2_EN','date'])[df_comb_ds.ADM2_EN==a]
#         g.fig.suptitle(f"{a} {df_comb_ds_adm.mean_cell.values}")
```

### Detecting start of a dry spell
While previously we looked at the overlapping days of observed and forecasted dry spell, we can also have as an objective to predict the start of the dry spell. 

Since this is a more strict definition, the performance can be expected to be even worse, which is indeed the case. It is expected that CHIRPS-GEFS is projecting more dry spells than observed ones due to how the data is constructed. CHIRPS-GEFS might still predict the next 15 days to be a dry spell while the dry spell has already started, and this doesn't have to be wrong. 
. 
However, what is surprising is that on so little dates that a dry spell started (or actually just on one), the forecast also predicted a dry spell..

```python
#compute the contigency table when using mean cell for different thresholds
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
t_list=[2,4,6,8,10,20]
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
```

```python
#set definition of forecasted dry spell as max cell having not more than 2 mm of rains in 15 days period
df_comb["dryspell_forec"]=np.where(df_comb.mean_cell<=2,1,0)
```

#### correlation number of admin2s with observed and forecasted dry spell
Explore if correlation with number of adm2s experiencing a dry spell at a date.  
There is not.. 
This is again only about the start of the dry spell. Could explore the more loose definition of having any overlap between observed and forecasted, but not expecting much more correlation (correct me if I am wrong..)

```python
#compute number of adm2s with dryspell per date
df_numadm=df_comb.groupby("date")[["dryspell_obs","dryspell_forec"]].sum()
```

```python
df_comb_adm1=df_comb.merge(df_bound_adm2[["ADM1_EN","ADM2_EN"]],on="ADM2_EN",how="left")
df_numadm1=df_comb_adm1.groupby(["date","ADM1_EN"],as_index=False)[["dryspell_obs","dryspell_forec"]].sum()

num_plots = len(df_numadm1.ADM1_EN.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,5))


for i,a in enumerate(df_numadm1.ADM1_EN.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    g=sns.regplot(data = df_numadm1[df_numadm1.ADM1_EN==a], x = 'dryspell_obs', y = 'dryspell_forec', fit_reg = False,
            scatter_kws = {'alpha' : 1/3},ax=ax)#,x_jitter = 0.2, y_jitter = 0.2)
    g.axes.set_title(a)
```

### 5 day forecast
Quick test to see if the 5 day instead of 15 day forecast gives better performance

```python
all_files_5day=glob.glob(os.path.join(chirpsgefs_processed_dir, "mwi_chirpsgefs_stats_adm2_5days_*.csv"))
```

```python tags=[]
#combine files of all dates into one
#takes a minute
df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files_5day)
df_chirpsgefs_5day = pd.concat(df_from_each_file, ignore_index=True)
```

```python
df_chirpsgefs_5day
```

```python tags=[]
#filter out dates that are outside the rainy season for each admin
onset_col="onset_date"
cessation_col="cessation_date"
rainy_adm_col="ADM2_EN"

df_rain = pd.read_csv(rainy_season_path, parse_dates=[onset_col, cessation_col])
 #remove entries where there is no onset and no cessation date, i.e. no rainy season
df_rain=df_rain[(df_rain.onset_date.notnull())|(df_rain.cessation_date.notnull())]
#if onset date or cessation date is missing, set it to Nov 1/Jul1 to make sure all data of that year is downloaded. This happens if e.g. the rainy season hasn't ended yet
df_rain.loc[df_rain.onset_date.isnull()]=df_rain[df_rain[onset_col].isnull()].assign(onset_date=lambda x: pd.to_datetime(f"{x.season_approx.values[0]}-11-01"))
df_rain.loc[df_rain.cessation_date.isnull()]=df_rain[df_rain[cessation_col].isnull()].assign(cessation_date=lambda x: pd.to_datetime(f"{x.season_approx.values[0]+1}-07-01"))

df_rain_adm = df_rain.groupby([rainy_adm_col, "season_approx"], as_index=False).agg({onset_col: np.min, cessation_col: np.max})
    
list_hist_rain_adm = []
for a in df_chirpsgefs_5day[adm_col].unique():
    dates_adm = pd.Index([])
    df_rain_seladm = df_rain_adm[df_rain_adm[rainy_adm_col] == a]
    for i in df_rain_seladm.season_approx.unique():
        df_rain_seladm_seas = df_rain_seladm[df_rain_seladm.season_approx == i]
        seas_range = pd.date_range(df_rain_seladm_seas.onset_date.values[0],df_rain_seladm_seas.cessation_date.values[0])
        dates_adm = dates_adm.union(seas_range)
    list_hist_rain_adm.append(df_chirpsgefs_5day[(df_chirpsgefs_5day[adm_col] == a) & (df_chirpsgefs_5day.date.isin(dates_adm))])
df_cg_fd = pd.concat(list_hist_rain_adm)
```

```python
#when initial analysis was done, 12-03 was the last date, so selecting those to keep the numbers the same
df_cg_fd=df_cg_fd[df_cg_fd.date<="2021-03-12"]
```

```python
chirps_rolling_sum_path_5day=os.path.join(dry_spells_processed_dir,"data_mean_values_long_5day.csv")
```

```python
#read historically observed 5 day rolling sum for all dates (so not only those with dry spells), derived from CHIRPS
#this sometimes gives a not permitted error --> move the chirps_rolling_sum_path file out of the folder and back in to get it to work (dont ask me why)
df_histobs=pd.read_csv(chirps_rolling_sum_path_5day)
df_histobs.date=pd.to_datetime(df_histobs.date)

#add start of the rolling sum 
df_histobs["date_start"]=df_histobs.date-timedelta(days=4)

#add adm2 and adm1 name
df_histobs=df_histobs.merge(df_bound_adm2[["ADM1_EN","ADM2_EN","ADM2_PCODE"]],left_on="pcode",right_on="ADM2_PCODE")
```

```python
#merge forecast and observed
#only include dates that have a forecast, i.e. merge on right
#date in df_chirpsgefs is the first date of the forecast
#so the values are the rolling sum for the date+timedelta(t=days_ahead) #where days_aheas is 5 in this case
df_histformerg=df_histobs.merge(df_cg_fd,how="right",left_on=["date_start","ADM2_EN"],right_on=["date","ADM2_EN"],suffixes=("obs","forec"))
```

```python
df_histformerg["diff_forecobs"]=df_histformerg["mean_cell"]-df_histformerg["rollsum_5d"]
```

```python
#date_forec_end is not correct!! didn't adjust correctly in computation script
df_histformerg[["dateforec","dateobs","diff_forecobs","mean_cell","rollsum_5d","rollsum_15d"]]
```

```python tags=[]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_histformerg,y="diff_forecobs",x="rollsum_5d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_histformerg.rollsum_5d.max()+20,10)
group = df_histformerg.groupby(pd.cut(df_histformerg.rollsum_5d, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="mean")
g.set_axis_labels("Observed 5 day sum (mm)", "Forecasted 5 day sum - Observed 5 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
```

```python

```
