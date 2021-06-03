---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.2
kernelspec:
  display_name: antact
  language: python
  name: antact
---

# Observed lower tercile precipitation in Burkina Faso
This notebook explores the occurrence of below average precipitation in Burkina Faso. This is thereafter used to assess the skill of below average forecasts. 
- Trigger #1 in March covering Apr-May-June. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%.
- Targeted Admin1s: Boucle de Mounhoun, Centre Nord, Sahel, Nord.

```{code-cell} ipython3
:tags: [remove_cell]

%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
:tags: [remove_cell]

import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from rasterstats import zonal_stats
from IPython.display import Markdown as md
from myst_nb import glue
from dateutil.relativedelta import relativedelta
import math
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import get_iri_data
```

```{code-cell} ipython3
:tags: [remove_cell]

country="bfa"
country_iso3="bfa"
config=Config()
parameters = config.parameters(country)
data_raw_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR)
data_processed_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR)
country_data_raw_dir = os.path.join(data_raw_dir,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country)
glb_data_raw_dir = os.path.join(data_raw_dir,"glb")
chirps_monthly_raw_dir = os.path.join(glb_data_raw_dir,"chirps","monthly")
chirps_country_processed_dir = os.path.join(data_processed_dir,country,"chirps")

chirps_monthly_path=os.path.join(chirps_monthly_raw_dir,"chirps_glb_monthly.nc")
chirps_country_processed_path = os.path.join(chirps_country_processed_dir,f"{country}_chirps_monthly.nc")
chirps_seasonal_lower_tercile_processed_path = os.path.join(chirps_country_processed_dir,"seasonal",f"{country}_chirps_seasonal_lowertercile.nc")
stats_reg_for_path=os.path.join(country_data_exploration_dir,f"{country}_iri_seasonal_forecast_stats_{''.join(adm_sel)}.csv")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```{code-cell} ipython3
:tags: [remove_cell]

adm_sel=["Boucle du Mouhoun","Nord","Centre-Nord","Sahel"]
```

```{code-cell} ipython3
# gdf_adm1=gpd.read_file(adm1_bound_path)
```

```{code-cell} ipython3
#show distribution of rainfall across months for 2020 to understand rainy season patterns
ds_country=xr.open_dataset(chirps_country_processed_path)
#show the data for each month of 2020, clipped to MWI
g=ds_country.sel(time=ds_country.time.dt.year.isin([2020])).precip.plot(
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
```

```{code-cell} ipython3
#get a feel for the distribution of rainfall during the two seasons of interest
#TODO: make subplots instead of separate plots
seas_len=3
ds_season=ds_country.rolling(time=seas_len,min_periods=seas_len).sum().dropna(dim="time",how="all")
#plot all values and tercile boundaries to get a feeling for the general distribution and if the tercile boundaries make sense
g=sns.displot(ds_season.sel(time=ds_season.time.dt.month==6).precip.values.flatten(),kde=True,aspect=2,color="#CCCCCC")
perc=np.percentile(ds_season.sel(time=ds_season.time.dt.month==6).precip.values.flatten()[~np.isnan(ds_season.sel(time=ds_season.time.dt.month==6).precip.values.flatten())], 33)
plt.axvline(perc,color="#C25048",label="below average")
plt.legend()
g.set(xlabel="Seasonal precipitation (mm)")
plt.title("Distribution of seasonal precipitation in AMJ from 2000-2020")
```

```{code-cell} ipython3
#plot all values and tercile boundaries to get a feeling for the general distribution and if the tercile boundaries make sense
g=sns.displot(ds_season.sel(time=ds_season.time.dt.month==10).precip.values.flatten(),kde=True,aspect=2,color="#CCCCCC")
perc=np.percentile(ds_season.sel(time=ds_season.time.dt.month==10).precip.values.flatten()[~np.isnan(ds_season.sel(time=ds_season.time.dt.month==10).precip.values.flatten())], 33)
plt.axvline(perc,color="#C25048",label="below average")
plt.legend()
g.set(xlabel="Seasonal precipitation (mm)")
plt.title("Distribution of seasonal precipitation in ASO from 2000-2020")
# g.set_title("Distribution of seasonal precipitation in January from 2000-2020")
```

```{code-cell} ipython3
# ds_season_below=xr.open_dataset(chirps_seasonal_lower_tercile_processed_path)
ds_season_below=rioxarray.open_rasterio(chirps_seasonal_lower_tercile_processed_path)
```

```{code-cell} ipython3
ds_season_below
```

```{code-cell} ipython3
#show the data for each month of 2020, clipped to MWI
g=ds_season_below.sel(time=ds_season_below.time.dt.year.isin([2018])).precip.plot(
    col="time",
    col_wrap=6,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation (mm)"
    },
    robust=True,
    cmap="YlOrRd",
)

df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

```{code-cell} ipython3
:tags: [remove_cell]

def compute_zonal_stats_xarray(raster,shapefile,lon_coord="lon",lat_coord="lat",var_name="prob"):
    raster_clip=raster.rio.set_spatial_dims(x_dim=lon_coord,y_dim=lat_coord).rio.clip(shapefile.geometry.apply(mapping),raster.rio.crs,all_touched=False)
    raster_clip_bavg=raster_clip.where(raster_clip[var_name] >=0)
    grid_mean = raster_clip_bavg.mean(dim=[lon_coord,lat_coord]).rename({var_name: "mean_cell"})
    grid_min = raster_clip_bavg.min(dim=[lon_coord,lat_coord]).rename({var_name: "min_cell"})
    grid_max = raster_clip_bavg.max(dim=[lon_coord,lat_coord]).rename({var_name: "max_cell"})
    grid_std = raster_clip_bavg.std(dim=[lon_coord,lat_coord]).rename({var_name: "std_cell"})
    grid_quant90 = raster_clip_bavg.quantile(0.9,dim=[lon_coord,lat_coord]).rename({var_name: "10quant_cell"})
    grid_percbavg = raster_clip_bavg.count(dim=[lon_coord,lat_coord])/raster_clip.count(dim=[lon_coord,lat_coord])*100
    grid_percbavg=grid_percbavg.rename({var_name: "bavg_cell"})
    zonal_stats_xr = xr.merge([grid_mean, grid_min, grid_max, grid_std,grid_quant90,grid_percbavg]).drop("spatial_ref")
    zonal_stats_df=zonal_stats_xr.to_dataframe()
    zonal_stats_df=zonal_stats_df.reset_index()
    zonal_stats_df=zonal_stats_df.drop("quantile",axis=1)
    return zonal_stats_df
```

```{code-cell} ipython3
:tags: [remove_cell]

#select the adms of interest
gdf_reg=gdf_adm1[gdf_adm1.ADM1_FR.isin(adm_sel)]
```

```{code-cell} ipython3
#compute stats
df_stats_reg=compute_zonal_stats_xarray(ds_season_below,gdf_reg,lon_coord="x",lat_coord="y",var_name="precip")
#some dates don't have forecasted values due to dry mask, remove these
df_stats_reg=df_stats_reg.dropna(subset=["mean_cell"])
df_stats_reg["end_time"]=pd.to_datetime(df_stats_reg["time"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_reg["end_month"]=df_stats_reg.end_time.dt.to_period("M")
df_stats_reg["start_time"]=df_stats_reg.end_time.apply(lambda x: x+relativedelta(months=-2))
df_stats_reg["start_month"]=df_stats_reg.start_time.dt.to_period("M")
```

```{code-cell} ipython3
#plot all values and tercile boundaries to get a feeling for the general distribution and if the tercile boundaries make sense
g=sns.displot(df_stats_reg.bavg_cell,kde=True,aspect=2,color="#CCCCCC")
# perc=np.percentile(ds_season.sel(time=ds_season.time.dt.month==10).precip.values.flatten()[~np.isnan(ds_season.sel(time=ds_season.time.dt.month==10).precip.values.flatten())], 33)
# plt.axvline(perc,color="#C25048",label="below average")
# plt.legend()
g.set(xlabel="Percentage of area with below average precipitation")
plt.title("Distribution of percentage of area with below average precipitation from 1982-2021")
```

```{code-cell} ipython3
df_for=pd.read_csv(stats_reg_for_path,parse_dates=["F"])
def get_forecastmonth(pub_month,leadtime):
    return pub_month+relativedelta(months=+int(leadtime))
df_for["for_start"]=df_for.apply(lambda x: get_forecastmonth(x.F,x.L), axis=1)
df_for["for_start_month"]=df_for.for_start.dt.to_period("M")
```

```{code-cell} ipython3
df_for_bavg=df_for[df_for.C==0]
```

```{code-cell} ipython3
df_obsfor=df_stats_reg.merge(df_for_bavg,left_on="start_month",right_on="for_start_month",suffixes=("_obs","_for"))
```

```{code-cell} ipython3
threshold_for_prob=40
```

```{code-cell} ipython3
 #plot the observed vs forecast-observed
g=sns.jointplot(data=df_obsfor[(df_obsfor.C==0)],x="bavg_cell",y=f"{threshold_for_prob}percth_cell", kind="hex",height=8,marginal_kws=dict(bins=40))
g.set_axis_labels("Percentage of area observed below average precipitation", "% of area forecasted >=40% probability below average", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])
plt.colorbar(cax=cbar_ax)
```

```{code-cell} ipython3
 #plot the observed vs forecast-observed for obs<=2mm
g=sns.jointplot(data=df_obsfor[(df_obsfor.C==0)],x="bavg_cell",y="max_cell_for", kind="hex",height=8,marginal_kws=dict(bins=40))#,xlim=(0,100),ylim=(0,100))
g.set_axis_labels("Percentage of area observed below average precipitation", "Max forecasted probability of below average", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])
plt.colorbar(cax=cbar_ax)
```

```{code-cell} ipython3
 #plot the observed vs forecast-observed for obs<=2mm
g=sns.jointplot(data=df_obsfor[(df_obsfor.L==1)],x="bavg_cell",y=f"{threshold_for_prob}percth_cell", kind="hex",height=8,marginal_kws=dict(bins=40))
# #compute the average value of the difference between the forecasted and observed values
# #do this in bins cause else very noisy mean
# bins = np.arange(0,100+2,1)
# group = df_obsfor.groupby(pd.cut(df_obsfor.bavg_cell, bins))
# plot_centers = (bins [:-1] + bins [1:])/2
# plot_values = group["40percth_cell"].median()
# g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Percentage of area observed below average precipitation", "% of area forecasted >=40% probability below average", fontsize=12)
```

```{code-cell} ipython3
#plot with different perc_for >40% and then different leadtimes
```

```{code-cell} ipython3
df_obsfor[f"max_cell_{threshold_for_prob}"]=np.where(df_obsfor.max_cell_for>=threshold_for_prob,1,0)

threshold_area=50
df_obsfor[f"obs_bavg_{threshold_area}"]=np.where(df_obsfor.bavg_cell>=threshold_area,1,0)
df_obsfor[f"for_bavg_{threshold_area}"]=np.where(df_obsfor["40percth_cell"]>=threshold_area,1,0)
```

```{code-cell} ipython3
def compute_confusionmatrix(df,target_var,predict_var, ylabel,xlabel,col_var=None,colp_num=3,title=None):
    #number of dates with observed dry spell overlapping with forecasted per month
    if col_var is not None:
        num_plots = len(df[col_var].unique())
    else:
        num_plots=1
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(15,8))
    if col_var is not None:
        for i, m in enumerate(df.sort_values(by=col_var)[col_var].unique()):
            ax = fig.add_subplot(rows,colp_num,i+1)
            y_target =    df.loc[df[col_var]==m,target_var]
            y_predicted = df.loc[df[col_var]==m,predict_var]
            cm = confusion_matrix(y_target=y_target, 
                                  y_predicted=y_predicted)

            plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax,class_names=["No","Yes"])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{col_var}={m}")
    else:
        ax = fig.add_subplot(rows,colp_num,1)
        y_target =    df[target_var]
        y_predicted = df[predict_var]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax,class_names=["No","Yes"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
```

```{code-cell} ipython3
cm_thresh=compute_confusionmatrix(df_obsfor,f"obs_bavg_50",f"for_bavg_10","boo","bla")
```

```{code-cell} ipython3
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df_obsfor.loc[df_obsfor.L==1,["time","40percth_cell","bavg_cell"]].melt(id_vars='time').rename(columns=str.title)
sns.barplot(x='Time', y='Value', data=tidy, ax=ax1,hue="Variable")
sns.despine(fig)
```

```{code-cell} ipython3
fig,ax=plt.subplots(figsize=(10,10))
sns.lineplot(data=df_obsfor, x="time", y="40percth_cell", hue="L",ax=ax)
# sns.lineplot(data=df_obsfor, x="time",y="bavg_cell",ax=ax,linestyle="--",marker="o")
```

```{code-cell} ipython3
#plot distribution precipitation with and without observed belowavg precip
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_obsfor[df_obsfor.C==0],x="L",y="bavg_cell",ax=ax,color="#66B0EC",hue="max_cell_40",palette={0:"#CCE5F9",1:'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Leadtime")
ax.get_legend().set_title("Dry spell occurred")
```

```{code-cell} ipython3
df_obsfor[(df_obsfor.C==0)&(df_obsfor.L==1)].corr()
```
