### Evaluating the forecast skill of ECMWF seasonal forecast in Malawi
This notebook is to compare the forecast skill of ECMWF's seasonal forecast for various lead times. We use the monthly total precipitation that is forecasted by this forecast. As ground truth, we use CHIRPS observations. 

We first look assess the skill at cell level. For this we compute the CRPS. We investigate this CRPS for different sets of data, e.g. for months with low rainfall. 
Thereafter, we assess the skill at the admin1 level. We compare the forecasted and observed values visually. Thereafter we compute the bias, and more specifically look at the areas, leadtimes, and months that are of interest for the trigger.  

`mwi_ecmwf_monthly_skill_dryspells.md` assesses the skill of the forecast for dry spells specifically. 

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xskillscore as xs
import numpy as np

import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
import glob
import itertools

import math
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
from src.utils_general.statistics import calc_mpe
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
```

```python
import plotly.express as px 
```

```python
import calendar
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR, "exploration", country_iso3,'chirps')

chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")
ecmwf_country_data_processed_dir = os.path.join(country_data_processed_dir,"ecmwf")
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
all_dry_spells_list_path=os.path.join(country_data_processed_dir,"dry_spells","full_list_dry_spells.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","chirps_monthly_total_precipitation_admin1.csv")
```

### Read in forecast and observational data

```python
da_lt=processing.get_ecmwf_forecast_by_leadtime("mwi")
```

```python
da_obs=xr.load_dataset(chirps_monthly_mwi_path)
da_obs=da_obs.precip
#some problem later on when using rioxarray..
# da_obs=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)
```

```python
#interpolate forecast data such that it has the same resolution as the observed values
#using "nearest" as interpolation method and not "linear" because the forecasts are designed to have sharp edged and not be smoothed
da_forecast=da_lt.interp(latitude=da_obs["latitude"],longitude=da_obs["longitude"],method="nearest")
# da_forecast=da_lt.interp(latitude=da_obs["y"],longitude=da_obs["x"],method="nearest")
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2020-06-01'
# end = '2020-10-31'

rf_list_slice = da_lt.sel(time=start,latitude=da_lt.latitude.values[10],longitude=da_lt.longitude.values[5])

rf_list_slice.dropna("leadtime").plot.line(label='Historical', c='grey',hue="number",add_legend=False)
rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(label='Historical', c='red',hue="number",add_legend=False)
plt.show()
```

```python
#plot distribution per month and then justify taking values between 50 and 200
```

#### Compute the Continuous Ranked Probability Score (CRPS)

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts.

```python
#other thing to select on is the area..
```

```python
da_forecast=da_forecast.sel(time=slice(da_obs.time.min(), da_obs.time.max()))
da_obs=da_obs.sel(time=slice(da_forecast.time.min(), da_forecast.time.max()))
```

```python
df_crps=processing.get_crps_ecmwf(da_obs,da_forecast)
for thresh in [210,180,170]:
    df_crps_th=processing.get_crps_ecmwf(da_obs,da_forecast,thresh=thresh).rename(columns={"crps":f"crps_{thresh}"})
    df_crps=pd.concat([df_crps,df_crps_th],axis=1)
```

```python
df_crps_norm=processing.get_crps_ecmwf(da_obs,da_forecast,normalization="mean")
for thresh in [210,180,170]:
    df_crps_th_norm=processing.get_crps_ecmwf(da_obs,da_forecast,normalization="mean",thresh=thresh).rename(columns={"crps":f"crps_{thresh}"})
    df_crps_norm=pd.concat([df_crps_norm,df_crps_th_norm],axis=1)
```

```python
# sel_m=[[1,2],[1,2,3,4],[11,12,1,2]]
sel_m=[[1,2],[11,12,1,2,3,4]]
df_crps_months_norm=processing.get_crps_ecmwf(da_obs,da_forecast,normalization="mean")
for m in sel_m:
    month_str="".join([calendar.month_abbr[i].lower() for i in m])
    da_obs_m = da_obs.where(da_obs.time.dt.month.isin(m), drop=True)
    da_forecast_m = da_forecast.where(da_forecast.time.dt.month.isin(m), drop=True)
    df_crps_m_norm=processing.get_crps_ecmwf(da_obs_m,da_forecast_m,normalization="mean").rename(columns={"crps":f"crps_{month_str}"})
    df_crps_months_norm=pd.concat([df_crps_months_norm,df_crps_m_norm],axis=1)
```

```python
sel_m=[[1,2],[11,12,1,2,3,4]]
df_crps_months=processing.get_crps_ecmwf(da_obs,da_forecast)
for m in sel_m:
    month_str="".join([calendar.month_abbr[i].lower() for i in m])
    da_obs_m = da_obs.where(da_obs.time.dt.month.isin(m), drop=True)
    da_forecast_m = da_forecast.where(da_forecast.time.dt.month.isin(m), drop=True)
    df_crps_m=processing.get_crps_ecmwf(da_obs_m,da_forecast_m).rename(columns={"crps":f"crps_{month_str}"})
    df_crps_months=pd.concat([df_crps_months,df_crps_m],axis=1)
```

```python
fig, axes = plt.subplots(1,2,figsize=(20,8))
for c in df_crps_months.columns:
    if "_" in c:
        label= f"{c.split('_')[-1]}"
    else:
        label="all"
    axes[0].plot(df_crps_months.index, df_crps_months[c], label=label)
    axes[1].plot(df_crps_months_norm.index, df_crps_months_norm[c], label=label)
axes[0].set_title("CRPS")
axes[0].set_ylabel("CRPS [mm]")
axes[1].set_title("Normalized CRPS")
axes[1].set_ylabel("Normalized CRPS [% error]")

for ax in axes:
    ax.set_xlabel("Leadtime [months]")
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()

fig.suptitle("CRPS for different months")
fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.9));
```

```python
fig, axes = plt.subplots(1,2,figsize=(20,8))
for c in df_crps.columns:
    if "_" in c:
        label= f"<={c.split('_')[-1]}"
    else:
        label="all"
    axes[0].plot(df_crps.index, df_crps[c], label=label)
    axes[1].plot(df_crps_norm.index, df_crps_norm[c], label=label)
axes[0].set_title("CRPS")
axes[0].set_ylabel("CRPS [mm]")
axes[1].set_title("Normalized CRPS")
axes[1].set_ylabel("Normalized CRPS [% error]")

for ax in axes:
    ax.set_xlabel("Leadtime [months]")
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()

fig.suptitle("CRPS for different thresholds")
fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.9));
```

### Compute the bias
While the CRPS gives a good indication of the skill across ensemble members, we want to understand better which direction the error has and whether it differs across ranges of precipitation. We do this by looking at the bias. 

Often the bias is computed using the MPE. However, for very small values the MPE is not suitable as it disproportionally explodes. This is also the case for our data. We therefore instead solely focus on the difference between forecasted and observed values, instead of looking at the percentual difference. 

To do so we aggregate the ensemble members to one number, for which we chose to use the median. 

We firstly plot the observed vs forecasted-observed values across all leadtimes, dates, and cells. 
From this we can see that

- Most months with very low precipitation were correctly classified. 
- Months with less than 300mm have the tendency to be overpredicted, i.e. we see a positive bias
- Months with more than 300mm have the tendency to be underpredicted, i.e. we see a negative bias

```python
df_obs=da_obs.to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forec=da_forecast.mean(dim="number").to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forobs=df_forec.merge(df_obs,how="left",on=["time","latitude","longitude"],suffixes=("_for","_obs"))
df_forobs["diff_forobs"]=df_forobs["precip_for"]-df_forobs["precip_obs"]
```

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs,y="diff_forobs",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs.precip_obs.max()+20,10)
group = df_forobs.groupby(pd.cut(df_forobs.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed monthly precipitation (mm)", "Forecasted - Observed monthly precipitation (mm)", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
g.fig.suptitle("Bias plot of observed vs forecasted values")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
```

Since our main months of interest are January and February, we zoom in on these months. 
From here we can see that the bias is a lot higher for these months compared to all months. Again for values up to 300 the forecast has a tendency to overpredict and for higher values to underpredict. 

It is hard to say whether this increased bias is due to the period or the range of precipitation, while these are largely intertwined. 

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
df_forobs_selm=df_forobs[(df_forobs.time.dt.month.isin([1,2]))]
g=sns.jointplot(data=df_forobs_selm,y="diff_forobs",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs_selm.precip_obs.max()+20,10)
group = df_forobs_selm.groupby(pd.cut(df_forobs_selm.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed monthly precipitation (mm)", "Forecasted - Observed monthly precipitation (mm)", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
g.fig.suptitle("Bias plot of observed vs forecasted values for January and February")
g.fig.subplots_adjust(top=0.95)
```

```python
#mean vs median
df_obs=da_obs.to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forec=da_forecast.median(dim="number").to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forobs=df_forec.merge(df_obs,how="left",on=["time","latitude","longitude"],suffixes=("_for","_obs"))
df_forobs["diff_forobs"]=df_forobs["precip_for"]-df_forobs["precip_obs"]
```

question: better use mean or median? 
does CI make sense? so many values that CI is very small

```python
def calc_diff_stats(observations,forecast):
    #median and mean make a hug difference! --> std quite large? 
    diff=forecast-observations
    diff_mean=diff.mean()
    diff_median=diff.median()
    diff_std=diff.std()
    diff_count=diff.count()
    ci95_hi = diff_mean + 1.95*diff_std/math.sqrt(diff_count)
    ci95_lo = diff_mean - 1.95*diff_std/math.sqrt(diff_count)
    return diff_median, diff_mean, diff_std, ci95_hi, ci95_lo

def compute_diff_cats(df,obs_col,for_col,threshold_list=None,adm_list=None,month_list=None):
    leadtimes=df.leadtime.sort_values().unique()
    df_diff=pd.DataFrame(leadtimes,columns=["leadtime"]).set_index('leadtime')
    for ilt, leadtime in enumerate(leadtimes):
        df_lt=df[df.leadtime==leadtime]
        med,me,std,ci_hi,ci_lo=calc_diff_stats(df_lt[obs_col],df_lt[for_col])
        df_diff.loc[leadtime,"all_med"]=med
        df_diff.loc[leadtime,"all_mean"]=me
        df_diff.loc[leadtime,"all_ci_lo"]=ci_lo
        df_diff.loc[leadtime,"all_ci_hi"]=ci_hi
        if threshold_list:
            for thresh in threshold_list:
                df_lt_thresh=df_lt[df_lt[obs_col]<=thresh]
                med,me,std,ci_hi,ci_lo = calc_diff_stats(df_lt_thresh[obs_col],df_lt_thresh[for_col])
                df_diff.loc[leadtime,f"thresh_{thresh}_med"]=med
        if month_list:
            for m in month_list:
                month_str="".join([calendar.month_abbr[i].lower() for i in m])
                df_lt_m=df_lt[df_lt.time.dt.month.isin(m)]
                med,me,std,ci_hi,ci_lo = calc_diff_stats(df_lt_m[obs_col],df_lt_m[for_col])
                df_diff.loc[leadtime,f"{month_str}_med"]=med
                df_diff.loc[leadtime,f"{month_str}_ci_lo"]=ci_lo
                df_diff.loc[leadtime,f"{month_str}_ci_hi"]=ci_hi
                
    return df_diff
```

```python
df_diff=compute_diff_cats(df_forobs,"precip_obs","precip_for",threshold_list=[210,180,170],month_list=[[m] for m in range(1,13)])
```

```python
# ax=sns.histplot(df_forobs,x="diff_forobs")
# ax.set_xlim(-20,20)
```

The median is very different, not sure which makes most sense here.. Probably the median, as that is also what we using when we aggregate to admin1. 

```python
fig, ax = plt.subplots()
ax.plot(df_diff.index,df_diff["all_mean"])
ax.fill_between(df_diff.index, df_diff.all_ci_lo, df_diff.all_ci_hi,alpha=0.2)
ax.plot(df_diff.index, df_diff["all_med"])
# for c in col_list:
#     ax.plot(df_crps.index, df_crps[c], label=c)
# ax.legend(bbox_to_anchor=(1.05, 1))
# if title is not None:
#     ax.set_title(title)
# ax.set_xlabel("Lead time [months]")
# ax.set_ylabel("Normalized CRPS [% error]")
ax.grid()
# if ylog:
#     ax.set_yscale('log')
#     ax.yaxis.set_major_formatter(ScalarFormatter())
```

When looking per month, we can see that the bias clearly differs per month. The months around the rainy season have the highest bias (nov, dec, jan, feb). This can either be because values are generally larger, or because the forecast has less skill. 

```python
px.line(df_diff.reset_index(), x='leadtime', y=[c for c in df_diff.columns if "thresh" not in c])
```

We can also look at the bias for different thresholds. Here we see that there is not much difference. This is however partly caused by the fact that most months have very low values. If you would separate by month and threshold, you would probably get different patterns

```python
px.line(df_diff.reset_index(), x='leadtime', y=[c for c in df_diff.columns if "thresh" in c]+["all"])
```

The CRPS looks a lot worse than the difference plots. Why? 

```python

```
