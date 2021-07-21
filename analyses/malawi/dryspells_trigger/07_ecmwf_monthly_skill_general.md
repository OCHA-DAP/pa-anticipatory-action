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
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
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
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation_admin1.csv")
```

# Determine general skill per leadtime


### Read in forecast and observational data

```python
da = processing.get_ecmwf_forecast("mwi")
```

```python
da_lt=processing.get_ecmwf_forecast_by_leadtime("mwi")
```

```python
da_obs=xr.open_dataset(chirps_monthly_mwi_path)
da_obs=da_obs.precip
#some problem later on when using rioxarray..
# da_obs=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)
```

```python
da_obs=da_obs.sel(time=slice(da_lt.time.min(), da_lt.time.max()))
```

```python
da_lt=da_lt.sel(time=slice(da_obs.time.min(), da_obs.time.max()))
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

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts.

```python
#other thing to select on is the area..
```

```python
#takes a minute to compute
df_crps=pd.DataFrame(columns=['leadtime', 'crps'])

#selm includes the months to select
#thresh the thresholds
subset_dict={"selm":[1,2],"thresh":[170,180,210]}

for leadtime in da_forecast.leadtime:
    forecast = da_forecast.sel(
    leadtime=leadtime.values)
    observations = da_obs #.reindex({'time': forecast.time})
    # For all dates
    crps = xs.crps_ensemble(observations, forecast,member_dim='number')
    append_dict = {'leadtime': leadtime.values,
                          'crps': crps.values,
                           'std': observations.std().values,
                           'mean': observations.mean().values,
              }

    if "selm" in subset_dict:
        month_str="".join([calendar.month_abbr[m].lower() for m in subset_dict["selm"]])
        # For rainy season only
        observations_rainy = observations.where(observations.time.dt.month.isin(subset_dict['selm']), drop=True)
        forecast_rainy = forecast.where(forecast.time.dt.month.isin(subset_dict['selm']), drop=True)
        crps_rainy = xs.crps_ensemble(
            observations_rainy,
            forecast_rainy,
            member_dim='number')
        append_dict.update({
                f'crps_{month_str}': crps_rainy.values,
                f'std_{month_str}': observations_rainy.std().values,
                f'mean_{month_str}': observations_rainy.mean().values
            })
        if "thresh" in subset_dict:
            for thresh in subset_dict["thresh"]:
                crps_thresh = xs.crps_ensemble(observations_rainy.where(observations_rainy<=thresh), forecast_rainy.where(observations_rainy<=thresh), member_dim='number')
                append_dict.update({
                    f'crps_{month_str}_{thresh}': crps_thresh.values,
                    f'std_{month_str}_{thresh}': observations_rainy.where(observations_rainy<=thresh).std().values,
                    f'mean_{month_str}_{thresh}': observations_rainy.where(observations_rainy<=thresh).mean().values
                })
    
    if "thresh" in subset_dict:
        for thresh in subset_dict["thresh"]:
            crps_thresh = xs.crps_ensemble(observations.where(observations<=thresh), forecast.where(observations<=thresh), member_dim='number')
            append_dict.update({
                f'crps_{thresh}': crps_thresh.values,
                f'std_{thresh}': observations.where(observations<=thresh).std().values,
                f'mean_{thresh}': observations.where(observations<=thresh).mean().values
            })
        df_crps= df_crps.append([append_dict], ignore_index=True)
```

```python
def get_crps(
    ds_observations: xr.Dataset,
    ds_forecast: xr.Dataset,
    normalization: str = None,
    thresh: [float] = None,
) -> pd.DataFrame:
    """
    :param ds_reanalysis: GloFAS reanalysis xarray dataset :param
    ds_reforecast: GloFAS reforecast xarray dataset :param
    normalization: (optional) Can be 'mean' or 'std', reanalysis metric
    to divide the CRPS :param thresh: (optional) Either a single value,
    or a dictionary with format {station name: thresh} :return:
    DataFrame with station column names and leadtime index
    """
    leadtimes = ds_forecast.leadtime.values
    df_crps = pd.DataFrame(index=leadtimes)

    for leadtime in leadtimes:
        forecast = (
            ds_forecast
            .sel(leadtime=leadtime)
            .dropna(dim="time",how="all")
        )
        
        forecast=forecast.sel(time=slice(ds_observations.time.min(), ds_observations.time.max()))
        observations=ds_observations.sel(time=slice(forecast.time.min(), forecast.time.max()))
        if normalization == "mean":
            norm = observations.mean().values
        elif normalization == "std":
            norm = observations.std().values
        elif normalization is None:
            norm = 1
        crps = (
                xs.crps_ensemble(
                    observations, forecast, member_dim="number"
                ).values
                / norm
            )
        df_crps.loc[leadtime, "crps"] = crps
#         # TODO: Add error for other normalization values
#         if thresh is not None:
#             for th in thresh:
#             idx = observations <= thresh
#             forecast, observations = forecast[:, idx], observations[idx]
#             crps = (
#                 xs.crps_ensemble(
#                     observations, forecast, member_dim="number"
#                 ).values
#                 / norm
#             )
#             df_crps.loc[leadtime, station] = crps

    return df_crps
```

```python
df_crps=get_crps(da_obs,da_forecast)
```

```python
for thresh in [210,180,170]:
    da_obs_thresh=da_obs.where(observations<=thresh)
    da_forecast_thresh=da_forecast.where(observations<=thresh)
    df_crps_th=get_crps(da_obs,da_forecast_thresh).rename(columns={"crps":f"crps_{thresh}"})
    df_crps=pd.concat([df_crps,df_crps_th],axis=1)
```

```python
df_crps
```

```python
df_crps_th
```

```python

```

```python
def plot_crps(df_crps, title_suffix=None, ylog=False):
    for basin, stations in STATIONS_BY_MAJOR_BASIN.items():
        fig, ax = plt.subplots()
        for station in stations:
            crps = df_crps[station]
            ax.plot(crps.index, crps, label=station)
        ax.legend()
        title = basin
        if title_suffix is not None:
            title += title_suffix
        ax.set_title(title)
        ax.set_xlabel("Lead time [days]")
        ax.set_ylabel("Normalized CRPS [% error]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid()
        if ylog:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())

```

```python
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                        normalization="mean")
plot_crps(df_crps * 100, title_suffix=" -- all discharge values")
```

```python
def plot_skill(df_crps, division_key=None,
              ylabel="CRPS [mm]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([k for k in df_crps.keys() if "crps" in k]):
        y = df[subset]
        if division_key is not None:
            dkey = f'{division_key}_{subset.split("_")[-1]}' if subset!="crps" else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls="-", c=f'C{i}')
    ax.plot([], [], ls="-", c='k')
    # Add colours to legend
    for i, subset in enumerate([k for k in append_dict.keys() if "crps" in k]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("ECMWF forecast skill in Malawi:\n 2000-2020 forecast")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.grid()
```

```python
def plot_skill_all(df_crps, division_key=None,
              ylabel="CRPS [mm]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([k for k in df_crps.keys() if "crps" in k and not "janfeb" in k]):
        y = df[subset]
        if division_key is not None:
            dkey = f'{division_key}_{subset.split("_")[-1]}' if subset!="crps" else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls="-", c=f'C{i}')
    ax.plot([], [], ls="-", c='k')
    # Add colours to legend
    for i, subset in enumerate([k for k in append_dict.keys() if "crps" in k and not "janfeb" in k]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("ECMWF forecast skill in Malawi:\n 2000-2020 forecast")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.grid()
```

```python
def plot_skill_selm(df_crps, division_key=None,
              ylabel="CRPS [mm]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([k for k in df_crps.keys() if "crps" in k and "janfeb" in k]):
        y = df[subset]
        if division_key is not None:
            dkey = f'{division_key}_{subset.split("_")[-1]}' if subset!="crps" else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls="-", c=f'C{i}')
    ax.plot([], [], ls="-", c='k')
    # Add colours to legend
    for i, subset in enumerate([k for k in append_dict.keys() if "crps" in k and "janfeb" in k]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("ECMWF forecast skill in Malawi:\n 2000-2020 forecast")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.grid()
```

```python
#performance pretty bad.. especially looking at the mean values, it is about 20% off on average for decjanfeb..
# Plot absolute skill
plot_skill_all(df_crps)

# Rainy season performs the worst, but this is likely because 
# the values during this time period are higher. Try using 
# reduced skill (dividing by standard devation).
plot_skill_all(df_crps, division_key='std', ylabel="RCRPS")

#This is perhpas not exactly what we want because we know this 
#data comes from the same location and the dataset has the same properties, 
#but we are splitting it up by mean value. Therefore try normalizing using mean
plot_skill_all(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")

```

```python
# Plot absolute skill
plot_skill_selm(df_crps)

# Rainy season performs the worst, but this is likely because 
# the values during this time period are higher. Try using 
# reduced skill (dividing by standard devation).
plot_skill_selm(df_crps, division_key='std', ylabel="RCRPS")

#This is perhpas not exactly what we want because we know this 
#data comes from the same location and the dataset has the same properties, 
#but we are splitting it up by mean value. Therefore try normalizing using mean
plot_skill_selm(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")

```

```python
lt=2
forecast_lt = da_forecast.sel(
leadtime=lt)#.dropna(dim='time')
forecast_ensmean=da_forecast.mean(dim="number")
```

```python
diff_forobs=forecast_ensmean-da_obs.precip
```

```python
from src.indicators.flooding.glofas import utils
```

```python
(mean_forecast.max()-observations.min())/mean_forecast.max()
```

```python
observations.squeeze().sel(time="2021-01-01",latitude=-11.575001,longitude=33.72499)
```

```python
mean_forecast.sel(time="2021-01-01",latitude=-11.575001,longitude=33.72499)
```

```python
bla=(mean_forecast - observations_test) / observations_test
```

```python
bla_max=bla.where(bla==bla.max(), drop=True).squeeze()
```

```python
bla_max
```

```python
mean_forecast.sel(time=bla_max.time,latitude=bla_max.latitude,longitude=bla_max.longitude)
```

```python
observations_test.squeeze().sel(time=bla_max.time,latitude=bla_max.latitude,longitude=bla_max.longitude)
```

```python
(mean_forecast - observations).sel(time="2021-01-01",latitude=-11.575001,longitude=33.72499) 
```

```python
da_observations=da_obs.to_array("precip").squeeze()

# da_observations =  ds_glofas_reanalysis[station]
# rp_val = df_return_period.loc[rp, station]
# da_observations_ev = da_observations[da_observations > rp_val]
# da_forecast = ds_glofas_reforecast[station]
mpe = np.empty(len(da_forecast.leadtime))
for ilt, leadtime in enumerate(da_forecast.leadtime):
    forecast=da_forecast.sel(leadtime=leadtime)
#     observations, forecast = utils.get_same_obs_and_forecast(da_observations, da_forecast, leadtime)
    mean_forecast=forecast.mean(dim="number")
    #with very small observed values, the mpe explodes
    observations_ge=observations.where(observations>=30)
    mpe[ilt]=(((mean_forecast - observations_ge) / observations_ge).sum(skipna=True)) / np.count_nonzero(~np.isnan(observations_ge)) *100
#     mpe[ilt] = utils.calc_mpe(da_observations, forecast)
#     observations_ev, forecast_ev = utils.get_same_obs_and_forecast(da_observations_ev, da_forecast, leadtime)
#     mpe_ev[ilt] = utils.calc_mpe(observations_ev, forecast_ev)

```

```python
xs.mape(da_observations,da_forecast,skipna=True)
```

```python
fig, ax = plt.subplots()
ax.plot(da_forecast.leadtime, mpe)
# ax.plot(da_forecast.leadtime, mpe_ev, '--', c=f'C{istation}')
# ax.plot([], [], 'k-', label='All values')
# ax.plot([], [], 'k--', label=f'RP > 1 in {rp} y')
# ax.set_ylim(-50, 10)
ax.axhline(y=0, c='k', ls=':')
ax.legend()
ax.grid()
ax.set_xlabel('Leadtime [y]')
ax.set_ylabel('% bias')
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
```

```python

```

```python
denominator = observations
```

```python

```

```python
mean_forecast.size
```

```python
mean_forecast = forecast.mean(axis=0)
denominator = observations
return (
    ((mean_forecast - observations) / denominator).sum()
    / len(observations.time)
    * 100
)
```

```python
df_obs=da_obs.precip.to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forec=forecast_ensmean.to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
```

```python
df_forobs=df_forec.merge(df_obs,how="left",on=["time","latitude","longitude"],suffixes=("_for","_obs"))
```

```python
df_forobs["diff_forobs"]=df_forobs["precip_for"]-df_forobs["precip_obs"]
```

```python
leadtimes=[2]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs[df_forobs.leadtime.isin(leadtimes)],y="diff_forobs",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
# bins = np.arange(0,df_forobs.rollsum_15d.max()+20,10)
# group = df_forobs.groupby(pd.cut(df_forobs.rollsum_15d, bins))
# plot_centers = (bins [:-1] + bins [1:])/2
# plot_values = group.diff_forecobs.median()
# g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
# g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python

```

```python
leadtimes=[2]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs[(df_forobs.leadtime.isin(leadtimes))],y="diff_forobs",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
# bins = np.arange(0,df_forobs.rollsum_15d.max()+20,10)
# group = df_forobs.groupby(pd.cut(df_forobs.rollsum_15d, bins))
# plot_centers = (bins [:-1] + bins [1:])/2
# plot_values = group.diff_forecobs.median()
# g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
# g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python
leadtimes=[2]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs[(df_forobs.leadtime.isin(leadtimes))&(df_forobs.time.dt.month.isin([1,2]))],y="diff_forobs",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
# bins = np.arange(0,df_forobs.rollsum_15d.max()+20,10)
# group = df_forobs.groupby(pd.cut(df_forobs.rollsum_15d, bins))
# plot_centers = (bins [:-1] + bins [1:])/2
# plot_values = group.diff_forecobs.median()
# g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
# g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python
import calendar
```

```python
# leadtimes=[2]
# months=range(1,13)
# for m in months:
#     df_sel=df_forobs[(df_forobs.time.dt.month==m)]
#     #plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
#     g=sns.jointplot(data=df_sel[df_sel.leadtime.isin(leadtimes)],y="diff_forobs",x="precip_obs", 
#                     kind="hex",height=16,joint_kws={ 'bins':'log'})
#     #compute the average value of the difference between the forecasted and observed values
#     #do this in bins cause else very noisy mean
#     bins = np.arange(0,df_sel.precip_obs.max()+20,10)
#     group = df_sel.groupby(pd.cut(df_sel.precip_obs, bins))
#     plot_centers = (bins [:-1] + bins [1:])/2
#     plot_values = group.diff_forobs.median()
#     g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
#     # g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
#     plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
#     # make new ax object for the cbar
#     cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
#     plt.colorbar(cax=cbar_ax)
#     g.ax_joint.legend()
#     g.fig.suptitle(f"Month = {calendar.month_name[m]}")
#     g.fig.tight_layout()
```

```python
leadtimes=[2]
df_sel=df_forobs[(df_forobs.precip_obs<=250)&(df_forobs.time.dt.month.isin([1,2]))]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_sel[df_sel.leadtime.isin(leadtimes)],y="diff_forobs",x="precip_obs", 
                kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_sel.precip_obs.max()+20,10)
group = df_sel.groupby(pd.cut(df_sel.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
# g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python
leadtimes=[2]
df_sel=df_forobs[(df_forobs.time.dt.month.isin([1,2]))]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_sel[df_sel.leadtime.isin(leadtimes)],y="diff_forobs",x="precip_obs", 
                kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_sel.precip_obs.max()+20,10)
group = df_sel.groupby(pd.cut(df_sel.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
# g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python
df_diff_forobs=diff_forobs.to_dataframe(name="precip").reset_index()
```

```python
df_diff_forobs
```

```python
leadtimes=[2]
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs[df_forobs.leadtime.isin(leadtimes)],y="diff_forecobs",x="mean_cell_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
# bins = np.arange(0,df_forobs.rollsum_15d.max()+20,10)
# group = df_forobs.groupby(pd.cut(df_forobs.rollsum_15d, bins))
# plot_centers = (bins [:-1] + bins [1:])/2
# plot_values = group.diff_forecobs.median()
# g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed 15 day sum (mm)", "Forecasted 15 day sum - Observed 15 day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_15days_density.png"))
```

```python
forecast_ensmean
```

```python
bias=forecast_ensmean-observations
```

```python
bias
```
