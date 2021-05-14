### Evaluating the forecast skill of ECMWF seasonal forecast in Malawi

This notebook is to compare the forecast skill of ECMWF's seasonal forecast for various lead times. We use the monthly total precipitation that is forecasted by this forecast. We are comparing the forecast against CHIRPS observations. It takes the processed ECMWF data created by ```get_ecmwf_seasonal_data.py```. 

We're specifically interested in the forecast skill during times where dry spells have the most impact, here estimated to be between Dec - Feb.

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os

import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xskillscore as xs
import numpy as np
from scipy.stats import rankdata

import read_in_data as rd
reload(rd)

mpl.rcParams['figure.dpi'] = 200


DATA_DIR = Path(os.environ["AA_DATA_DIR"])
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
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
country_dir = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR, "exploration", country_iso3,'chirps')
# chirps_monthly_dir=os.path.join(drought_data_exploration_dir,"CHIRPS")
# chirps_monthly_path=os.path.join(chirps_monthly_dir,"chirps_global_monthly.nc")
chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")
```

### Read in forecast and observational data

```python
da = rd.get_ecmwf_forecast()
```

```python
da_lt=rd.get_ecmwf_forecast_by_leadtime()
```

```python
# rd.compute_stats_per_admin("malawi")
```

```python
da_obs=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)
#only select the years for which we also identified dry spells
da_obs=da_obs.sel(time=da_obs.time.dt.year.isin(range(2000,2021)))
```

```python
#interpolate forecast data such that it has the same resolution as the observed values
#not sure if nearest or linear is most suitable here..
#TODO: add this to read_in_data?
da_lt_interp=da_lt.interp(latitude=da_obs["y"],longitude=da_obs["x"],method="nearest")
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2020-06-01'
# end = '2020-10-31'

rf_list_slice = da_lt.sel(time=start,latitude=da_lt.latitude.values[5],longitude=da_lt.longitude.values[5])

rf_list_slice.dropna("leadtime").plot.line(label='Historical', c='grey',hue="number",add_legend=False)
rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(label='Historical', c='red',hue="number",add_legend=False)
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts.

```python
def is_rainy_season(month):
    # June through October
    return (month >= 12) | (month <= 2)

def is_dry_season(month):
    # June through October
    return (month < 12) & (month > 2)

df_crps=pd.DataFrame(columns=['leadtime', 'crps'])

for leadtime in da_lt_interp.leadtime:
    forecast = da_lt_interp.sel(
    leadtime=leadtime.values).dropna(dim='time')
    observations = da_obs.reindex({'time': forecast.time})
    # For all dates
    crps = xs.crps_ensemble(observations, forecast,member_dim='number')

    # For rainy season only
    observations_rainy = observations.sel(time=is_rainy_season(observations['time.month']))
    crps_rainy = xs.crps_ensemble(
        observations_rainy,
        forecast.sel(time=is_rainy_season(forecast['time.month'])),
        member_dim='number')
    # Dry season only
    observations_dry = observations.sel(time=is_dry_season(observations['time.month']))
    crps_dry = xs.crps_ensemble(
        observations_dry,
        forecast.sel(time=is_dry_season(forecast['time.month'])),
        member_dim='number')
    df_crps = df_crps.append([{'leadtime': leadtime.values,
                              'crps': crps.precip.values,
                               'std': observations.precip.std().values,
                               'mean': observations.precip.mean().values,
                              'crps_decjanfeb': crps_rainy.precip.values,
                               'std_decjanfeb': observations_rainy.precip.std().values,
                               'mean_decjanfeb': observations_rainy.precip.mean().values,
                                'crps_othermonths': crps_dry.precip.values,
                               'std_othermonths': observations_dry.precip.std().values,
                               'mean_othermonths': observations_dry.precip.mean().values
                              }], ignore_index=True)
```

```python
df_crps
```

```python
def plot_skill(df_crps, division_key=None,
              ylabel="CRPS [mm]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([None, 'decjanfeb', 'othermonths']):
        ykey = f'crps_{subset}' if subset is not None else 'crps'
        y = df[ykey]
        if division_key is not None:
            dkey = f'{division_key}_{subset}' if subset is not None else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls="-", c=f'C{i}')
    ax.plot([], [], ls="-", c='k')#, label=f'version {version}')
    # Add colours to legend
    for i, subset in enumerate(['full year', 'decjanfeb', 'othermonths']):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("ECMWF forecast skill in Malawi:\n 2000-2020 forecast")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(ylabel)
    ax.legend()
```

```python
#performance pretty bad.. especially looking at the mean values, it is about 20% off on average for decjanfeb..
# Plot absolute skill
plot_skill(df_crps)

# Rainy season performs the worst, but this is likely because 
# the values during this time period are higher. Try using 
# reduced skill (dividing by standard devation).
plot_skill(df_crps, division_key='std', ylabel="RCRPS")

#This is perhpas not exactly what we want because we know this 
#data comes from the same location and the dataset has the same properties, 
#but we are splitting it up by mean value. Therefore try normalizing using mean
plot_skill(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")

```

```python
#TODO: understand if ROC is good measure for this data (with ensemble and different rasters)
#if so, understand what it exactly means in this case

# ROC for probabilistic forecasts and bin_edges='continuous' default
roc = xs.roc(observations.precip > 100, (forecast > 100).mean("number"), return_results='all_as_metric_dim')#,dim=["y","x","time"])

plt.figure(figsize=(4, 4))
plt.plot([0, 1], [0, 1], 'k:')
roc.to_dataset(dim='metric').plot.scatter(y='true positive rate', x='false positive rate')
roc.sel(metric='area under curve').values[0]
```

### Bias: rank histogram

Plot a rank histogram for the forecast and re-forecast, for both full year and rainy season, to evaluate bias

```python
#TODO: understand and possibly implement for ECMWF
```

```python
def get_rank(observations, forecast):
    # Create array of both obs and forecast
    rank_array = np.concatenate(([observations], forecast))
    # Calculate rank and take 0th array, which should be the obs
    rank = rankdata(rank_array, axis=0)[0]
    return rank

def plot_hist(da_forecast, da_reanalysis):
    fig, ax = plt.subplots()
    for leadtime in da_forecast.leadtime:
        forecast = da_forecast.sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_reanalysis.reindex({'time': forecast.time})
        rank = get_rank(observations.values, forecast.values)
        ax.hist(rank, histtype='step', label=int(leadtime),
               bins=np.arange(0.5, max(rank)+1.5, 1), alpha=0.8)
    ax.legend(loc=9, title="Lead time (days)")
    ax.set_xlabel('Rank')
    ax.set_ylabel('Number')
    
for version in [2, 3]:
    forecast_list = [da_glofas_reforecast]
    if version == 2:
        forecast_list += [da_glofas_forecast]
    for da_forecast in forecast_list:
        plot_hist(da_forecast[version], da_glofas_reanalysis[version])
        plot_hist(da_forecast[version].sel(
            time=is_rainy_season(da_forecast[version]['time.month'])),
                  da_glofas_reanalysis[version])
    

```

### Skill vs spread
RMSE vs root of average variance (RAV)


```python
def calc_rmse(observations, forecast):
    return (((observations - forecast.mean(axis=0)) ** 2).sum() \
            / len(observations.time)) ** (1/2)

def calc_rav(forecast):
    return (forecast.std(axis=0) ** 2).mean()**(1/2)

df_skill_spread = {
    2: pd.DataFrame(columns=['leadtime', 'rmse', 'rav']),
    3: pd.DataFrame(columns=['leadtime', 'rmse', 'rav']),
}   


for version in [2,3]:
    da_forecast = da_glofas_reforecast[version]
    da_reanalysis = da_glofas_reanalysis[version]
    for leadtime in da_forecast.leadtime:
        forecast = da_forecast.sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_reanalysis.reindex({'time': forecast.time})
        rmse = calc_rmse(observations, forecast)
        rav = calc_rav(forecast)
        
        
        df_skill_spread[version] = df_skill_spread[version].append([
                                {'leadtime': leadtime.values,
                                  'rmse':rmse.values,
                                     'rav': rav.values
                                  }], ignore_index=True)
    
```

```python
# Plot absolute skill
fig, ax = plt.subplots()
for version, ls in zip([2,3], [':', '--']):    
    df = df_skill_spread[version]
    ax.plot(df['leadtime'], df['rmse'], ls=ls, c='C0')
    ax.plot(df['leadtime'], df['rav'], ls=ls, c='C1')
    ax.plot([], [], ls=ls, c='k', label=f'version {version}')
# Add colours to legend
for i, subset in enumerate(['RMSE', 'spread']):
    ax.plot([], [], c=f'C{i}', label=subset)
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("Error [m$^3$ s$^{-1}$]")
ax.legend()
```

### Bias magnitude
To try and quantify bias magnitude, examine MAE and RSME differences for positive and negative errors

```python
def calc_mae(observations, forecast):
    return np.abs(observations - forecast.mean(axis=0)).sum() \
        / len(observations.time)

# mean error
def calc_me(observations, forecast):
    return (forecast.mean(axis=0) - observations).sum() \
        / len(observations.time)

def calc_mpe(observations, forecast):
    mean_forecast = forecast.mean(axis=0)
    return ((mean_forecast - observations) / mean_forecast).sum() \
        / len(observations.time) * 100


def calc_skew(diff):
    return (np.mean(diff) - np.median(diff)) / np.std(diff)

df_bias = {
    2: pd.DataFrame(columns=['leadtime']),
    3: pd.DataFrame(columns=['leadtime']),
}   


for version in [2,3]:
    da_forecast = da_glofas_reforecast[version]
    da_reanalysis = da_glofas_reanalysis[version]
    for leadtime in da_forecast.leadtime:
        forecast = da_forecast.sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_reanalysis.reindex({'time': forecast.time})
        observations_rainy = observations.sel(time=is_rainy_season(observations['time.month']))
        forecast_rainy = forecast.sel(time=is_rainy_season(forecast['time.month']))
        diff = (forecast - observations).values.flatten()
        diff_rainy = (forecast_rainy - observations_rainy).values.flatten()

        df_bias[version] = df_bias[version].append([
                                {'leadtime': leadtime.values,  
                                'me': calc_me(observations, forecast),
                                 'me_rainy': calc_me(observations_rainy, forecast_rainy),
                                 'skew': calc_skew(diff),
                                 'skew_rainy': calc_skew(diff_rainy),
                                'mpe': calc_mpe(observations, forecast),
                              'mpe_rainy': calc_mpe(observations_rainy, forecast_rainy)
                                }], ignore_index=True)
        

    
```

```python
fig, ax = plt.subplots()
for version, ls in zip([2,3], [':', '--']):    
    df = df_bias[version]
    ax.plot(df['leadtime'], df['mpe'], ls=ls, c='C0')
    ax.plot(df['leadtime'], df['mpe_rainy'], ls=ls, c='C1')
    ax.plot([], [], ls=ls, c='k', label=f'version {version}')
# Add colours to legend
for i, subset in enumerate(['full year', 'rainy']):
    ax.plot([], [], c=f'C{i}', label=subset)

    
ax.set_xlabel("Lead time (days)")
#ax.set_ylabel("Mean error [m$^3$ s$^{-1}$]")
ax.set_ylabel("Mean error (%)")

ax.legend()
```

```python

```
