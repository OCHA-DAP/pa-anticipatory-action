### Evaluating the forecast skill of GloFAS in Bangladesh

This notebook is to compare the forecast skill of GloFAS for various lead times. We are comparing the reforecast product (lead time 5-30 days) against the reanalysis product. This is an improvement on the ```process-glofas``` notebook and takes the processed GloFAS data created by ```get_glofas_data.py```. 

We're specifically interested in the forecast skill during times of potential flooding, here estimated to be between June - Oct. 

```python
from importlib import reload
from pathlib import Path
import os

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
SKILL_DIR = DATA_DIR / 'exploration/bangladesh/GLOFAS_Data'
SKILL_FILE = 'forecast_skill.csv'
GLOFAS_VERSION = 2
```

### Read in forecast and reanalysis

Forecast data is shifted to match the day it is supposed to be forecasting. Reforecast is not interpolated, but we do read in the interpolated version to make plotting easier.

```python
da_glofas_reanalysis = rd.get_glofas_reanalysis(version=GLOFAS_VERSION)
da_glofas_forecast = rd.get_glofas_forecast(version=GLOFAS_VERSION)
da_glofas_reforecast = rd.get_glofas_reforecast(version=GLOFAS_VERSION, interp=False)
da_glofas_reforecast_interp = rd.get_glofas_reforecast(version=GLOFAS_VERSION)
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2000-06-01'
end = '2000-10-31'
rf_list_slice = da_glofas_reforecast_interp.sel(time=slice(start,end))
ra_slice = da_glofas_reanalysis.sel(time=slice(start, end))

rf_list_slice.mean(axis=1).plot.line(x='time', add_legend=True)
ra_slice.plot.line(label='Historical', c='k')
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts. This is also what GloFAS uses in evaluating their own forecast skill.

```python
def is_rainy_season(month):
    # June through October
    return (month >= 6) & (month <= 10)

def is_dry_season(month):
    # June through October
    return (month < 6) | (month > 10)


df_crps = pd.DataFrame(columns=['leadtime', 'crps'])
for leadtime in da_glofas_reforecast.leadtime[:-1]:
    forecast = da_glofas_reforecast.sel(
        leadtime=leadtime.values).dropna(dim='time')
    observations = da_glofas_reanalysis.reindex({'time': forecast.time})
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
                              'crps': crps.values,
                               'std': observations.std().values,
                               'mean': observations.mean().values,
                              'crps_rainy': crps_rainy.values,
                               'std_rainy': observations_rainy.std().values,
                               'mean_rainy': observations_rainy.mean().values,
                                'crps_dry': crps_dry.values,
                               'std_dry': observations_dry.std().values,
                               'mean_dry': observations_dry.mean().values
                              }], ignore_index=True)
```

Plot the skill for the different month divisions

```python
# Get skill from GloFAS website
df_skill = pd.read_csv(SKILL_DIR / SKILL_FILE, header=None)

# Plot absolute skill
fig, ax = plt.subplots()
ax.plot(df_crps['leadtime'], df_crps['crps'], 
        label='Full year')
ax.plot(df_crps['leadtime'], df_crps['crps_rainy'], 
        label='Rainy season')
ax.plot(df_crps['leadtime'], df_crps['crps_dry'], 
        label='Dry season')
ax.plot(df_skill[0], df_skill[1], label='From website')
ax.set_title("GloFAS forecast skill at Bahadurabad:\n 1999-2018 reforecast")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("CRPS [m$^3$ s$^{-1}$]")
ax.legend()
```

Rainy season performs the worst, but this is likely because the values during this time period are higher. Try using reduced skill (dividing by standard devation).

```python
# Plot reduced skill with std
fig, ax = plt.subplots()
ax.plot(df_crps['leadtime'], 
        df_crps['crps'] / df_crps['std'], 
        label='Full year')
ax.plot(df_crps['leadtime'], 
        df_crps['crps_rainy'] / df_crps['std_rainy'], 
        label='Rainy season')
ax.plot(df_crps['leadtime'], 
        df_crps['crps_dry'] / df_crps['std_dry'], 
        label='Dry season')
ax.set_title("GloFAS relative forecast skill at Bahadurabad:\n 1999-2018 reforecast")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("RCRPS")
ax.legend()
```

This is perhpas not exactly what we want because we know this data comes from the same location and the dataset has the same properties, but we are splitting it up by mean value. Therefore try normalizing using mean

```python
# Plot normalized skill with mean
fig, ax = plt.subplots()
ax.plot(df_crps['leadtime'], 
        df_crps['crps'] / df_crps['mean'], 
        label='Full year')
ax.plot(df_crps['leadtime'], 
        df_crps['crps_rainy'] / df_crps['mean_rainy'], 
        label='Rainy season')
ax.plot(df_crps['leadtime'], 
        df_crps['crps_dry'] / df_crps['mean_dry'], 
        label='Dry season')
ax.set_title("GloFAS relative forecast skill at Bahadurabad:\n 1999-2018 reforecast")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("NCRPS (CRPS / mean)")
ax.legend()
```

### Bias: rank histogram

Plot a rank histogram for the forecast and re-forecast, for both full year and rainy season, to evaluate bias

```python
def get_rank(observations, forecast):
    # Create array of both obs and forecast
    rank_array = np.concatenate(([observations], forecast))
    # Calculate rank and take 0th array, which should be the obs
    rank = rankdata(rank_array, axis=0)[0]
    return rank

def plot_hist(da_forecast):
    fig, ax = plt.subplots()
    for leadtime in da_forecast.leadtime[:-1]:
        forecast = da_forecast.sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_glofas_reanalysis.reindex({'time': forecast.time})
        rank = get_rank(observations.values, forecast.values)
        ax.hist(rank, histtype='step', label=int(leadtime),
               bins=np.arange(0.5, max(rank)+1.5, 1), alpha=0.8)
    ax.legend(loc=9, title="Lead time (days)")
    ax.set_xlabel('Rank')
    ax.set_ylabel('Number')

for da_forecast in [da_glofas_reforecast, da_glofas_forecast]:
    plot_hist(da_forecast)
    da_forecast = da_forecast.sel(
        time=is_rainy_season(da_forecast['time.month']))
    plot_hist(da_forecast)
    

```

```python

```
