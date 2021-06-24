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

from utils import utils

mpl.rcParams['figure.dpi'] = 200


DATA_DIR = Path(os.environ["AA_DATA_DIR"])
SKILL_FILE = 'forecast_skill.csv'
LEADTIMES_V2 = [5, 10, 15, 20, 25, 30]
```

### Read in forecast and reanalysis

Forecast data is shifted to match the day it is supposed to be forecasting. Reforecast is not interpolated, but we do read in the interpolated version to make plotting easier.

```python
da_glofas_reanalysis = {
    2: utils.get_glofas_reanalysis(version=2),
    3: utils.get_glofas_reanalysis()
}

da_glofas_forecast = {
    2: utils.get_glofas_forecast(version=2, leadtimes=LEADTIMES_V2),
}

da_glofas_reforecast = {
    2: utils.get_glofas_reforecast(version=2, interp=False, leadtimes=LEADTIMES_V2),
    3: utils.get_glofas_reforecast(interp=False)
}

da_glofas_reforecast_interp = {
    2: utils.get_glofas_reforecast(version=2, leadtimes=LEADTIMES_V2),
    3: utils.get_glofas_reforecast()
}
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2001-01-01'
end = '2001-10-31'
version = 2

rf_list_slice = da_glofas_reforecast_interp[version].sel(time=slice(start,end))
ra_slice = da_glofas_reanalysis[version].sel(time=slice(start, end))

rf_list_slice.mean(axis=1).plot.line( x='time', add_legend=True)
ra_slice.plot.line(label='Historical', c='k')
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts. This is also what GloFAS uses in evaluating their own forecast skill.

```python
skill_thresh_vals = [90000, 95000, 100000]

df_crps = {
    2: pd.DataFrame(columns=['leadtime', 'crps']),
    3: pd.DataFrame(columns=['leadtime', 'crps'])
}

for version in [2,3]:
    for leadtime in da_glofas_reforecast_interp[version].leadtime:
        forecast = da_glofas_reforecast_interp[version].sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_glofas_reanalysis[version].reindex({'time': forecast.time})
        # For all dates
        crps = xs.crps_ensemble(observations, forecast,member_dim='number')
        append_dict = {'leadtime': leadtime.values,
                                  'crps': crps.values,
                                   'std': observations.std().values,
                                   'mean': observations.mean().values,
                      }
        # For high values only
        for thresh in skill_thresh_vals:
            idx = observations > thresh
            crps = xs.crps_ensemble(observations[idx], forecast[:, idx], member_dim='number')
            append_dict.update({
                f'crps_{thresh}': crps.values,
                f'std_{thresh}': observations[idx].std().values,
                f'mean_{thresh}': observations[idx].mean().values
            })
        df_crps[version] = df_crps[version].append([append_dict], ignore_index=True)
sum(idx)
```

```python
def plot_skill(df_crps, division_key=None, add_line_from_website=False,
              ylabel="CRPS [m$^3$ s$^{-1}$]"):
    fig, ax = plt.subplots()
    for version, ls in zip([2,3], [':', '--']):
        df = df_crps[version].copy()
        for i, subset in enumerate([None] + skill_thresh_vals):
            ykey = f'crps_{subset}' if subset is not None else 'crps'
            y = df[ykey]
            if division_key is not None:
                dkey = f'{division_key}_{subset}' if subset is not None else division_key
                y /= df[dkey]
            ax.plot(df['leadtime'], y, ls=ls, c=f'C{i}')
        ax.plot([], [], ls=ls, c='k', label=f'version {version}')
    if add_line_from_website:
        ax.plot(df_skill[0], df_skill[1], ls='-', c='k', lw=0.5, label='from website')
    # Add colours to legend
    for i, subset in enumerate(['full year'] + [f'>{thresh}' for thresh in skill_thresh_vals]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("GloFAS forecast skill at Bahadurabad:\n 1999-2018 reforecast")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    ax.legend()
```

```python
# Get skill from GloFAS website
df_skill = pd.read_csv(utils.GLOFAS_EXPLORATION_FOLDER / SKILL_FILE, header=None)

# Plot absolute skill
plot_skill(df_crps, add_line_from_website=True)

# Rainy season performs the worst, but this is likely because 
# the values during this time period are higher. Try using 
# reduced skill (dividing by standard devation).
plot_skill(df_crps, division_key='std', ylabel="RCRPS")

#This is perhpas not exactly what we want because we know this 
#data comes from the same location and the dataset has the same properties, 
#but we are splitting it up by mean value. Therefore try normalizing using mean
plot_skill(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")

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

def plot_hist(da_forecast, da_reanalysis):
    fig, ax = plt.subplots()
    for leadtime in da_forecast.leadtime:
        forecast = da_forecast.sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_reanalysis.reindex({'time': forecast.time}).dropna(dim='time')
        forecast = forecast.reindex({'time': observations.time})
        rank = get_rank(observations.values, forecast.values)
        ax.hist(rank, histtype='step', label=int(leadtime),
               bins=np.arange(0.5, max(rank)+1.5, 1), alpha=0.8)
    ax.legend(loc=9, title="Lead time (days)")
    ax.set_xlabel('Rank')
    ax.set_ylabel('Number')
    
for version in [2, 3]:
    forecast_list = [da_glofas_reforecast_interp]
    #if version == 2:
    #    forecast_list += [da_glofas_forecast]
    for da_forecast in forecast_list:
        observations =  da_glofas_reanalysis[version]
        plot_hist(da_forecast[version], observations)
        o = observations[observations > 110000]
        plot_hist(da_forecast[version], observations[observations > 100000])
    

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
        diff = (forecast - observations).values.flatten()
        append_dict =  {'leadtime': leadtime.values,  
                        'me': calc_me(observations, forecast),
                        'mpe': calc_mpe(observations, forecast)}
        for thresh in skill_thresh_vals:
            idx = observations > thresh
            append_dict.update({
                f'me_{thresh}': calc_me(observations[idx], forecast[:,idx]),
                f'mpe_{thresh}': calc_mpe(observations[idx], forecast[:,idx])
            })
        df_bias[version] = df_bias[version].append([
                               append_dict], ignore_index=True)
    
```

```python
fig, ax = plt.subplots()
for version, ls in zip([2,3], [':', '--']):    
    df = df_bias[version]
    for i, cname in enumerate(['mpe'] + [f'mpe_{thresh}' for thresh in skill_thresh_vals]):
        ax.plot(df['leadtime'], df[cname], ls=ls, c=f'C{i}')
    ax.plot([], [], ls=ls, c='k', label=f'version {version}')
# Add colours to legend
for i, subset in enumerate(['full year'] + [f'>{thresh}' for thresh in skill_thresh_vals]):
    ax.plot([], [], c=f'C{i}', label=subset)

    
ax.set_xlabel("Lead time (days)")
#ax.set_ylabel("Mean error [m$^3$ s$^{-1}$]")
ax.set_ylabel("Mean error (%)")

ax.legend()
```
