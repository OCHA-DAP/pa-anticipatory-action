### Exploring GloFAS forecast skill

This notebook evaluates GloFAS forecast skill at stations in Chikwawa and Nsanje. We are looking specifically at skill in predicting extreme streamflow values across various leadtimes.

```python
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, pearsonr
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy.interpolate import interp1d
from scipy.stats import rankdata
import os
from pathlib import Path
import sys
from datetime import timedelta

import read_in_data as rd
from importlib import reload
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import utils

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
SAVE_FIG = True
LEADTIMES = [5, 10, 15, 20, 25, 30]
stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}
```

### Read in the GloFAS data

```python
da_glofas_reanalysis = {}
da_glofas_reforecast = {}
da_glofas_reforecast_interp = {}
da_glofas_forecast = {}
da_glofas_forecast_summary = {}
da_glofas_reforecast_summary = {}

for station in STATIONS: 
    da_glofas_reanalysis[station] = utils.get_glofas_reanalysis('mwi', version=GLOFAS_VERSION)[station]
    da_glofas_reforecast[station] = utils.get_glofas_reforecast('mwi', LEADTIMES, interp=False, version=GLOFAS_VERSION)[station]
    da_glofas_reforecast_interp[station] = utils.get_glofas_reforecast('mwi', LEADTIMES, interp=True, version=GLOFAS_VERSION)[station]
    da_glofas_forecast[station] = utils.get_glofas_forecast('mwi', LEADTIMES, version=GLOFAS_VERSION)[station]
    da_glofas_forecast_summary[station] = utils.get_da_glofas_summary(da_glofas_forecast[station])
    da_glofas_reforecast_summary[station] = utils.get_da_glofas_summary(da_glofas_reforecast_interp[station])
```

### Read in the return period thresholds

```python
df_rps = pd.read_csv(EXPLORE_DIR / 'glofas_rps.csv')
rp_thresh = [2, 3, 5] # The rp thresholds that we're concerned about checking
skill_thresh_vals = {}
for station in stations_adm2.values():
    skill_thresh_vals[station] = [df_rps[df_rps['rp']==thresh][station].iloc[0] for thresh in rp_thresh]
```

### Calculate the skill across lead times

```python
def plot_skill(df_crps, division_key=None, add_line_from_website=False,
              ylabel="CRPS [m$^3$ s$^{-1}$]"):
    fig, ax = plt.subplots()
    for station, ls in zip(stations_adm2.values(), [':', '--']):
        df = df_crps[station].copy()
        for i, subset in enumerate([None] + skill_thresh_vals[station]):
            ykey = f'crps_{subset}' if subset is not None else 'crps'
            y = df[ykey]
            if division_key is not None:
                dkey = f'{division_key}_{subset}' if subset is not None else division_key
                y /= df[dkey]
            ax.plot(df['leadtime'], y, ls=ls, c=f'C{i}')
        ax.plot([], [], ls=ls, c='k', label=f'{station}')
    if add_line_from_website:
        ax.plot(df_skill[0], df_skill[1], ls='-', c='k', lw=0.5, label='from website')
    # Add colours to legend
    for i, subset in enumerate(['full year'] + [f'{thresh}-year return period' for thresh in rp_thresh]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    ax.legend()
```

```python
df_crps = {
    'Nsanje': pd.DataFrame(columns=['leadtime', 'crps']),
    'Chikwawa': pd.DataFrame(columns=['leadtime', 'crps'])
}

for code, station in stations_adm2.items(): 
    for leadtime in da_glofas_reforecast_interp[code].leadtime:
        forecast = da_glofas_reforecast_interp[code].sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_glofas_reanalysis[code].reindex({'time': forecast.time})
        # For all dates
        crps = xs.crps_ensemble(observations, forecast,member_dim='number')
        append_dict = {'leadtime': leadtime.values,
                                  'crps': crps.values,
                                   'std': observations.std().values,
                                   'mean': observations.mean().values,
                      }
        # For high values only
        for thresh in skill_thresh_vals[station]:
            idx = observations > thresh
            crps = xs.crps_ensemble(observations[idx], forecast[:, idx], member_dim='number')
            append_dict.update({
                f'crps_{thresh}': crps.values,
                f'std_{thresh}': observations[idx].std().values,
                f'mean_{thresh}': observations[idx].mean().values
            })
        df_crps[station] = df_crps[station].append([append_dict], ignore_index=True)
```

```python
plot_skill(df_crps)
if SAVE_FIG: plt.savefig(PLOT_DIR / 'skill.png')
plot_skill(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")
if SAVE_FIG: plt.savefig(PLOT_DIR / 'skill_ncrps.png')
```

### Investigate bias

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
    
for code, station in stations_adm2.items():
    forecast_list = [da_glofas_reforecast_interp[code]]
    for da_forecast in forecast_list:
        observations =  da_glofas_reanalysis[code]
        plot_hist(da_forecast, observations)
        if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_glofas_bias.png')
        # Select observations based on a 3-year return period threshold
        thresh = df_rps[df_rps['rp']==3][station].iloc[0]
        o = observations[observations > thresh]
        plot_hist(da_forecast, o)
        if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_glofas_bias_3rp.png')
```

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
    'Nsanje': pd.DataFrame(columns=['leadtime']),
    'Chikwawa': pd.DataFrame(columns=['leadtime']),
}   


for code, station in stations_adm2.items():
    da_forecast = da_glofas_reforecast_interp[code]
    da_reanalysis = da_glofas_reanalysis[code]
    for leadtime in da_forecast.leadtime:
        forecast = da_forecast.sel(
            leadtime=leadtime.values).dropna(dim='time')
        observations = da_reanalysis.reindex({'time': forecast.time})
        diff = (forecast - observations).values.flatten()
        append_dict =  {'leadtime': leadtime.values,  
                        'me': calc_me(observations, forecast),
                        'mpe': calc_mpe(observations, forecast)}
        for thresh in skill_thresh_vals[station]:
            idx = observations > thresh
            append_dict.update({
                f'me_{thresh}': calc_me(observations[idx], forecast[:,idx]),
                f'mpe_{thresh}': calc_mpe(observations[idx], forecast[:,idx])
            })
        df_bias[station] = df_bias[station].append([
                               append_dict], ignore_index=True)
```

```python
fig, ax = plt.subplots()
for station, ls in zip(stations_adm2.values(), [':', '--']):    
    df = df_bias[station]
    for i, cname in enumerate(['mpe'] + [f'mpe_{thresh}' for thresh in skill_thresh_vals[station]]):
        ax.plot(df['leadtime'], df[cname], ls=ls, c=f'C{i}')
    ax.plot([], [], ls=ls, c='k', label=f'{station}')
# Add colours to legend
for i, subset in enumerate(['full year'] + [f'{thresh}-year return period' for thresh in rp_thresh]):
    ax.plot([], [], c=f'C{i}', label=subset)

    
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("Mean error (%)")

ax.legend()
if SAVE_FIG: plt.savefig(PLOT_DIR / 'mpe.png')
```
