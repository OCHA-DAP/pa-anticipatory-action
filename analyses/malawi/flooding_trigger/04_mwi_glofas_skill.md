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
from matplotlib.ticker import MaxNLocator, ScalarFormatter

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import utils

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
EXPLORE_DIR = PRIVATE_DIR / 'exploration' / 'mwi' / 'flooding'

SAVE_FIG = True
LEADTIMES = [x + 1 for x in range(10)]

stations_adm2 = {
    'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}
COUNTRY_ISO3 = 'mwi'
```

Read in the processed GloFAS data and calculate the return periods. We can also plot the river discharge thresholds for each return period level.

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=False)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

```python
rp_label = [str(int(x)) for x in df_return_period.index]
rp_label[0] = '1.5'
for code, station in stations_adm2.items():
    fig, ax = plt.subplots()
    ax.set_title(station)
    rp = df_return_period[code]
    ax.plot(rp_label, rp, 'o-', label=station)
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')
    ax.legend()
    
    if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_return_periods.png')
```

Look into forecast skill by calculating the CRPS. We'll first do this by looking at all discharge values, and then recalculate looking specifically at extreme discharge values (eg. at the 3-year return period level). 

```python
def plot_crps(df_crps, title_suffix=None, ylog=False):
    for code, station in stations_adm2.items():
        fig, ax = plt.subplots()
        crps = df_crps[code]
        ax.plot(crps.index, crps, label=station)
        ax.legend()
        title = station
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

if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_ncrps_all.png')
```

```python
rp = 3
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                         normalization="mean", 
                         thresh=df_return_period.loc[rp].to_dict())
plot_crps(df_crps * 100, title_suffix=f" -- values > RP 1 in {rp} y", ylog=False)

if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_ncrps_{rp}_rp.png')
```

We'll also look at the forecast bias, again specifically at extreme water discharge levels. 

```python
# Rank histogram
def plot_hist(da_observations, da_forecast, station_name, rp=None, leadtimes=None):
    if leadtimes is None:
        leadtime = da_forecast.leadtime.values
    fig, ax = plt.subplots()
    for leadtime in leadtimes:
        observations, forecast = utils.get_same_obs_and_forecast(da_observations, da_forecast, leadtime)
        rank = utils.get_rank(observations.values, forecast.values)
        ax.hist(rank, histtype='step', label=int(leadtime),
               bins=np.arange(0.5, max(rank)+1.5, 1), alpha=0.8)
    ax.legend(loc=9, title="Lead time (days)")
    ax.set_xlabel('Rank')
    ax.set_ylabel('Number')
    title = station_name
    if rp is not None:
        title += f': > 1 in {rp} y'
    ax.set_title(title)

rp = 3
for code, station in stations_adm2.items():
    da_observations =  ds_glofas_reanalysis[code]
    da_forecast = ds_glofas_reforecast[code]
    plot_hist(da_observations, da_forecast, station, leadtimes=LEADTIMES)
    
    if SAVE_FIG: 
        plt.savefig(PLOT_DIR / f'{station}_rank_hist_all.png')
    
    rp_val = df_return_period.loc[rp, code]
    o = da_observations[da_observations > rp_val]
    # Needs at least about 50 vals to work, not sure why
    if len(o) > 50:
        plot_hist(o, da_forecast, station, leadtimes=LEADTIMES, rp=rp)
        
        if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_rank_hist_{rp}_rp.png')
```

```python
rp = 2
for code, station in stations_adm2.items():
    fig, ax = plt.subplots()
    
    da_observations =  ds_glofas_reanalysis[code]
    rp_val = df_return_period.loc[rp, code]
    da_observations_ev = da_observations[da_observations > rp_val]
    da_forecast = ds_glofas_reforecast[code]
    mpe = np.empty(len(da_forecast.leadtime))
    mpe_ev = np.empty(len(da_forecast.leadtime))
    for ilt, leadtime in enumerate(da_forecast.leadtime):
        observations, forecast = utils.get_same_obs_and_forecast(da_observations, da_forecast, leadtime)
        mpe[ilt] = utils.calc_mpe(observations, forecast)
        observations_ev, forecast_ev = utils.get_same_obs_and_forecast(da_observations_ev, da_forecast, leadtime)
        mpe_ev[ilt] = utils.calc_mpe(observations_ev, forecast_ev)
    ax.plot(da_forecast.leadtime, mpe, label='All values')
    ax.plot(da_forecast.leadtime, mpe_ev, label=f'RP > 1 in {rp} y')
    ax.set_ylim(-50, 10)
    ax.axhline(y=0, c='k', ls=':')
    ax.legend()
    ax.grid()
    ax.set_title(station)
    ax.set_xlabel('Leadtime [y]')
    ax.set_ylabel('% bias')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if SAVE_FIG: plt.savefig(PLOT_DIR / f'{station}_perc_bias.png')
```
