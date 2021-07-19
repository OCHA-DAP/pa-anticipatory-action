```python
from pathlib import Path
import os
from importlib import reload
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
# chdir because otherwise can't import get_glofas_data
# should only run this once
os.chdir(path_mod)

from src.indicators.flooding.glofas import utils, glofas
from src.utils_general.statistics import calc_mpe
import src.nepal.get_glofas_data as ggd

reload(utils)
```

```python
mpl.rcParams['figure.dpi'] = 200


COUNTRY_ISO3 = 'npl'
STATIONS = {
    'Koshi': ['Chatara_v3', 'Simle_v3', 'Majhitar_v3', 'Kampughat_v3'],
    'Karnali': ['Chisapani_v3', 'Asaraghat_v3', 'Dipayal_v3', 'Samajhighat_v3'],
    'Rapti': ['Kusum_v3'],
    'Bagmati': ['Rai_goan_v3'],
    'Babai': ['Chepang_v3']
}
STATIONS_BY_MAJOR_BASIN = {
    'Koshi': ['Chatara_v3', 'Simle_v3', 'Majhitar_v3', 'Kampughat_v3', 'Rai_goan_v3'],
    'Karnali': ['Chisapani_v3', 'Asaraghat_v3', 'Dipayal_v3', 'Samajhighat_v3', 'Kusum_v3', 'Chepang_v3'],
}
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=ggd.LEADTIMES,
    interp=False
)
```

## Return period

```python
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

```python
rp_label = [str(int(x)) for x in df_return_period.index]
rp_label[0] = '1.5'
for basin, stations in STATIONS.items():
    fig, ax = plt.subplots()
    ax.set_title(basin)
    for station in stations:
        rp = df_return_period[station]
        ax.plot(rp_label, rp, 'o-', label=station)
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')
    ax.legend()
```

## Skill

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
rp = 1.5
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                         normalization="mean", 
                         thresh=df_return_period.loc[rp].to_dict())
plot_crps(df_crps * 100, title_suffix=f" -- values > RP 1 in {rp} y", ylog=True)
```

```python

```

## Bias

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

rp = 1.5
leadtimes = [5, 10, 15, 20]
for stations in STATIONS.values():
    for station in stations:
        da_observations =  ds_glofas_reanalysis[station]
        da_forecast = ds_glofas_reforecast[station]
        plot_hist(da_observations, da_forecast, station, leadtimes=[5, 10, 15, 20])
        rp_val = df_return_period.loc[rp, station]
        o = da_observations[da_observations > rp_val]
        # Needs at least about 50 vals to work, not sure why
        if len(o) > 50:
            plot_hist(o, da_forecast, station, leadtimes=leadtimes, rp=rp)

```

### Mean percent error


```python
rp = 1.5
for basin, stations in STATIONS_BY_MAJOR_BASIN.items():
    fig, ax = plt.subplots()
    for istation, station in enumerate(stations):
        da_observations =  ds_glofas_reanalysis[station]
        rp_val = df_return_period.loc[rp, station]
        da_observations_ev = da_observations[da_observations > rp_val]
        da_forecast = ds_glofas_reforecast[station]
        mpe = np.empty(len(da_forecast.leadtime))
        mpe_ev = np.empty(len(da_forecast.leadtime))
        for ilt, leadtime in enumerate(da_forecast.leadtime):
            observations, forecast = utils.get_same_obs_and_forecast(da_observations, da_forecast, leadtime)
            mean_forecast = forecast.mean(axis=0)
            mpe[ilt] = calc_mpe(observations, mean_forecast)
            observations_ev, forecast_ev = utils.get_same_obs_and_forecast(da_observations_ev, da_forecast, leadtime)
            mean_forecast_ev = forecast_ev.mean(axis=0)
            mpe_ev[ilt] = calc_mpe(observations_ev, mean_forecast_ev)
        ax.plot(da_forecast.leadtime, mpe, label=station, c=f'C{istation}')
        ax.plot(da_forecast.leadtime, mpe_ev, '--', c=f'C{istation}')
    ax.plot([], [], 'k-', label='All values')
    ax.plot([], [], 'k--', label=f'RP > 1 in {rp} y')
    ax.set_ylim(-50, 10)
    ax.axhline(y=0, c='k', ls=':')
    ax.legend()
    ax.grid()
    ax.set_title(basin)
    ax.set_xlabel('Leadtime [y]')
    ax.set_ylabel('% bias')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
```
