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
import os
from pathlib import Path
import sys

import read_in_data as rd
from importlib import reload
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config

config = Config()
mpl.rcParams['figure.dpi'] = 300

GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
```

### Read in GloFAS data

```python
da_glofas_reanalysis = {}
da_glofas_reforecast = {}
da_glofas_forecast = {}
da_glofas_forecast_summary = {}
da_glofas_reforecast_summary = {}

for station in STATIONS: 
    da_glofas_reanalysis[station] = rd.get_glofas_reanalysis(version=GLOFAS_VERSION, station=station)
    da_glofas_reforecast[station] = rd.get_glofas_reforecast(version=GLOFAS_VERSION, station=station)
    da_glofas_forecast[station] = rd.get_glofas_forecast(version=GLOFAS_VERSION, station=station)
    da_glofas_forecast_summary[station] = rd.get_da_glofas_summary(da_glofas_forecast[station])
    da_glofas_reforecast_summary[station] = rd.get_da_glofas_summary(da_glofas_reforecast[station])
```

### Read in the baseline impact data

```python
df_mvac_flood_ta = pd.read_csv(os.path.join(config.DATA_PRIVATE_DIR, 'processed', 'malawi', 'mvac_dodma_flood_ta.csv'))
```

### Calculate the return period

```python
def get_return_period_function(observations, station):
    df_rp = (observations.to_dataframe()
                 .rename(columns={station: 'discharge'})
                 .resample(rule='A', kind='period')
                 .max() 
                 .sort_values(by='discharge', ascending=False)
                )
    df_rp["year"] = df_rp.index.year
     
    n = len(df_rp)
    df_rp['rank'] = np.arange(n) + 1
    df_rp['exceedance_probability'] = df_rp['rank'] / (n+1)
    df_rp['rp'] = 1 / df_rp['exceedance_probability']
    return interp1d(df_rp['rp'], df_rp['discharge'])

rp_dict = {}

for station in STATIONS:
    f_rp = get_return_period_function(da_glofas_reanalysis[station], station)
    rp_dict[station] = {}
    for year in [1.5, 2, 3, 4, 5, 10, 20]:
        val = 10*np.round(f_rp(year) / 10)
        rp_dict[station][year] = val
```

```python
pd.DataFrame(rp_dict)
```

### Overview of historical discharge

```python
# Return periods to focus on, with display colours
rps = {
    3: '#32a852',
    5: '#9c2788'
}

for station in STATIONS: 
    da_plt = da_glofas_reanalysis[station].sel(time=slice('1999-01-01','2020-12-31'))
    df_flood = df_mvac_flood_ta[df_mvac_flood_ta['name']==station]

    fig, ax = plt.subplots()
    da_plt.plot(x='time', add_legend=True, ax=ax)
    ax.set_title(f'Historical streamflow at {station}')
    ax.set_xlabel("Date")
    ax.set_ylabel('Discharge [m$^3$ s$^{-1}$]')
    ax.axvspan(np.datetime64('2010-01-01'), np.datetime64('2019-12-31'), alpha=0.2, color='gray', label='Flooding monitoring')

    for year in df_flood.Year.unique():
        ax.axvspan(np.datetime64(f'{str(year)}-01-01'), np.datetime64(f'{str(year)}-12-31'), color='#ffb2a6', label='Flooding in TA')
    
    for key, value in rps.items():
        ax.axhline(rp_dict[station][key], 0, 1, color=value, label=f'{str(key)} return period')
        
    ax.legend()
    
    plt.savefig(f'C:/Users/Hannah/Desktop/mwi_plots/{station}_historical.png')
```

### Checking out the skill

```python
def is_rainy_season(month):
    # Oct to April
    return (month >= 10) | (month <= 4)

def is_dry_season(month):
    # May to Sept
    return (month < 10) & (month > 4)

def get_skill(da_glofas_reforecast, da_glofas_reanalysis):
    
    df_crps = pd.DataFrame(columns=['leadtime', 'crps'])

    for leadtime in da_glofas_reforecast.leadtime:
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
        # Total summary
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
    return df_crps
```

```python
def plot_skill(df_crps, title, division_key=None, add_line_from_website=False,
              ylabel="CRPS [m$^3$ s$^{-1}$]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([None, 'rainy', 'dry']):
        ykey = f'crps_{subset}' if subset is not None else 'crps'
        y = df[ykey]
        if division_key is not None:
            dkey = f'{division_key}_{subset}' if subset is not None else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls='-', c=f'C{i}')
    #ax.plot([], [], ls=ls, c='k', label=f'version {version}')
    if add_line_from_website:
        ax.plot(df_skill[0], df_skill[1], ls='-', c='k', lw=0.5, label='from website')
    # Add colours to legend
    for i, subset in enumerate(['full year', 'rainy', 'dry']):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title(title)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    ax.legend()
    
```

```python
df_crps = dict()

for station in STATIONS:
    df_crps[station] = get_skill(da_glofas_reforecast[station], da_glofas_reanalysis[station])
    plot_skill(df_crps[station], f"GloFAS forecast skill at {station}:\n 1999-2019 reforecast")
    #plt.savefig(f'C:/Users/Hannah/Desktop/mwi_plots/{station}_crps.png')
    plot_skill(df_crps[station], f"GloFAS forecast skill at {station}:\n 1999-2019 reforecast", division_key='std', ylabel="RCRPS")
    #plt.savefig(f'C:/Users/Hannah/Desktop/mwi_plots/{station}_rcrps.png')
    plot_skill(df_crps[station], f"GloFAS forecast skill at {station}:\n 1999-2019 reforecast", division_key='mean', ylabel="NCRPS (CRPS / mean)")
    #plt.savefig(f'C:/Users/Hannah/Desktop/mwi_plots/{station}_ncrps.png')
```
