---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: Python [conda env:anact] *
    language: python
    name: conda-env-anact-py
---

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
import seaborn as sns

import read_in_data as rd
from importlib import reload
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
ADM2_SEL = ['Chikwawa', 'Nsanje']

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}
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

### Explore Floodscan data

```python
df_floodscan = rd.get_floodscan_processed()
df_floodscan = df_floodscan[df_floodscan['ADM2_EN'].isin(ADM2_SEL)]
df_floodscan = df_floodscan[['ADM2_EN','date', 'mean_cell', 'max_cell', 'min_cell']]
df_floodscan['date'] = pd.to_datetime(df_floodscan['date'])
```

```python
df_floodscan
```

```python
# Get rolling average to smooth out potential noise
df_floodscan['mean_cell_rolling'] = df_floodscan.groupby('ADM2_EN')['mean_cell'].transform(lambda x: x.rolling(5, 1).mean())
```

```python
for district in ADM2_SEL:
    fig, ax = plt.subplots()
    sns.lineplot(data=df_floodscan[df_floodscan['ADM2_EN']==district], x="date", y="mean_cell", lw=0.25, label='Original')
    sns.lineplot(data=df_floodscan[df_floodscan['ADM2_EN']==district], x="date", y="mean_cell_rolling", lw=0.25, label='5-day moving\navg')   
    ax.set_ylabel('Mean flooded fraction')
    ax.set_xlabel('Date')
    ax.set_title(f'Flooding in {district}, 1998-2020')
    ax.legend()
    plt.savefig(PLOT_DIR / f'{district}_floodscan_adm2.png')
```

### Understand relationship between GloFAS (streamflow) and Floodscan (% flooding in adm2)

```python
for key,value in stations_adm2.items():
    df_floodscan_sel = df_floodscan[df_floodscan['ADM2_EN']==value]
    df_glofas_sel = da_glofas_reanalysis[key].to_dataframe().reset_index()
    df_merged = pd.merge(df_floodscan_sel, df_glofas_sel, how='right', left_on='date', right_on='time').dropna()
    
    fig, ax = plt.subplots()
    ax.scatter(x=df_merged[key], y=df_merged["mean_cell_rolling"], alpha=0.2, s=1.5)
    ax.set_ylabel('Flooded fraction')
    ax.set_xlabel('Discharge [m$^3$ s$^{-1}$]')
    ax.set_title(f'Daily water discharge vs max\nflooded fraction in {value}')
    plt.savefig(PLOT_DIR / f'{value}_floodscan_mean_rolling_vs_glofas.png')
```
