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
import geopandas as gpd
from scipy.stats import norm, pearsonr
from scipy import stats
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy.interpolate import interp1d
import os
from pathlib import Path
import sys
import seaborn as sns
from functools import reduce
from datetime import timedelta

import read_in_data as rd
from importlib import reload
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.floodscan import floodscan

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
GLOFAS_VERSION = 3
SAVE_PLOT = False

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}
```

### Visualize Floodscan data


Take a look at an example of the Floodscan data. Here we're clipping it to a bounding box around the two districts of interest.

```python
fs = floodscan.Floodscan()
fs_raw = fs.read_raw_dataset()

ds_sel = (
    fs_raw.sel(time='2015-02-10')[['SFED_AREA']]
    .to_array()
    .rio.write_crs("EPSG:4326")
    .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
)

geometries = [
    {
        'type': 'Polygon',
        'coordinates': [
            [[34.1011082082,-17.12960312],
             [35.4722024947,-17.12960312],
             [35.4722024947,-15.5314191359],
             [34.1011082082,-15.5314191359],
             [34.1011082082,-17.12960312]
            ]]
    }
]
clipped = ds_sel.rio.clip(geometries)
clipped.plot()
if SAVE_PLOT: plt.savefig(PLOT_DIR / 'floodscan_overview.png')
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
df_floodscan_path = Path(os.environ['AA_DATA_DIR'])/'private'/'processed'/'mwi'/'floodscan'/'mwi_floodscan_stats_adm2.csv'
df_floodscan = pd.read_csv(df_floodscan_path)
df_floodscan = df_floodscan[['ADM2_EN','date', 'mean_cell', 'max_cell', 'min_cell']]
df_floodscan = df_floodscan[df_floodscan['ADM2_EN'].isin(stations_adm2.values())]
df_floodscan['date'] = pd.to_datetime(df_floodscan['date'])
```

```python
df_floodscan
```

Calculate the 5-day rolling average to smooth out potential noise from the Floodscan data.

```python
for station in stations_adm2.values():
    fig, ax = plt.subplots()
    df_floodscan_sel = df_floodscan[df_floodscan['ADM2_EN']==station]
    df_floodscan_sel['mean_cell_rolling'] = df_floodscan_sel['mean_cell'].transform(lambda x: x.rolling(5, 1).mean())
    sns.lineplot(data=df_floodscan_sel, x="date", y="mean_cell", lw=0.25, label='Original')
    sns.lineplot(data=df_floodscan_sel, x="date", y="mean_cell_rolling", lw=0.25, label='5-day moving\navg')   
    ax.set_ylabel('Mean flooded fraction')
    ax.set_xlabel('Date')
    ax.set_title(f'Flooding in {station}, 1998-2020')
    ax.legend()
    if SAVE_PLOT: plt.savefig(PLOT_DIR / f'{station}_flooding_fraction.png')
```

### Clean data


TODO: Remove long-term trend!


### Get 'ground-truth' flood events

```python
def get_groups_above_threshold(observations, threshold):
    return np.where(np.diff(np.hstack(([False],
                                           observations > threshold,
                                           [False]))))[0].reshape(-1, 2)

# Assign an eventID to each flood 
# ie. consecutive dates in a dataframe filtered to keep only outliers in flood fraction
def get_groups_consec_dates(df):
    dt = df['date']
    day = pd.Timedelta('1d')
    breaks = dt.diff() != day
    groups = breaks.cumsum()
    groups = groups.reset_index()
    groups.columns = ['index', 'eventID']
    df_out = df.merge(groups, left_index=True, right_on='index')
    return df_out

# Get basic summary statistics for each flood event
def get_flood_summary(df):
    s1 = df.groupby('eventID')['date'].min().reset_index().rename(columns={'date': 'start_date'})
    s2 = df.groupby('eventID')['date'].max().reset_index().rename(columns={'date': 'end_date'})
    s3 = df.groupby('eventID')['date'].count().reset_index().rename(columns={'date': 'num_days'})
    s4 = df.groupby('eventID')['mean_cell_rolling'].max().reset_index().rename(columns={'mean_cell_rolling': 'max_flood_frac'})
    dfs = [s1, s2, s3, s4]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['eventID'],
                                            how='outer'), dfs)
    return df_merged
```

Find the dates that are significant outliers (std>3) in mean flooding fraction across all pixels in the area of interest. We'll consider each group of consecutive dates to be a significant flooding event.

```python
flooding = {}

for station in stations_adm2.values():
    df_floodscan_sel = df_floodscan[df_floodscan['ADM2_EN']==station]
    df_floodscan_sel['mean_cell_rolling'] = df_floodscan_sel['mean_cell'].transform(lambda x: x.rolling(5, 1).mean())
    df_floods_summary = df_floodscan_sel[(np.abs(stats.zscore(df_floodscan_sel['mean_cell_rolling'])) >= 3)]
    df_floods_summary = get_groups_consec_dates(df_floods_summary)
    df_floods_summary = get_flood_summary(df_floods_summary)
    
    # In the cases where two flood events are separated by less than 1 month, 
    # we'll merge them together to be considered as a single event. 
    for i in range(1, len(df_floods_summary.index)-1):
        start_buffer = pd.to_datetime(df_floods_summary['start_date'].iloc[i,]) - timedelta(days=30)
        end_buffer = pd.to_datetime(df_floods_summary['end_date'].iloc[i-1,]) + timedelta(days=30)
        
        if start_buffer < end_buffer:
            df_floods_summary['end_date'].iloc[i-1,] = df_floods_summary['end_date'].iloc[i,]
            df_floods_summary['num_days'].iloc[i-1] = (df_floods_summary['end_date'][i-1] - df_floods_summary['start_date'][i-1]).days

    # Now we need to drop the rows with the same end date 
    # and keep the one with the longer duration
    df_summary_clean = (
        df_floods_summary
        .sort_values('num_days')
        .groupby('end_date')
        .tail(1)
        .sort_values('start_date')
        .reset_index(drop=True)
    )
    
    flooding[station] = df_summary_clean

    df_summary_clean.to_csv(EXPLORE_DIR / f'{station}_floodscan_event_summary.csv')
```

```python
flooding['Nsanje']
```

```python
flooding['Chikwawa']
```

### Understand relationship between GloFAS (streamflow) and Floodscan (% flooding in adm2)

```python
for station in STATIONS:
    df_glofas_sel = da_glofas_reanalysis[station].to_dataframe().reset_index()
    df_merged = pd.merge(df_floodscan, df_glofas_sel, how='right', left_on='date', right_on='time').dropna()
    
    fig, ax = plt.subplots()
    ax.scatter(x=df_merged[station], y=df_merged["mean_cell_rolling"], alpha=0.2, s=1.5)
    ax.set_ylabel('Flooded fraction')
    ax.set_xlabel('Discharge [m$^3$ s$^{-1}$]')
    ax.set_title(f'Daily water discharge vs mean\nflooded fraction in {station}')
    if SAVE_PLOT: plt.savefig(PLOT_DIR / f'{station}_floodscan_mean_rolling_vs_glofas.png')
```
