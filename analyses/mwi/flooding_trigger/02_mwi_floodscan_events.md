### Estimating flood events from Floodscan data

This notebook analyzes pre-processed data on average daily surface water coverage for Chikwawa and Nsanje in Malawi. A dataset of estimated historical flood events is output.

```python
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rio
from scipy import stats
from functools import reduce
import os
from pathlib import Path
import sys
from datetime import timedelta

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.floodscan import floodscan

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
EXPLORE_DIR = PRIVATE_DIR / 'exploration' / 'mwi' / 'flooding'

SAVE_PLOT = False
SAVE_DATA = False

stations_adm2 = {
    'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}
```

Take a look at an example of the Floodscan data to get a general sense of the scale. Here we're clipping it to a bounding box around the two districts of interest.

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

Read in the pre-processed Floodscan data and filter for our variables and districts of interest.

```python
df_floodscan_path = Path(os.environ['AA_DATA_DIR'])/'private'/'processed'/'mwi'/'floodscan'/'mwi_floodscan_stats_adm2.csv'
df_floodscan = pd.read_csv(df_floodscan_path)
df_floodscan = df_floodscan[['ADM2_EN','date', 'mean_cell', 'max_cell', 'min_cell']]
df_floodscan = df_floodscan[df_floodscan['ADM2_EN'].isin(stations_adm2.values())]
df_floodscan['date'] = pd.to_datetime(df_floodscan['date'])
```

Calculate the 5-day rolling average to smooth out potential noise and create simple plots to show the changes in flooding fraction over time.

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

Remove dates outside of the rainy season. We're loosely defining the rainy season as being between October - April (inclusive). 

```python
df_floodscan['month'] = pd.DatetimeIndex(df_floodscan['date']).month
df_floodscan_rainy = df_floodscan.loc[(df_floodscan['month'] >= 10) | (df_floodscan['month'] <= 4)]
```

Now with this cleaned up data we can identify consecutive dates of significantly above average (>3 standard deviations) surface water coverage. We'll consider these to be flood events. This threshold is set with the intent to capture events that are significant outliers, but could be refined/validated with future work.

```python
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

# Merge overlapping flood events
# Each row in the input df should be an event
# With start and end date columns: ['start_date'] and ['end_date']
def merge_events(df):
    df['flood_id'] = 0
    f_id = 1
    
    # Loop through all of the events and tag the ones that are part of an overlap
    for i in range(1, len(df.index)):        
        start = df['start_date'].iloc[i,]
        end = df['end_date'].iloc[i-1,]
        if start < end:
            df.loc[i, 'flood_id'] = f_id
            df.loc[i-1, 'flood_id'] = f_id
        else:           
            df.loc[i-1, 'flood_id'] = f_id
            f_id += 1
    
    # Now for each event, extract the min start data and max end date
    df_start = df.groupby('flood_id')['start_date'].min().to_frame().reset_index()
    df_end = df.groupby('flood_id')['end_date'].max().to_frame().reset_index()
    
    df_events = df_start.merge(df_end, on='flood_id').sort_values(by='start_date')
    return df_events
```

```python
outlier_thresh = 3

for station in stations_adm2.values():
    df_floodscan_sel = df_floodscan_rainy[df_floodscan_rainy['ADM2_EN']==station]
    df_floodscan_sel['mean_cell_rolling'] = df_floodscan_sel['mean_cell'].transform(lambda x: x.rolling(5, 1).mean())
    df_floods_summary = df_floodscan_sel[(np.abs(stats.zscore(df_floodscan_sel['mean_cell_rolling'])) >= outlier_thresh)]
    df_floods_summary = get_groups_consec_dates(df_floods_summary)
    df_floods_summary = get_flood_summary(df_floods_summary)
    
    df_summary_clean = merge_events(df_floods_summary)
    if SAVE_DATA: df_summary_clean.to_csv(EXPLORE_DIR / f'{station}_floodscan_event_summary.csv', index=False)   
```
