# High water-level flood events

We use the water level data to define flood events as when the water level goes 
above the danger / warning level as defined by DHM. We want to know how
often these events correspond to a GloFAS RP exceedance. 

```python
import os
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils, glofas

pd.options.mode.chained_assignment = None  # default='warn'
mpl.rcParams['figure.dpi'] = 200
```

```python

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
DHM_DIR = DATA_DIR / 'private/exploration/npl/dhm'
WL_PROCESSED_DIR = DHM_DIR / 'processed'
WL_OUTPUT_FILENAME = 'waterl_level_procssed.csv'
STATION_INFO_FILENAME = 'npl_dhm_station_info.xlsx'

GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"

COUNTRY_ISO3 = 'npl'
DURATION = 1

STATIONS = [
    'Chatara',
    'Chisapani',
]

LEVEL_TYPES = ['warning', 'danger']
RP_LIST = [1.5, 2, 5]
```

```python
df_station_info = pd.read_excel(DHM_DIR / STATION_INFO_FILENAME, index_col='station_name')
df_wl = pd.read_csv(WL_PROCESSED_DIR / WL_OUTPUT_FILENAME, index_col='date')


ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
df_return_period =  pd.read_excel(GLOFAS_RP_FILENAME, index_col='rp')

```

### Create dataframe with both water level and river discharge

```python
df_station_dict = {}
for station in STATIONS:
    wl = df_wl[[station]]
    rd = ds_glofas_reanalysis[station].to_dataframe().drop(columns=['step', 'surface', 'valid_time'])
    data = (pd.merge(wl, rd, 
                     how='inner', 
                     left_index=True, 
                     right_index=True, 
                     suffixes=['_wl', '_rd'])
            .dropna()
            .rename(columns={f"{station}_wl": "water_level",
                    f"{station}_rd": "river_discharge"})
           )
    # Fill in the gaps so that the group finding works
    data = data.reindex(pd.date_range(data.index.min(), data.index.max()))
    # Get the water level events
    for level_type in LEVEL_TYPES:
        level_val = df_station_info.at[station, f'{level_type}_level']
        events = utils.get_groups_above_threshold(data['water_level'], level_val, min_duration=DURATION)
        event_start_indices = [event[0] for event in events]
        data[f"event_{level_type}"] = False
        data[f"event_{level_type}"].iloc[event_start_indices] = True
    # Get river discharge events
    for rp in RP_LIST:
        rp_val = df_return_period.loc[rp, station]
        events = utils.get_groups_above_threshold(data['river_discharge'], rp_val, min_duration=DURATION)
        event_start_indices = [event[0] for event in events]
        data[f"event_{rp}"] = False
        data[f"event_{rp}"].iloc[event_start_indices] = True
    df_station_dict[station] = data
    fig, ax = plt.subplots()
    ax.plot(data.water_level, data.river_discharge, '.')
    idx = data['event_warning'] == True
    ax.plot(data.water_level[idx], data.river_discharge[idx], 'xr')
```

## Compare events

Want to check how many years with events for a given level type or RP
to make sure we're comparing similar types of events

```python
for station in STATIONS:
    print(station)
    df_station = df_station_dict[station]
    df_station = df_station.groupby(df_station.index.year).sum() > 0
    print(f"Total years: {len(df_station)}")
    for level_type in LEVEL_TYPES:
        n = df_station[f"event_{level_type}"].sum()
        print(f"{level_type} level: {n}")
    for rp in RP_LIST:
        n = df_station[f"event_{rp}"].sum()
        print(f"1 in {rp} y: {n}")
    print('\n')
```

Unfortuantely there is some mismatch. 
For Chatara, it seems that warning level corresponds to 1 in 1.5 year, and danger level to 1 in 2 year.
For chisapani, warning level is closer to 1 in 2 year (probably more like 1 in 3 year) and danger level to 1 in 5 year. 

```python
# Settle on RP and level type
rp = 2
event_level_type = 'danger'
```

```python
days_before_buffer = 5 # How many days the true event can occur before the GloFAS event
days_after_buffer = 30 # How many days the tru event can occur after the GloFAS event

df_station_stats = pd.DataFrame(columns=['station', 'TP', 'FP', 'FN'])

for station in STATIONS:
    df_station = df_station_dict[station]
    df_true_events = df_station[df_station[f"event_{event_level_type}"]]
    df_true_events['detections'] = 0
    TP = 0
    FP = 0
    glofas_events = df_station[df_station[f"event_{rp}"]].index
    print(f"{station}")
    print(f"True events: {len(df_true_events)}")
    print(f"Glofas events: {len(glofas_events)}")
    for glofas_event in glofas_events:
        # Check if any events are around that date
        days_offset = (df_true_events.index - glofas_event) /  np.timedelta64(1, 'D')
        detected = (days_offset > -1 * days_before_buffer) & (days_offset < days_after_buffer)
        df_true_events.loc[detected, 'detections'] += 1
        # If there were any detections, it's  a TP. Otherwise a FP
        if sum(detected):
            TP += 1
        else:
            FP += 1
    df_station_stats = df_station_stats.append({
        'station': station,
        'TP': TP,
        'FP': FP,
        'FN': len(df_true_events[df_true_events['detections'] == 0])
    }, ignore_index=True)
df_station_stats
```

# 


## Make plots for presentation

```python
def plot_arrow(ax, x, y, c):
    ax.annotate(" ", 
                     xy=(x, y+y*0.3),
                     xytext=(x, y+y*0.5),
                    arrowprops=dict(facecolor=c, shrink=0.05, headlength=3,
                               width=1, headwidth=3, lw=0.5))
    
for station in STATIONS:
    df = df_station_dict[station]
    fig, (ax2, ax1) = plt.subplots(2)
    fig.suptitle(station)
    # Water level
    
    ax1.plot(df['water_level'])
    level_val = df_station_info.at[station, f'{level_type}_level']
    idx = df['water_level'] >= level_val
    #ax1.plot(df.loc[idx, 'water_level'], '.', c='C3')
    ax1.axhline(level_val, c='C3')
    for event in list(df.index[df[f'event_{event_level_type}']]):
        plot_arrow(ax1, event, level_val, 'C3')
    
    ax1.set_ylabel('DHM water level [m]')
    ax1.set_ylim(None, level_val+ 0.7 * level_val)
   
    # River discharge
    ax2.plot(df['river_discharge'], c='C2')
    rp_val = df_return_period.loc[rp, station]
    idx = df['river_discharge'] >= rp_val
    #ax2.plot(df.loc[idx, 'river_discharge'], '.', c='C1')
    ax2.axhline(rp_val, c='C1')
    for event in list(df.index[df[f'event_{rp}']]):
        plot_arrow(ax2, event, rp_val, 'C1')
    
    ax2.set_ylabel('GloFAS river discharge [m$^3$ s$^{-1}$]')
    ax2.set_ylim(None, rp_val + 0.7 * rp_val)

events
```

```python

```
