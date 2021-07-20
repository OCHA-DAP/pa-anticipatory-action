# Event timeline

Go through each date and figure out when an activation would have occured. Also, do a more thorough
historical analysis, taking into account all forecast lead times.


```python
import sys
from pathlib import Path
import os
from importlib import reload

import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
reload(utils)

mpl.rcParams['figure.dpi'] = 200
```

```python
COUNTRY_ISO3 = 'npl'
MAIN_RP = 2
RP_LIST = [1.5, 2]
FORECAST_PERCENTILE_LIST = [50, 25]
# Use "_v3" for the GloFAS model v3 locs, or empty string for the original v2 ones
VERSION_LOC = "_v3" 
STATIONS = [
    'Chatara',
    'Chisapani',
]
DURATION = 1

LEVEL_TYPES = ['warning', 'danger']

LEADTIMES = [x+1 for x in range(7)]
LEADTIMES_BY_TRIGGER = {
    "action": [1, 2, 3],
    "readiness": [4, 5, 6, 7]
}

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
DHM_DIR = DATA_DIR / 'private/exploration/npl/dhm'
WL_PROCESSED_DIR = DHM_DIR / 'processed'
WL_OUTPUT_FILENAME = 'waterl_level_procssed.csv'
STATION_INFO_FILENAME = 'npl_dhm_station_info.xlsx'

GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"
```

```python
df_station_info = pd.read_excel(DHM_DIR / STATION_INFO_FILENAME, index_col='station_name')
df_wl = pd.read_csv(WL_PROCESSED_DIR / WL_OUTPUT_FILENAME, index_col='date', parse_dates=True)

ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=True, shift_dates=False
)
ds_glofas_forecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)
df_return_period =  pd.read_excel(GLOFAS_RP_FILENAME, index_col='rp')
pd.options.mode.chained_assignment = None  # default='warn'
```

### Create dataframe with both river discharge and water level

```python
df_station_dict = {}
for station in STATIONS:
    wl = df_wl[[station]]
    rd = (ds_glofas_reanalysis[station + VERSION_LOC]
              .to_dataframe()
              .drop(columns=['step', 'surface', 'valid_time'])
              .rename(columns={f"{station+VERSION_LOC}": station}))
    data = (pd.merge(wl, rd, 
                     how='right', 
                     left_index=True, 
                     right_index=True, 
                     suffixes=['_wl', '_rd'])
            #.dropna()
            .rename(columns={f"{station}_wl": "water_level",
                    f"{station}_rd": "river_discharge"})
           )
    # Fill in the gaps so that the group finding works - not sure this is needed
    data = data.reindex(pd.date_range(data.index.min(), data.index.max()))
    # Add in the forecast data
    for leadtime in LEADTIMES:
        for percentile in FORECAST_PERCENTILE_LIST:
            forecast = (ds_glofas_forecast_summary[station + VERSION_LOC]
                    .sel(leadtime=leadtime, percentile=percentile)
                    .to_dataframe()
                    .drop(columns=['surface', 'leadtime', 'percentile'])
                    .rename(columns={f"{station+VERSION_LOC}": station}))
            data = (pd.merge(data, forecast,
                        how='left',
                        left_index=True,
                        right_index=True,
                        )
               .rename(columns={station: f"forecast_lt{leadtime}_p{percentile}"}))
    # Keep only the forecast date range
    data = data.dropna(subset=["forecast_lt1_p50"])
    # Get the water level events
    level_type = "danger"
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
        data[f"event_rp{rp}"] = False
        data[f"event_rp{rp}"].iloc[event_start_indices] = True
    # Go through each date and add activations
    for rp in RP_LIST:
        rp_val = df_return_period.loc[rp, station]
        for percentile in FORECAST_PERCENTILE_LIST:
            for event_type, leadtimes in LEADTIMES_BY_TRIGGER.items():
                data[f'event_{event_type}_rp{rp}_p{percentile}'] = data.apply(lambda row: 
                                          (row[[f'forecast_lt{leadtime}_p{percentile}' for leadtime in leadtimes]] >= rp_val).any(),
                                          axis=1)
    df_station_dict[station] = data
data
```

### Get TP, FP, and FN

```python
# Find TP, FP and FN

def get_consecutive_groups(x: pd.Series, n=2):
    # Find indices whre False / True change. Skip the first entry of the array because it always changes there
    l = np.where(x.shift() != x)[0][1:]
    # Convert array to pairs
    return [l[i:i+n] for i in range(0, len(l), n)]


def get_station_stats(df_station_dict, event_var, rp=MAIN_RP):

    days_before_buffer = 0 # Event can occur at the earliest on the same day as the trigger
    days_after_buffer = 30 # How many days the true event can occur after the GloFAS event
    df_station_stats = pd.DataFrame(columns=['station', 'TP', 'FP', 'FN', 'event_type', 'percentile'])

    for station in STATIONS:
        df_station = df_station_dict[station]
        if event_var == 'event_danger':
            df_station = df_station.dropna(subset=['water_level'])
        df_true_events = df_station[df_station[event_var]][[event_var]]
        rp_val = df_return_period.loc[rp, station]
        for event_type, leadtimes in LEADTIMES_BY_TRIGGER.items():
            for percentile in FORECAST_PERCENTILE_LIST:
                glofas_event_indices = get_consecutive_groups(df_station[f"event_{event_type}_rp{rp}_p{percentile}"])
                df_true_events['detections'] = 0
                TP = 0
                FP = 0    
                for (glofas_event_start_index, glofas_event_end_index) in glofas_event_indices:
                    glofas_event = df_station.index[glofas_event_start_index]
                    # Check the maximum lead time 
                    max_leadtime = -1
                    for leadtime in leadtimes:
                        if df_station.loc[glofas_event, f'forecast_lt{leadtime}_p{percentile}'] >= rp_val:
                            max_leadtime = leadtime
                    if max_leadtime == -1:
                        print('Something went wrong!')
                    # Check if any events are around that date
                    days_offset = (df_true_events.index - glofas_event) /  np.timedelta64(1, 'D')
                    # Add the max leadtime to the days after buffer
                    detected = (days_offset >= -1 * days_before_buffer) & (days_offset <= days_after_buffer + max_leadtime)
                    df_true_events.loc[detected, 'detections'] += 1
                    # If there were any detections, it's  a TP. Otherwise a FP
                    if not sum(detected):
                        FP += 1
                df_station_stats = df_station_stats.append({
                    'station': station,
                    'TP': len(df_true_events[df_true_events['detections'] > 0]),
                    'FP': FP,
                    'FN': len(df_true_events[df_true_events['detections'] == 0]),
                    'event_type': event_type,
                    'percentile': 100 - percentile
                }, ignore_index=True)

    df_station_stats['precision'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FP'].astype(int))
    df_station_stats['recall'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FN'].astype(int))
    return df_station_stats
```

```python
df_station_stats = get_station_stats(df_station_dict, "event_danger", rp=MAIN_RP)
df_station_stats
```

```python
df_station_stats = get_station_stats(df_station_dict, "event_rp1.5", rp=1.5)
df_station_stats
```

### Get list of historical activations

```python
# Go through each date and list the activations

event_bools = {
    'readiness': False,
    #'action': False,
    'rp1.5': False
    #'danger': False
}
rp = 1.5

for station in STATIONS:
    for percentile in FORECAST_PERCENTILE_LIST:
        print(station)
        print(100-percentile)
        df_station = df_station_dict[station]#.copy().dropna(subset=["water_level"])
        df_events = pd.DataFrame(columns=['event', 'leadtimes'])
        rp_val = df_return_period.loc[rp, station]
        for date, row in df_station.iterrows():
            for event_type in event_bools.keys():
                cname = f'event_{event_type}'
                if event_type in ['readiness', 'action']:
                    cname += f"_rp{rp}_p{percentile}"
                if row[cname]:
                    if not event_bools[event_type]:
                        s = 'flood' if event_type == 'danger' else event_type + ' trigger'
                        print(date, s)
                        # Find lead time
                        if event_type in ['readiness', 'action']:
                            forecast_leadtimes = []
                            for leadtime in LEADTIMES_BY_TRIGGER[event_type]:
                                if row[f'forecast_lt{leadtime}_p{percentile}'] > rp_val:
                                    forecast_leadtimes += [leadtime]
                            df_events.loc[date, 'leadtimes'] = forecast_leadtimes
                            #print(forecast_leadtimes)
                    event_bools[event_type] = True
                else:
                    event_bools[event_type] = False
```

```python
df_station.columns
```

### Plots

```python
mpl.rcParams['hatch.linewidth'] = 0.5
for station in STATIONS:
    df_station = df_station_dict[station].copy().dropna(subset=["water_level"])
    thresh = df_station_info.at[station, f'danger_level']
    rp_val = df_return_period.loc[MAIN_RP, station]
    years = df_station.index.year.unique()
    fig, axs = plt.subplots(len(years), figsize=(8, 20))
    for year, ax in zip(years, axs):
        data = df_station[df_station.index.year == year]
        # Plot everything
        ax.plot(data.index, data.water_level, c='k', label='water level', alpha=0.75, lw=0.75)
        ax2 = ax.twinx()
        ax2.plot(data.index, data.river_discharge, c='C0', label='river_discharge', alpha=0.5, lw=0.5)
        ax.plot([], [], 'C0-', label='river discharge', alpha=0.5, lw=0.5)
        ax.axhline(y=thresh, c='C3', label='danger / 1 in 2 y', alpha=0.75, lw=0.5)
        
        y0 = -0.05 * thresh
        y1 = thresh*1.2
        ax.set_ylim(y0, y1)
        ax2.set_ylim(-0.05 * rp_val, rp_val*1.2)
        ax.set_xticklabels([])
        ax.text(data.index[0], thresh + 0.2, f"{year}")
        ax.grid()
        # Add the activations
        for event_type, hatch, colour, lw in zip(['readiness', 'action','danger'], 
                                             ['/////', '\\\\\\\\\\', '.....'], 
                                             ['C2', 'C1', 'C6'],
                                             [0.5, 0.5, 2]):
            cname = f'event_{event_type}'
            if event_type != "danger":
                cname += f"_rp{MAIN_RP}"
            ax.fill_between(data.index, y0, y1, where=data[cname], 
                            facecolor='none', hatch=hatch, edgecolor=colour, lw=lw, 
                            alpha=.75, label=event_type)
    fig.supylabel('Water level [m]')
    fig.suptitle(station)    
    ax.legend()
```

```python

```