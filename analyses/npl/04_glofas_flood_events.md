```python
from pathlib import Path
import os
import sys

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from matplotlib.ticker import MaxNLocator


path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
```

```python
mpl.rcParams['figure.dpi'] = 200

LEADTIMES = [x + 1 for x in range(10)]

COUNTRY_ISO3 = 'npl'

FINAL_STATIONS = ["Chatara", "Chisapani", "Asaraghat"]
# Use "_v3" for the GloFAS model v3 locs, or empty string for the original v2 ones
VERSION_LOC = "_v3"
USE_INCORRECT_AREA_COORDS = False

MAIN_RP = 1.5
MAIN_FORECAST_PROB = 50
    

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"

DURATION = 1
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3, use_incorrect_area_coords=USE_INCORRECT_AREA_COORDS)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=True,
    use_incorrect_area_coords=USE_INCORRECT_AREA_COORDS
)
ds_glofas_reforecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)

#df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
df_return_period =  pd.read_excel(GLOFAS_RP_FILENAME, index_col='rp')

```

```python
days_before_buffer = 5
days_after_buffer = 30

rp_list = [1.5, 2, 5]

df_station_stats = pd.DataFrame(columns=['station', 'rp', 'leadtime', 'TP', 'FP', 'FN'])

for station in df_return_period.columns:
    #for rp in df_return_period.index:
    for rp in rp_list:
        rp_val = df_return_period.loc[rp, station]
        observations = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station + VERSION_LOC].values
        forecast = ds_glofas_reforecast_summary[station + VERSION_LOC].sel(percentile=MAIN_FORECAST_PROB)

        # The GlofAS event takes place on the Nth day (since for an event)
        # you require N days in a row
        event_groups = utils.get_groups_above_threshold(observations, rp_val, min_duration=DURATION)
        event_dates = [event_group[0] + DURATION - 1 for event_group in event_groups]

        for leadtime in LEADTIMES:
            TP = 0
            FN = 0
            
            forecast_groups = utils.get_groups_above_threshold(forecast.sel(leadtime=leadtime), rp_val, min_duration=DURATION)
            forecast_dates = np.array([forecast_group[0] + DURATION - 1 for forecast_group in forecast_groups])
            forecast_detections = np.zeros(len(forecast_dates))
            
            for event_date in event_dates:
                # Check if any events are around that date
                days_offset = forecast_dates - event_date

                detected = (days_offset >= -1 * days_before_buffer) & (days_offset <= days_after_buffer)
                

                # If there were any detections, it's  a TP. Otherwise a FP
                if sum(detected):
                    TP += 1
                else:
                    FN += 1
                forecast_detections[detected] += 1
                
            FP = sum(forecast_detections == 0)
            df_station_stats = df_station_stats.append({
                'station': station,
                'leadtime': leadtime,
                'rp': rp,
                'TP': TP,
                'FP': FP,
                'FN': FN,
            }, ignore_index=True)

```

```python
df_station_stats['precision'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FP'].astype(int))
df_station_stats['recall'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FN'].astype(int))
```

```python
#rp_dict = {
#    1.5: '-', 
#    2: '--',
#    5: ':'
#}
rp_dict = {MAIN_RP: '-'}
plot_numbers = True
plot_precision_recall = True
leadtime_range = (1, 10)

for istation, station in enumerate(FINAL_STATIONS):
    if plot_numbers:
        qs = ['TP', 'FP', 'FN']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == MAIN_RP)]
            for iq, q in enumerate(qs):
                ax.plot(data['leadtime'], data[q], ls=ls, c=f'C{iq}')
        ax.set_title(station)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Leadtime [days]')
        ax.set_ylabel('Number')
        # Make legend
        for iq, q in enumerate(qs):
            ax.plot([], [], c=f'C{iq}', label=q)
        if len(rp_dict) > 1:
            for rp, ls in rp_dict.items():
                ax.plot([], [], c='k', ls=ls, label=rp)
        ax.legend()
        ax.set_xlim(leadtime_range)
    
    if plot_precision_recall:
        qs = ['precision', 'recall']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == MAIN_RP)]
            for iq, q in enumerate(qs):
                ax.plot(data['leadtime'], data[q], ls=ls, c=f'C{iq}')
        ax.set_title(station)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Leadtime [days]')
        ax.set_ylabel('Fraction')
        ax.set_ylim(-0.1, 1.1)
        # Make legend
        for iq, q in enumerate(qs):
            ax.plot([], [], c=f'C{iq}', label=q)
        if len(rp_dict) > 1:
            for rp, ls in rp_dict.items():
                ax.plot([], [], c='k', ls=ls, label=rp)
        ax.legend()
        ax.set_xlim(leadtime_range)
```

```python
# Print out some stats
def round_to_5(x):
    return (np.around(x/5, decimals=0)*5).astype(int)
for station in FINAL_STATIONS:
    data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp)]
    data.loc[:, "POD"] = round_to_5(data["recall"].fillna(-1) * 100)
    data.loc[:, "FAR"] = round_to_5((1 - data["precision"]).fillna(-1) * 100)
    print(station)
    print(data[["leadtime", "POD", "FAR"]])
```

### Make plot showing event comparison

```python
rp = 1.5
leadtimes = [7, 3] # Longer first

for station in FINAL_STATIONS:

    thresh = df_return_period.loc[rp, station]
    fig, axs = plt.subplots(3, figsize=(8,6))
    observations = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station + VERSION_LOC]
    forecast = utils.get_glofas_forecast_summary(ds_glofas_reforecast)[station + VERSION_LOC].sel(percentile=MAIN_FORECAST_PROB)
    forecast_1 = forecast.sel(leadtime=leadtimes[0])
    forecast_2 = forecast.sel(leadtime=leadtimes[1])
    fig.supylabel('River dischange [m$^3$ s$^{-1}$]')
    fig.supxlabel('Year')
    fig.suptitle(station)
    for i, q in enumerate([forecast_1, forecast_2, observations]):
        ax = axs[i]
        if i == 2:
            c1 = 'k'
            c2 = 'C3'
            title = 'Modelled'
        else:
            c1 = 'C0'
            c2 = 'C1'
            title = f'{leadtimes[i]}-day lead time'
            ax.set_xticklabels([])
        x = q.time
        y = q.values
        ax.plot(x, y, '-', c=c1, lw=0.5)
        ax.set_ylim(0, 8000)
        #ax.set_xlim(0, 120000)
        ax.minorticks_on()
        ax.axhline(thresh, c=c2, lw=0.5)
        ax.set_title(title)
        for detection in utils.get_groups_above_threshold(y, thresh, DURATION):
            ax.plot(x[detection[0]], y[detection[0]], 'o', c=c2, lw=2, mfc='none')
```

### Compare Chisapani to Asaraghat

```python
# Want to compare Chisapani to Asaraghat
ls_list = ['-', '--']
stations = ['Asaraghat', 'Chisapani']
fig, ax = plt.subplots()
for istation, station in enumerate(stations):
    qs = ['precision', 'recall']
    data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == MAIN_RP)]
    for iq, q in enumerate(qs):
        ax.plot(data['leadtime'], data[q], ls=ls_list[istation], c=f'C{iq}')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Leadtime [days]')
ax.set_ylabel('Fraction')
ax.set_ylim(-0.1, 1.1)
# Make legend
for iq, q in enumerate(qs):
    ax.plot([], [], c=f'C{iq}', label=q)
for istation, station in enumerate(stations):
    ax.plot([], [], c='k', ls=ls_list[istation], label=station)
ax.legend()
```

```python
rp = 1.5
leadtimes = [1, 3, 5, 7, 10]
for station in ['Asaraghat', 'Chisapani']:
    df = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp) & (df_station_stats.leadtime.isin(leadtimes))]
    print(np.round(100 * (1-df['recall'])))
```
