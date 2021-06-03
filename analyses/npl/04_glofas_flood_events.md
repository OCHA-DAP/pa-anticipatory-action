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

from src.indicators.flooding.glofas import utils, glofas
```

```python
mpl.rcParams['figure.dpi'] = 200

LEADTIMES = [x + 1 for x in range(20)]

COUNTRY_ISO3 = 'npl'
STATIONS = {
    'Koshi': ['Chatara', 'Simle', 'Majhitar'],
    'Karnali': ['Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat'],
    'West Rapti': ['Kusum'],
    'Bagmati': ['Rai_goan'],
    'Babai': ['Chepang']
}

```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=True
)

df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

```python
def get_da_glofas_summary(da_glofas):
    nsig_max = 3
    percentile_dict = {
        percentile: percentile for percentile in np.arange(0, 105, 5)
    }
    coord_names = ["leadtime", "time"]
    data_vars_dict = {
        var_name: (coord_names, np.percentile(da_glofas, percentile_value, axis=1))
        for var_name, percentile_value in percentile_dict.items()
    }
    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(time=da_glofas.time, leadtime=da_glofas.leadtime),
    )

```

```python
ndays = 3
forecast_prob = 50

days_before_buffer = 5
days_after_buffer = 25

rp_list = [1.5, 2]

df_station_stats = pd.DataFrame(columns=['station', 'rp', 'leadtime', 'TP', 'FP', 'FN'])

for station in df_return_period.columns:
    #for rp in df_return_period.index:
    for rp in rp_list:
        rp_val = df_return_period.loc[rp, station]
        observations = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station].values
        forecast = get_da_glofas_summary(ds_glofas_reforecast[station])[forecast_prob]

        # The GlofAS event takes place on the Nth day (since for an event)
        # you require N days in a row
        event_groups = utils.get_groups_above_threshold(observations, rp_val, min_duration=ndays)
        event_dates = [event_group[0] + ndays - 1 for event_group in event_groups]

        for leadtime in LEADTIMES:
            TP = 0
            FN = 0
            
            forecast_groups = utils.get_groups_above_threshold(forecast.sel(leadtime=leadtime), rp_val, min_duration=ndays)
            forecast_dates = np.array([forecast_group[0] + ndays - 1 for forecast_group in forecast_groups])
            forecast_detections = np.zeros(len(forecast_dates))
            
            for event_date in event_dates:
                # Check if any events are around that date
                days_offset = forecast_dates - event_date

                detected = (days_offset > -1 * days_before_buffer) & (days_offset < days_after_buffer)
                

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
rp = 2
plot_numbers = False
plot_precision_recall = True
for station in ['Chatara', 'Kampughat', 'Asaraghat', 'Samajhighat', 'Chepang']:
    if plot_numbers:
        fig, ax = plt.subplots()
        data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp)]
        ax.plot(data['leadtime'], data['TP'], label='TP')
        ax.plot(data['leadtime'], data['FP'], label='FP')
        ax.plot(data['leadtime'], data['FN'], label='FN')
        ax.set_title(station)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Leadtime [days]')
        ax.set_ylabel('Number')
        ax.legend()
    
    if plot_precision_recall:
        fig, ax = plt.subplots()
        data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp)]
        ax.plot(data['leadtime'], data['precision'], label='precision')
        ax.plot(data['leadtime'], data['recall'], label='recall')
        ax.set_title(station)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Leadtime [days]')
        ax.set_ylabel('Fraction')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
```

```python
rp = 2
station = 'Chatara'
forecast_prob = 50
GLOFAS_DETECTION_WINDOW_BEHIND = 5
GLOFAS_DETECTION_WINDOW_AHEAD = 30

thresh = df_return_period.loc[rp, station]
fig, axs = plt.subplots(3, figsize=(8,6))
observations = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station]
forecast = get_da_glofas_summary(ds_glofas_reforecast[station])[forecast_prob]
forecast_1 = forecast.sel(leadtime=5)
forecast_2 = forecast.sel(leadtime=10)
fig.supylabel('River dischange [m$^3$ s${-1}$]')
fig.supxlabel('Year')
fig.suptitle(station)
for i, q in enumerate([forecast_2, forecast_1, observations]):
    if i == 2:
        c1 = 'k'
        c2 = 'C3'
    else:
        c1 = 'C0'
        c2 = 'C1'
    ax = axs[i]
    x = q.time
    y = q.values
    ax.plot(x, y, '-', c=c1, lw=0.5)
    ax.set_ylim(0, 10000)
    #ax.set_xlim(0, 120000)
    ax.minorticks_on()
    ax.axhline(thresh, c='r', lw=0.5)
    for detection in utils.get_groups_above_threshold(y, thresh, ndays):
        a = detection[0] - GLOFAS_DETECTION_WINDOW_BEHIND
        b = detection[0] + GLOFAS_DETECTION_WINDOW_AHEAD
        #ax.plot(x[a:b], y[a:b], '-r', lw=1, alpha=0.5)
        ax.plot(x[detection[0]], y[detection[0]], 'o', c=c2, lw=2, mfc='none')
```
