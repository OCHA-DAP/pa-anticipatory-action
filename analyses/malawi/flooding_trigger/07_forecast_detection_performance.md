### Calculating GloFAS detection performance

This notebook calculates and visualizes statistics on GloFAS forecast performance in detecting historical flood events. We compare performance across various trigger thresholds and forecast lead times. We also compare detection performance across various definitions/sources of historical flooding.

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
from datetime import timedelta

import read_in_data as rd
from importlib import reload
from matplotlib.ticker import MaxNLocator

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import utils as utils

reload(utils)
config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
LEADTIMES = [5, 10, 15, 20, 25, 30]
SAVE_PLOT = False
EVENT = 'combined' # 'rco' or 'floodscan' or 'combined'
COUNTRY_ISO3 = 'mwi'

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}

DURATION =3
```

Read in the processed GloFAS data and get a summary of the reforecasted values. The summary includes percentile values to summarize the distribution of the ensemble forecast. We'll also get a dataframe of calculated return period thresholds and read in the dataset of historical events.

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=True
)
ds_glofas_reforecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
#ds_glofas_reforecast_summary['glofas_1'].sel(leadtime=5).sel(percentile=50).values
```

We'll check to see how the GloFAS forecast performs across various leadtimes in detecting streamflow exceedance events from the reanalysis (historical) GloFAS data. These events don't necessarily correspond to floods, but these results give us a sense of how the GloFAS forecast performs across various leadtimes at predicting the kinds of streamflow levels that we are interested in. 

```python
forecast_prob = 50

days_before_buffer = 5
days_after_buffer = 30

rp_list = [1.5, 2, 5]

df_station_stats = pd.DataFrame(columns=['station', 'rp', 'leadtime', 'TP', 'FP', 'FN'])

for station in df_return_period.columns:
    for rp in rp_list:
        rp_val = df_return_period.loc[rp, station]
        observations = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station].values
        forecast = ds_glofas_reforecast_summary[station].sel(percentile=forecast_prob)

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
            
df_station_stats['precision'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FP'].astype(int))
df_station_stats['recall'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FN'].astype(int))
```

Plot the detection stats against the historical GloFAS data.

```python
rp_dict = { 
    2: '-',
    5: '--'
}

plot_numbers = True
plot_precision_recall = True
leadtime_range = (1, 30)

for istation, station in enumerate(STATIONS):
    if plot_numbers:
        qs = ['TP', 'FP', 'FN']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp)]
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
            data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp)]
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

Now we'll see how the GloFAS forecast performs in detecting historical flood events as defined in our event dataset (eg. from the RCO, Floodscan, EM-DAT or combined). We'll first plot the time series of streamflow forecast against the timing of our events to see what's going on.

```python

```
