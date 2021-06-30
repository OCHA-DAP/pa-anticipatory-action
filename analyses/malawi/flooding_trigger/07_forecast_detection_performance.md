### Calculating GloFAS detection performance

This notebook calculates and visualizes statistics on GloFAS forecast performance in detecting historical flood events. We compare performance across various trigger thresholds and forecast lead times. We also compare detection performance across various definitions/sources of historical flooding.

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

from matplotlib.ticker import MaxNLocator

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import utils

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
LEADTIMES = [5, 10, 15, 20, 25, 30]
SAVE_PLOT = True
EVENT = 'combined' # 'rco', 'floodscan', 'combined'
COUNTRY_ISO3 = 'mwi'

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}

DURATION = 3
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
df_return_period = utils.get_return_periods(ds_glofas_reanalysis, method='analytical')
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

for code, station in stations_adm2.items():
    if plot_numbers:
        qs = ['TP', 'FP', 'FN']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == code) & (df_station_stats['rp'] == rp)]
            for iq, q in enumerate(qs):
                ax.plot(data['leadtime'], data[q], ls=ls, c=f'C{iq}', alpha=0.75)
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
        
        if SAVE_PLOT: plt.savefig(PLOT_DIR / f'reanalysis_reforecast_performance_{station}_numbers.png')
    
    if plot_precision_recall:
        qs = ['precision', 'recall']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == code) & (df_station_stats['rp'] == rp)]
            for iq, q in enumerate(qs):
                ax.plot(data['leadtime'], data[q], ls=ls, c=f'C{iq}', alpha=0.75)
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
        
        if SAVE_PLOT: plt.savefig(PLOT_DIR / f'reanalysis_reforecast_performance_{station}_stats.png')
```

Now we'll see how the GloFAS forecast performs in detecting historical flood events as defined in our event dataset (eg. from the RCO, Floodscan, EM-DAT or combined). We'll first read in all of the historical event datasets.

```python
event_sources = ['combined', 'rco', 'emdat', 'floodscan']
events = {}
for station in stations_adm2.values():
    sources = {}
    for source in event_sources:
        sources[source] = pd.read_csv(EXPLORE_DIR / f'{station}_{source}_event_summary.csv')
    events[station] = sources
```

Compute the standard performance metrics between each of the event sources and the GloFAS forecast data.

```python
days_before_buffer = 30
days_after_buffer = 30
start_slice = '1998-01-01'
end_slice = '2019-12-31'

forecast_prob = 50

rp_list = [1.5, 2, 5]
lt_list = [5, 10, 15, 20]

df_detection_stats = pd.DataFrame(columns=['station', 'lead_time', 'return_period', 'source', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1'])

def filter_event_dates(df_event, start, end):
    return df_event[(df_event['start_date']<str(end)) & (df_event['start_date']>str(start))].reset_index()

# TODO: Here we're limiting the time window to 1998-2019. 
# Could better tailor this to be more specific to each source.
# Floodscan starts 1998-01-12 and ends 2020-12-31
# EM-DAT starts in 2000 (when geocoding events started) and goes until 2020
# RCO has events as early as 1973 and ends 2020

for code, station in stations_adm2.items():
    
    for rp in rp_list:
        rp_val = df_return_period.loc[rp, code]
        
        detection_stats = {}
        
        for lt in lt_list: 
            df_glofas_event = utils.get_glofas_activations(ds_glofas_reforecast_summary[code]
                                                           .sel(leadtime=lt)
                                                           .sel(percentile=forecast_prob)
                                                           .sel(time=slice(start_slice, end_slice)), 
                                                           rp_val, DURATION)

            for source in event_sources: 

                df_event = filter_event_dates(events[station][source], start_slice, end_slice) 
                dict_performance = utils.get_clean_stats_dict(df_glofas_event, df_event, days_before_buffer, days_after_buffer)
                dict_performance['return_period'] = rp
                dict_performance['station'] = station
                dict_performance['lead_time'] = lt
                detection_stats[source] = dict_performance

            df_detection_stats = df_detection_stats.append(pd.DataFrame
                                  .from_dict(detection_stats)
                                  .transpose()
                                  .reset_index()
                                  .rename(columns={'index':'source'}))  
```

Plot out the results.

```python
fig, axs = plt.subplots(2, len(rp_list), figsize=(15, 10), sharex=True, sharey=True)

for istation, station in enumerate(stations_adm2.values()):
    
    for irp, rp in enumerate(rp_list):
    
        ax = axs[istation, irp]

        for isource, source in enumerate(event_sources): 
            df_sel = df_detection_stats[(df_detection_stats['station'] == station) & (df_detection_stats['source'] == source) & (df_detection_stats['return_period'] == rp)]
            ax.plot(df_sel['lead_time'], df_sel['precision'], color=f'C{isource}', ls='--', lw=0.75)
            ax.plot(df_sel['lead_time'], df_sel['recall'], color=f'C{isource}', ls='-', lw=0.75)  

            # Add to the legend
            ax.plot([], [], label=source.capitalize(), color=f'C{isource}')
        ax.plot([], [], color='k', label='Precision', ls='--')
        ax.plot([], [], color='k', label='Recall')        

        ax.legend()
        ax.set_ylim([0, 1])
        ax.set_title(f'{station}:\n1 in {rp} year threshold')
    
if SAVE_PLOT: plt.savefig(PLOT_DIR / f'reforecast_event_performance.png')
```
