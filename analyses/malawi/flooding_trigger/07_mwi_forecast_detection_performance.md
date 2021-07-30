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
PRIVATE_DIR = config.DATA_PRIVATE_DIR
EXPLORE_DIR = PRIVATE_DIR / 'exploration' / 'mwi' / 'flooding'

SAVE_PLOT = True
EVENT = 'RCO' # 'rco', 'floodscan', 'combined'
COUNTRY_ISO3 = 'mwi'

stations_adm2 = {
    #'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}

DURATION = 3
LEADTIMES = [x + 1 for x in range(10)]
FORECAST_PROB = 50
RP_LIST = [1.5, 2, 5]
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
df_station_stats = pd.DataFrame(columns=['station', 'rp', 'leadtime', 'TP', 'FP', 'FN', 'precision', 'recall'])

for station in df_return_period.columns:
    
    da_glofas_reanalysis = ds_glofas_reanalysis[station].reindex(time=ds_glofas_reforecast.time)
    observations = da_glofas_reanalysis.values
    
    for rp in RP_LIST:
        rp_val = df_return_period.loc[rp, station]

        observation_groups = utils.get_groups_above_threshold(observations, rp_val, min_duration=DURATION)
        df_observation_groups = utils.get_dates_from_groups(observation_groups, da_glofas_reanalysis)

        for leadtime in LEADTIMES:
            
            da_glofas_reforecast = ds_glofas_reforecast_summary[station].sel(percentile=FORECAST_PROB).sel(leadtime=leadtime)
            forecast = da_glofas_reforecast.values
            forecast_groups = utils.get_groups_above_threshold(forecast, rp_val)
            df_forecast_groups = utils.get_dates_from_groups(forecast_groups, da_glofas_reforecast)

            TP, FP, FN, precision, recall = utils.get_detection_stats(df_observation_groups, df_forecast_groups)

            df_station_stats = df_station_stats.append({
                'station': station,
                'leadtime': leadtime,
                'rp': rp,
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'precision': precision,
                'recall': recall
            }, ignore_index=True)
```

Plot the detection stats against the historical GloFAS data.

```python
rp_dict = { 
    2: '-',
    5: '--'
}

plot_numbers = True
plot_precision_recall = True
leadtime_range = (1, 10)

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
#event_sources = ['rco']
events = {}
for station in stations_adm2.values():
    sources = {}
    for source in event_sources:
        sources[source] = pd.read_csv(EXPLORE_DIR / f'{station}_{source}_event_summary.csv')
        #sources[source] = pd.read_csv(EXPLORE_DIR / f'all_{source}_event_summary.csv')
    events[station] = sources
```

Compute the standard performance metrics between each of the event sources and the GloFAS forecast data.

```python
start_slice = '1998-01-01'
end_slice = '2019-12-31'

def filter_event_dates(df_event, start, end):
    return df_event[(df_event['start_date']<str(end)) & (df_event['start_date']>str(start))].reset_index()


df_station_stats = pd.DataFrame(columns=['station', 'rp', 'leadtime', 'source', 'TP', 'FP', 'FN', 'precision', 'recall'])

# TODO: Here we're limiting the time window to 1998-2019. 
# Could better tailor this to be more specific to each source.
# Floodscan starts 1998-01-12 and ends 2020-12-31
# EM-DAT starts in 2000 (when geocoding events started) and goes until 2020
# RCO has events as early as 1973 and ends 2020

for code, station in stations_adm2.items():
    
    for rp in RP_LIST:
        rp_val = df_return_period.loc[rp, code]
        
        for leadtime in LEADTIMES:
            
            da_glofas_reforecast = ds_glofas_reforecast_summary[code].sel(percentile=FORECAST_PROB).sel(leadtime=leadtime)
            forecast = da_glofas_reforecast.values
            forecast_groups = utils.get_groups_above_threshold(forecast, rp_val)
            df_forecast_groups = utils.get_dates_from_groups(forecast_groups, da_glofas_reforecast)

            for source in event_sources: 

                df_event = filter_event_dates(events[station][source], start_slice, end_slice) 
                TP, FP, FN, precision, recall = utils.get_detection_stats(df_event, df_forecast_groups)
                
                df_station_stats = df_station_stats.append({
                    'station': station,
                    'leadtime': leadtime,
                    'source': source,
                    'rp': rp,
                    'TP': TP,
                    'FP': FP,
                    'FN': FN,
                    'precision': precision,
                    'recall': recall
                }, ignore_index=True)
```

Plot out the results.

```python
fig, axs = plt.subplots(len(stations_adm2.values()), len(RP_LIST), figsize=(15, 5 * len(stations_adm2.values())), sharex=True, sharey=True)

for istation, station in enumerate(stations_adm2.values()):
    
    for irp, rp in enumerate(RP_LIST):
        if len(stations_adm2.values()) > 1: 
            ax = axs[istation, irp]
        else: 
            ax = axs[irp]

        for isource, source in enumerate(event_sources): 
            df_sel = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['source'] == source) & (df_station_stats['rp'] == rp)]
            ax.plot(df_sel['leadtime'], df_sel['precision'], color=f'C{isource}', ls='--', lw=0.75)
            ax.plot(df_sel['leadtime'], df_sel['recall'], color=f'C{isource}', ls='-', lw=0.75)  

            # Add to the legend
            ax.plot([], [], label=source.capitalize(), color=f'C{isource}')
        ax.plot([], [], color='k', label='Precision', ls='--')
        ax.plot([], [], color='k', label='Recall')        

        ax.legend()
        ax.set_ylim([0, 1])
        ax.set_title(f'{station}:\n1 in {rp} year threshold')
    
if SAVE_PLOT: plt.savefig(PLOT_DIR / f'reforecast_event_performance.png')
```
