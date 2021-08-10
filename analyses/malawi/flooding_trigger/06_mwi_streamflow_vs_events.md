### Comparing flood events with historical streamflow

This notebook looks at the correlation between peaks in historical streamflow (from GloFAS reanalysis) and the timing of past flood events identified from various sources. 

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
from datetime import timedelta

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
COUNTRY_ISO3 = 'mwi'

stations_adm2 = {
   # 'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}

DURATION = 3
```

Read in the historical GloFAS data (reanalysis) and the various event datasets.

```python
#event_sources = ['combined', 'rco', 'emdat', 'floodscan']
event_sources = ['rco']

ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)

# TODO: Rerun with return period values that are specified by GloFAS
df_return_period = utils.get_return_periods(ds_glofas_reanalysis, method='analytical')

events = {}
for station in stations_adm2.values():
    sources = {}
    for source in event_sources:
        #sources[source] = pd.read_csv(EXPLORE_DIR / f'{station}_{source}_event_summary.csv')
        sources[source] = pd.read_csv(EXPLORE_DIR / f'all_{source}_event_summary.csv') # Using the combined dataset for Nsanje and Chikwawa
    events[station] = sources
```

Plot out the historical streamflow against the timing of each of the historical events.

```python
rp_list = [1.5, 2, 5]
start_slice = '1998-01-01'
end_slice = '2019-12-31'

def filter_event_dates(df_event, start, end):
    return df_event[(df_event['start_date']<str(end)) & (df_event['start_date']>str(start))].reset_index()

for code, station in stations_adm2.items(): 
    
    fig, axs = plt.subplots(len(event_sources), figsize=(10,5 * len(stations_adm2.values())), squeeze=False, sharex=True, sharey=True)
    fig.suptitle(f'Historical streamflow at {station}')
    
    for isource, source in enumerate(event_sources):
               
        da_plt = ds_glofas_reanalysis[code].sel(time=slice(start_slice, end_slice))
        df_event = filter_event_dates(events[station][source],start_slice, end_slice) 
        
        observations = da_plt.values
        x = da_plt.time

        ax = axs[isource, 0]
        ax.plot(x, observations, c='k', lw=0.75, alpha=0.75)
        ax.set_ylabel('Discharge [m$^3$ s$^{-1}$]')
        ax.set_title(source.upper())

        for i in range(0,len(df_event['start_date'])):
            ax.axvspan(np.datetime64(df_event['start_date'][i]), np.datetime64(df_event['end_date'][i]), alpha=0.25, color='#3ea7f7')
        for irp, rp in enumerate(rp_list):
            ax.axhline(df_return_period.loc[rp, code],  0, 1, color=f'C{irp+1}', alpha=1, lw=0.75, label=f'1 in {str(rp)}-year return period')

        ax.legend()

        if SAVE_PLOT: plt.savefig(PLOT_DIR / f'{station}_streamflow_vs_events.png')
```

For each of the event sources, compare the detection statistics against GloFAS for various return periods.

```python
days_before_buffer = 30
days_after_buffer = 30
start_slice = '1998-01-01'
end_slice = '2019-12-31'

rp_list = [1.5, 2, 5]

df_detection_stats = pd.DataFrame(columns=['station', 'return_period', 'source', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1'])

# TODO: Here we're limiting the time window to 1998-2019. 
# Could better tailor this to be more specific to each source.
# Floodscan starts 1998-01-12 and ends 2020-12-31
# EM-DAT starts in 2000 (when geocoding events started) and goes until 2020
# RCO has events as early as 1973 and ends 2020
for code, station in stations_adm2.items():
    
    for rp in rp_list:
        rp_val = df_return_period.loc[rp, code]
        df_glofas_event = utils.get_glofas_activations(ds_glofas_reanalysis[code].sel(time=slice(start_slice, end_slice)), rp_val, DURATION)
        
        detection_stats = {}
        
        for source in event_sources: 
            
            df_event = filter_event_dates(events[station][source], start_slice, end_slice) 
            dict_performance = utils.get_clean_stats_dict(df_glofas_event, df_event, days_before_buffer, days_after_buffer)
            dict_performance['return_period'] = rp
            dict_performance['station'] = station
            detection_stats[source] = dict_performance
            
        df_detection_stats = df_detection_stats.append(pd.DataFrame
                              .from_dict(detection_stats)
                              .transpose()
                              .reset_index()
                              .rename(columns={'index':'source'}))          
```

Plot out the results.

```python
fig, axs = plt.subplots(len(stations_adm2.values()), figsize=(10, 5 * len(stations_adm2.values())), sharex=True, sharey=True)

for istation, station in enumerate(stations_adm2.values()):
    
    if len(stations_adm2.values())>1:
        ax = axs[istation]
    else:
        ax = axs
    
    for isource, source in enumerate(event_sources): 
        df_sel = df_detection_stats[(df_detection_stats['station'] == station) & (df_detection_stats['source'] == source)]
        ax.plot(df_sel['return_period'], df_sel['precision'], color=f'C{isource}', ls='--', lw=0.75, marker='o')
        ax.plot(df_sel['return_period'], df_sel['recall'], color=f'C{isource}', ls='-', lw=0.75, marker='x')  
        
        # Add to the legend
        ax.plot([], [], label=source.capitalize(), color=f'C{isource}')
    ax.plot([], [], color='k', marker='o', label='Precision', ls='--')
    ax.plot([], [], color='k', marker='x', label='Recall')        

    ax.legend()
    ax.set_title(f'Flood detection performance at {station} against GloFAS reanalysis')
    
if SAVE_PLOT: plt.savefig(PLOT_DIR / f'reanalysis_event_performance.png')
```