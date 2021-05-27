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
reload(utils)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import utils

reload(utils)
config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
LEADTIMES = [5, 10, 15, 20, 25, 30]

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}
```

### Read in GloFAS data

```python
da_glofas_reanalysis = {}
da_glofas_reforecast = {}
da_glofas_reforecast_interp = {}
da_glofas_forecast = {}
da_glofas_forecast_summary = {}
da_glofas_reforecast_summary = {}

for station in STATIONS: 
    da_glofas_reanalysis[station] = utils.get_glofas_reanalysis('mwi', version=GLOFAS_VERSION)[station]
    da_glofas_reforecast[station] = utils.get_glofas_reforecast('mwi', LEADTIMES, interp=False, version=GLOFAS_VERSION)[station]
    da_glofas_reforecast_interp[station] = utils.get_glofas_reforecast('mwi', LEADTIMES, interp=True, version=GLOFAS_VERSION)[station]
    da_glofas_forecast[station] = utils.get_glofas_forecast('mwi', LEADTIMES, version=GLOFAS_VERSION)[station]
    da_glofas_forecast_summary[station] = utils.get_da_glofas_summary(da_glofas_forecast[station])
    da_glofas_reforecast_summary[station] = utils.get_da_glofas_summary(da_glofas_reforecast_interp[station])
```

### Read in the baseline impact data

```python
floodscan_events = {}
for station in stations_adm2.values():
    floodscan_events[station] = pd.read_csv(EXPLORE_DIR / f'{station}_floodscan_event_summary.csv')
```

### Calculate the return period

```python
def get_return_period_function(observations, station):
    df_rp = (observations.to_dataframe()
                 .rename(columns={station: 'discharge'})
                 .resample(rule='A', kind='period')
                 .max() 
                 .sort_values(by='discharge', ascending=False)
                )
    df_rp["year"] = df_rp.index.year
     
    n = len(df_rp)
    df_rp['rank'] = np.arange(n) + 1
    df_rp['exceedance_probability'] = df_rp['rank'] / (n+1)
    df_rp['rp'] = 1 / df_rp['exceedance_probability']
    return interp1d(df_rp['rp'], df_rp['discharge'])

rp_dict = {}

for code, station in stations_adm2.items(): 
    f_rp = get_return_period_function(da_glofas_reanalysis[code], code)
    rp_dict[station] = {}
    for year in [1.5, 2, 3, 4, 5, 10, 20]:
        val = 10*np.round(f_rp(year) / 10)
        rp_dict[station][year] = val

df_rps = pd.DataFrame(rp_dict).reset_index().rename(columns={'index': 'rp'})
df_rps.to_csv(EXPLORE_DIR / 'glofas_rps.csv')
```

### Overview of historical discharge

```python
# Return periods to focus on, with display colours
rps = {
    3: '#32a852',
    5: '#9c2788'
}

for code, station in stations_adm2.items(): 
    da_plt = da_glofas_reanalysis[code].sel(time=slice('1999-01-01','2019-12-31'))
    df_floodscan_event = floodscan_events[station]

    fig, ax = plt.subplots()
    da_plt.plot(x='time', add_legend=True, ax=ax)
    ax.set_title(f'Historical streamflow at {station}')
    ax.set_xlabel("Date")
    ax.set_ylabel('Discharge [m$^3$ s$^{-1}$]')

    for i in range(0,len(df_floodscan_event['start_date'])):
        ax.axvspan(np.datetime64(df_floodscan_event['start_date'][i]), np.datetime64(df_floodscan_event['end_date'][i]), alpha=0.5, color='#FE5E1E')
    
    for key, value in rps.items():
        ax.axhline(rp_dict[station][key], 0, 1, color=value, label=f'{str(key)} return period')
        
    ax.legend()
    
    plt.savefig(PLOT_DIR / f'{station}_historical_discharge_glofas_overview_rps.png')
```

### Identifying glofas events

```python
def get_groups_above_threshold(observations, threshold):
    return np.where(np.diff(np.hstack(([False],
                                           observations > threshold,
                                           [False]))))[0].reshape(-1, 2)

def get_glofas_activations(da_glofas, thresh, ndays):
    vals = da_glofas.values
    groups = get_groups_above_threshold(vals, thresh)
    groups = [group for group in groups if group[1] - group[0] >= ndays]
    df_glofas_act = pd.DataFrame(groups, columns=['start_index', 'end_index'])
    df_glofas_act['num_days'] = df_glofas_act['end_index'] - df_glofas_act['start_index']
    df_glofas_act['start_date'] = df_glofas_act['start_index'].apply(lambda x: da_glofas.time[x].values)
    df_glofas_act['end_date'] = df_glofas_act['end_index'].apply(lambda x: da_glofas.time[x].values)
    return df_glofas_act

def get_detection_stats(df_glofas, df_impact, buffer):
    TP = 0 
    FP = 0
    tot_events = len(df_impact.index)
    tot_activations = len(df_glofas.index)
    df_impact_copy = df_impact.copy()
    
    # Add buffer around the flood event dates to account for some uncertainty if desired
    df_impact_copy['start_date_buffer'] = pd.to_datetime(df_impact_copy["start_date"]) - timedelta(days=buffer)
    df_impact_copy['end_date_buffer'] = pd.to_datetime(df_impact_copy["end_date"]) + timedelta(days=buffer)

    for index, row in df_glofas.iterrows():
        TP_ = False
        act_dates = np.array(pd.date_range(row['start_date'], row['end_date']))

        for index,row in df_impact_copy.iterrows():
            event_dates = np.array(pd.date_range(row['start_date_buffer'], row['end_date_buffer']))
            
            if (set(act_dates) & set(event_dates)):
                TP+=1
                TP_ = True
                df_impact_copy = df_impact_copy.drop([index,])              
        if not TP_:
            FP+=1

    FN = tot_events - TP    
    return TP, FP, FN

def get_more_stats(TP, FP, FN):
    try:
        precision = TP / (TP + FP)
    except Exception as e: 
        precision = None
    recall = TP / (TP + FN)
    try:
        f1 = 2 / ((1/recall) + (1/precision))
    except Exception as e: 
        f1 = None
    return precision, recall, f1

def get_clean_stats_dict(df_glofas, df_impact, buffer):
    stats = {}
    TP, FP, FN =  get_detection_stats(df_glofas, df_impact, buffer)
    precision, recall, f1 = get_more_stats(TP, FP, FN)
    
    stats['TP'] = TP
    stats['FP'] = FP
    stats['FN'] = FN
    stats['precision'] = precision
    stats['recall'] = recall
    stats['f1'] = f1
    
    return stats 
```

Compare against the GloFAS reanalysis

```python
# Select the station and desired return period
THRESH_DAYS = 3
BUFFER = 30

detection_stats_all = {}

for code, station in stations_adm2.items(): 
    
    detection_stats = {}
    df_floodscan_event = floodscan_events[station]
    
    for rp, thresh in rp_dict[station].items():        
        df_glofas_act = get_glofas_activations(da_glofas_reanalysis[code], thresh, THRESH_DAYS)
        rp_stats = get_clean_stats_dict(df_glofas_act, df_floodscan_event, BUFFER)
        detection_stats[rp] = rp_stats

    # Convert dict to dataframe for plotting and accessibility
    df_detection_stats = (pd.DataFrame
                          .from_dict(detection_stats)
                          .transpose()
                          .reset_index()
                          .rename(columns={'index':'return_period'}))
    
    detection_stats_all[station] = df_detection_stats

    # Plot precision vs recall
    fig, ax = plt.subplots()
    plt.plot(df_detection_stats['return_period'], df_detection_stats['precision'], label='Precision')
    plt.plot(df_detection_stats['return_period'], df_detection_stats['recall'], label='Recall')
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Percent")
    ax.set_title(f'Glofas detection performance at {station}\n(reanalysis against Floodscan)')
    ax.legend()
    plt.savefig(PLOT_DIR / f'{station}_precision_recall_floodscan_reanalysis_b30.png')
```

Compare GloFAS reforecast against Floodscan

```python
THRESH_DAYS = 3
RP_ARR = [2,3, 5]
LEADTIMES = [5, 10, 15, 20, 25, 30]

df_detect_stats = pd.DataFrame(columns=['TP', 'FP', 'FN', 'precision', 'recall', 'f1', 'station', 'lead_time', 'return_period'])

for code, station in stations_adm2.items():
    
    df_floodscan_event = floodscan_events[station]

    # Calculate the detection performance
    for lt in LEADTIMES: 

        for rp in RP_ARR:

            thresh = rp_dict[station][rp] 
            da_glofas = da_glofas_reforecast_summary[code].sel(leadtime=lt)[['median']].to_array()[0]

            detection_stats = {}
            df_glofas_act = get_glofas_activations(da_glofas, thresh, THRESH_DAYS)
            stats = get_clean_stats_dict(df_glofas_act, df_floodscan_event, BUFFER)
            stats['station'] = station
            stats['lead_time'] = lt
            stats['return_period'] = rp
            df_detect_stats = df_detect_stats.append(stats, ignore_index=True)
            
    df_sel = df_detect_stats[df_detect_stats['station']==station]

    # Visualize detection performance
    fig, ax = plt.subplots()
    for i, rp in enumerate(RP_ARR):
        df = df_sel[df_sel['return_period']==rp]
        for cname, ls in zip(['precision', 'recall'], [':', '--']):
            ax.plot(df['lead_time'], df[cname], ls=ls, c=f'C{i}')
        ax.plot([], [], c=f'C{i}', label=f'{rp}-year RP')
    
    for cname, ls in zip(['precision', 'recall'], [':', '--']):
           ax.plot([], [], ls=ls, c='k', label=cname)
        
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Percent")
    ax.set_title(f'Glofas detection performance at {station}\n(reforecast against Floodscan)')
    ax.legend()
    plt.savefig(PLOT_DIR / f'{station}_precision_recall_floodscan_reforecast_b30.png')
```

Compare GloFAS reanalysis against reforecast

```python
THRESH_DAYS = 3
RP_ARR = [2, 3, 5]
LEADTIMES = [5, 10, 15, 20, 25, 30]

df_detect_stats = pd.DataFrame(columns=['TP', 'FP', 'FN', 'precision', 'recall', 'f1', 'station', 'lead_time', 'return_period'])

for code, station in stations_adm2.items(): 
    
    df_floodscan_event = floodscan_events[station]

    # Calculate the detection performance
    for lt in LEADTIMES: 

        for rp in RP_ARR:

            thresh = rp_dict[station][rp] 
            da_glofas = da_glofas_reforecast_summary[code].sel(leadtime=lt)[['median']].to_array()[0]
            
            detection_stats = {}
            df_glofas_act = get_glofas_activations(da_glofas, thresh, THRESH_DAYS)
            df_event = get_glofas_activations(da_glofas_reanalysis[code], thresh, THRESH_DAYS)
            stats = get_clean_stats_dict(df_glofas_act, df_event, 0)
            stats['station'] = station
            stats['lead_time'] = lt
            stats['return_period'] = rp
            df_detect_stats = df_detect_stats.append(stats, ignore_index=True)
            
    df_sel = df_detect_stats[df_detect_stats['station']==station]

    # Visualize detection performance
    fig, ax = plt.subplots()
    for i, rp in enumerate(RP_ARR):
        df = df_sel[df_sel['return_period']==rp]
        for cname, ls in zip(['precision', 'recall'], [':', '--']):
            ax.plot(df['lead_time'], df[cname], ls=ls, c=f'C{i}')
        ax.plot([], [], c=f'C{i}', label=f'{rp}-year RP')
    
    for cname, ls in zip(['precision', 'recall'], [':', '--']):
           ax.plot([], [], ls=ls, c='k', label=cname)
        
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Percent")
    ax.set_title(f'Glofas detection performance at {station}\n(reanalysis against reforecast)')
    ax.legend()
    plt.savefig(PLOT_DIR / f'{station}_precision_recall_reanalysis_reforecast_b0.png')
```

### Summarizing potential trigger options

```python
def summarize_trigger(df_events, station, code, rp, leadtime, duration, buffer):
    thresh = rp_dict[station][rp]
    glofas_vals = da_glofas_reforecast_summary[code].sel(leadtime=leadtime)[['median']].to_array()[0]   
    df_glofas_act = get_glofas_activations(glofas_vals, thresh, duration)
    stats = get_clean_stats_dict(df_glofas_act, df_events, buffer)
    return stats
```

```python
STATION = 'Nsanje'
CODE = 'glofas_1'
RP = 2
LT = 5
DUR = 3
```

```python
summarize_trigger(floodscan_events[STATION], STATION, CODE, RP, LT, DUR, 30)
```
