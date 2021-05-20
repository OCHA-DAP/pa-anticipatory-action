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
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
```

### Read in GloFAS data

```python
da_glofas_reanalysis = {}
da_glofas_reforecast = {}
da_glofas_forecast = {}
da_glofas_forecast_summary = {}
da_glofas_reforecast_summary = {}

for station in STATIONS: 
    da_glofas_reanalysis[station] = rd.get_glofas_reanalysis(version=GLOFAS_VERSION, station=station)
    da_glofas_reforecast[station] = rd.get_glofas_reforecast(version=GLOFAS_VERSION, station=station)
    da_glofas_forecast[station] = rd.get_glofas_forecast(version=GLOFAS_VERSION, station=station)
    da_glofas_forecast_summary[station] = rd.get_da_glofas_summary(da_glofas_forecast[station])
    da_glofas_reforecast_summary[station] = rd.get_da_glofas_summary(da_glofas_reforecast[station])
```

### Read in the baseline impact data

```python
df_mvac_flood_ta = pd.read_csv(os.path.join(config.DATA_PRIVATE_DIR, 'processed', 'mwi', 'mvac_dodma_flood_ta.csv'))
df_floodscan_event = pd.read_csv(EXPLORE_DIR / 'floodscan_event_summary.csv')
```

```python
# Add buffer around floodscan dates to account for some uncertainty
df_floodscan_event['start_date_buffer'] = pd.to_datetime(df_floodscan_event["start_date"]) - timedelta(days=30)
df_floodscan_event['end_date_buffer'] = pd.to_datetime(df_floodscan_event["end_date"]) + timedelta(days=30)
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

for station in STATIONS:
    f_rp = get_return_period_function(da_glofas_reanalysis[station], station)
    rp_dict[station] = {}
    for year in [1.5, 2, 3, 4, 5, 10, 20]:
        val = 10*np.round(f_rp(year) / 10)
        rp_dict[station][year] = val

df_rps = pd.DataFrame(rp_dict)
```

### Overview of historical discharge

```python
# Return periods to focus on, with display colours
rps = {
    3: '#32a852',
    5: '#9c2788'
}

for station in STATIONS: 
    da_plt = da_glofas_reanalysis[station].sel(time=slice('1999-01-01','2020-12-31'))
    df_flood_mvac = df_mvac_flood_ta[df_mvac_flood_ta['name']==station]

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

def get_detection_stats(df_glofas, df_impact, buffer=True):
    TP = 0 
    FP = 0
    tot_events = len(df_impact.index)
    tot_activations = len(df_glofas.index)

    for index, row in df_glofas.iterrows():

        TP_ = False
        act_dates = np.array(pd.date_range(row['start_date'], row['end_date']))

        for index,row in df_impact.iterrows():
            if buffer:
                event_dates = np.array(pd.date_range(row['start_date_buffer'], row['end_date_buffer']))
            else:
                event_dates = np.array(pd.date_range(row['start_date'], row['end_date']))
            if (set(act_dates) & set(event_dates)):
                TP+=1
                TP_ = True
                df_impact = df_impact.drop([index,])
                

        if not TP_:
            FP+=1

    FN = tot_events - TP 
    
    return TP, FP, FN

def get_more_stats(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / ((1/recall) + (1/precision))
    return precision, recall, f1
```

```python
# Select the station and desired return period

for station in STATIONS:
    
    dur = 3
    detection_stats = {}

    for rp in df_rps.index:

        rp_stats = {}

        # Get the indices from where the threshold would have been met
        vals = da_glofas_reanalysis[station].values
        groups = get_groups_above_threshold(vals, df_rps.at[rp, station])
        groups = [group for group in groups if group[1] - group[0] >= dur]
        groups_fill = [np.arange(group[0], group[1], 1) for group in groups]

        # Convert to more readable format
        df_glofas_act = pd.DataFrame(groups, columns=['start_index', 'end_index'])
        df_glofas_act['num_days'] = df_glofas_act['end_index'] - df_glofas_act['start_index']
        df_glofas_act['start_date'] = df_glofas_act['start_index'].apply(lambda x: da_glofas_reanalysis[station].time[x].values)
        df_glofas_act['end_date'] = df_glofas_act['end_index'].apply(lambda x: da_glofas_reanalysis[station].time[x].values)

        # Get the output statistics of hits vs misses
        TP, FP, FN =  get_detection_stats(df_glofas_act, df_floodscan_event, True)
        precision, recall, f1 = get_more_stats(TP, FP, FN)

        rp_stats['TP'] = TP
        rp_stats['FP'] = FP
        rp_stats['FN'] = FN
        rp_stats['precision'] = precision
        rp_stats['recall'] = recall
        rp_stats['f1'] = f1

        detection_stats[rp] = rp_stats

    # Convert dict to dataframe for plotting and accessibility
    df_detection_stats = (pd.DataFrame
                          .from_dict(detection_stats)
                          .transpose()
                          .reset_index()
                          .rename(columns={'index':'return_period'}))

    # Plot precision vs recall
    fig, ax = plt.subplots()
    plt.plot(df_detection_stats['return_period'], df_detection_stats['precision'], label='Precision')
    plt.plot(df_detection_stats['return_period'], df_detection_stats['recall'], label='Recall')
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Percent")
    ax.set_title(f'GloFAS reanalysis detection performance\nacross return period thresholds at {station}')
    ax.legend()
    plt.savefig(PLOT_DIR / f'{station}_precision_recall.png')
```
