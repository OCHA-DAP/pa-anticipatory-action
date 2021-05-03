```python
from importlib import reload
from pathlib import Path
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

import read_in_data as rd
reload(rd)

mpl.rcParams['figure.dpi'] = 200


DATA_DIR = Path(os.environ["AA_DATA_DIR"])
SKILL_DIR = DATA_DIR / 'exploration/bangladesh/GLOFAS_Data'
SKILL_FILE = 'forecast_skill.csv'
LEADTIMES_V2 = [5, 10, 15, 20, 25, 30]
MAIN_VERSION = 3
```

### Create GloFAS objects

```python
da_glofas_reanalysis = {
    2: rd.get_glofas_reanalysis(version=2),
    3: rd.get_glofas_reanalysis()
}

da_glofas_forecast = {
    2: rd.get_glofas_forecast(version=2, leadtimes=LEADTIMES_V2),
    3: rd.get_glofas_forecast(),

}

da_glofas_reforecast = {
    2: rd.get_glofas_reforecast(version=2, interp=False, leadtimes=LEADTIMES_V2),
    3: rd.get_glofas_reforecast(interp=False)
}

da_glofas_reforecast_interp = {
    2: rd.get_glofas_reforecast(version=2, leadtimes=LEADTIMES_V2),
    3: rd.get_glofas_reforecast()
}
```

### Calculate return periods

```python
def get_return_period_function(observations):
    df_rp = (observations.to_dataframe()[[rd.STATION]]
                 .rename(columns={rd.STATION: 'discharge'})
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
for version in [2,3]:
    f_rp = get_return_period_function(da_glofas_reanalysis[version])
    print(f'Version {version}')
    for year in [1.5, 2, 3, 4, 5, 10, 20]:
        val = 5000*np.round(f_rp(year) / 5000)
        print(year, val)
        if version == MAIN_VERSION:
            rp_dict[year] = val
# Make 1 in 5 year always 100,000
rp_dict[4] = 95000
rp_dict[5] = 100000
```

 ### Read in FFWC data with events

```python
df_ffwc_wl = rd.get_events(rd.read_in_ffwc())
```

### Add GloFAS to FFWC

```python
da_glofas_forecast_summary = rd.get_da_glofas_summary(da_glofas_forecast[MAIN_VERSION])
da_glofas_reforecast_summary = rd.get_da_glofas_summary(da_glofas_reforecast_interp[MAIN_VERSION])


# Create final df
df_final = df_ffwc_wl.copy()

# Add glofas obs
df_glofas = da_glofas_reanalysis[MAIN_VERSION].to_dataframe()[[rd.STATION]].rename(columns={rd.STATION: 'glofas_observed'})
df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)

# Add glofas forecasts
glofas_columns = ['median',
'1sig-', '2sig-', '3sig-', 
'1sig+', '2sig+', '3sig+']
for leadtime in da_glofas_reforecast_summary.leadtime:
    df_glofas_reforecast = da_glofas_reforecast_summary.sel(leadtime=leadtime).to_dataframe()[glofas_columns]
    df_glofas_forecast = da_glofas_forecast_summary.sel(leadtime=leadtime).to_dataframe()[glofas_columns]
    df_glofas = (pd.concat([df_glofas_reforecast, df_glofas_forecast])
                 .rename(columns={cname: f'glofas_{int(leadtime)}day_{cname}' for cname in glofas_columns}))
    df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)
    
# Any event elements that are NA should be False    
df_final['event'] = df_final['event'].fillna(False)

# Don't bother starting before FFWC observations do
df_final = df_final[df_final.index >= df_ffwc_wl.index[0]]
```

### Event detection

```python
# Glofas event detection
GLOFAS_DETECTION_WINDOW_AHEAD = 30
GLOFAS_DETECTION_WINDOW_BEHIND = 5

GLOFAS_MIN_DAYS_ABOVE_THRESH = 3

def get_glofas_detections(glofas_var, thresh):
    groups = rd.get_groups_above_threshold(glofas_var, thresh)
    return [group[0] + GLOFAS_MIN_DAYS_ABOVE_THRESH - 1 for group in groups
            if group[1] - group[0] >= GLOFAS_MIN_DAYS_ABOVE_THRESH   
           ]

def get_detection_stats(df_final, glofas_var_name, thresh_array, use_glofas_events=False,
                       window_behind=GLOFAS_DETECTION_WINDOW_BEHIND,
                       window_ahead=GLOFAS_DETECTION_WINDOW_AHEAD):
    df_ds = pd.DataFrame(data={'thresh': thresh_array, 'TP': 0, 'FN': 0, 'FP': 0, 'day_offset': None})
    nthresh = len(thresh_array)
    # Drop any NAs in the glofas columns
    if use_glofas_events:
        df = df_final[['glofas_observed', glofas_var_name]]
    else:
        df = df_final[['observed', 'event', glofas_var_name]]
        event_var_name = 'event'
    df = df[df[glofas_var_name].notna()]
    for ithresh, row in df_ds.iterrows():
        if use_glofas_events:
            groups = get_glofas_detections(df['glofas_observed'], row['thresh'])
            events = [group + GLOFAS_MIN_DAYS_ABOVE_THRESH - 1 for group in groups]
            event_var_name = f'event_glofas_{ithresh}'
            df[event_var_name] = False
            df[event_var_name][events] = True
        detections = get_glofas_detections(df[glofas_var_name], row['thresh'])
        detected_event_dates = []
        day_offset = {}
        for detection in detections:
            detected_events = df[event_var_name][detection-window_behind:
                                    detection+window_ahead]
            detected_event_dates += (
                list(detected_events[detected_events==True].index))
            if sum(detected_events) == 0:
                df_ds.at[ithresh, 'FP'] += 1
            # Get index where events are detected
            idx_detection = np.where(detected_events)[0]
            for idx in idx_detection:
                date_of_event = detected_events.index[idx]
                if date_of_event not in day_offset:
                    day_offset[date_of_event] = []
                day_offset[date_of_event] += [idx - window_behind]
        # Get the unique event dates
        detected_event_dates = list(set(detected_event_dates))
        event_dates = df.index[df[event_var_name]]
        events_are_detected = [event_date in detected_event_dates for event_date in event_dates]
        df_ds.at[ithresh, 'TP'] = sum(events_are_detected)
        df_ds.at[ithresh, 'FN'] = sum([not event_detected for event_detected in events_are_detected])
        df_ds.at[ithresh, 'day_offset'] = day_offset.copy()
    # Take the minimum value for each event detection
    df_ds['day_offset_reduced'] = df_ds['day_offset'].apply(lambda x: [value[0] for value in x.values() if x is not None])
    return df_ds

def get_more_stats(df):
    df['precision'] = df['TP'] / (df['TP'] + df['FP'])
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    df['f1'] = 2 / ((1/df['recall']) + (1/df['precision']))
    return df


```

```python
def plot_stats(df_final, glofas_var_name, thresh_array, x_axis_units='[m$^3$ s$^{-1}$]'):
    df_ds = get_detection_stats(df_final, glofas_var_name,
                                thresh_array)
    df_ds = get_more_stats(df_ds)

    # Plot results
    x = thresh_array
    fig, ax = plt.subplots()
    ax.plot(x, df_ds['TP'], label='TP')
    ax.plot(x, df_ds['FP'], label='FP')
    #ax.plot(x, df_ds['FN'], label='FN')
    df = df_final[['observed', 'event', glofas_var_name]]
    df = df[df[glofas_var_name].notna()]
    ax.axhline(len(df[df['event']]), c='C0', ls='--')
    #ax.axhline(0, c='C2', ls='--')
    #ax2 = ax.twinx()
    #ax2.plot(x, precision, label='precision', c='r')
    #ax2.plot(x, recall, label='recall', c='y')
    #ax2.plot(x, f1, label='F1', c='k')
    #ax2.set_ylim(-0.05, 1.05)
    ax.legend(loc=2)
    #ax2.legend(loc=0)
    #ax2.axhline(0.2, c='r', ls='--')

    ax.set_xlabel(f'GloFAS trigger threshold {x_axis_units}')
    ax.set_ylabel('Number')
    
    
def print_stats_for_val(df_final, glofas_var_name, thresh_array, trigger_val):
    df_ds = get_detection_stats(df_final, glofas_var_name,
                                thresh_array)
    i = np.argmin(np.abs(thresh_array - trigger_val))
    print(f'Stats for trigger value of {trigger_val}:')
    print(f'TP: {df_ds["TP"][i]:.0f}, FP: {df_ds["FP"][i]:.0f}, FN: {df_ds["FN"][i]:.0f}')
    
def plot_offset_days(df_final, glofas_var_name, thresh_array):
    df_ds = get_detection_stats(df_final, glofas_var_name, thresh_array)
    fig, ax = plt.subplots()
    for i, row in df_ds.iterrows():
        y = row['day_offset_reduced']
        x = [row['thresh']] * len(y)
        ax.plot(x, y, '.', c='C0', alpha=0.2)
        if row['thresh'] == 97000:
            ax.plot(x, y, '.', c='C1', alpha=0.4)
    ax.axhline(0, c='k', ls=':', lw=0.5, alpha=0.5)
    ax.set_ylabel('Detection day offset')
    x_axis_units='[m$^3$ s$^{-1}$]'
    ax.set_xlabel(f'GloFAS trigger threshold {x_axis_units}')
```

```python
thresh_array = np.arange(70000, 110000, 500)
glofas_var_name = 'glofas_observed'

plot_stats(df_final, glofas_var_name, thresh_array)
plot_offset_days(df_final, glofas_var_name, thresh_array)

# Print out last year's trigger stats for 1 in 5 year val
print_stats_for_val(df_final, glofas_var_name, thresh_array, rp_dict[5])


```

```python
def plot_years(df_final, thresh, glofas_var='glofas_observed', glofas_xlims=(40000, 140000), forecast_var=None):
    fig, axs = plt.subplots(17, 2, figsize=(10, 30))
    fig.autofmt_xdate()
    axs = axs.flat
    df_final['year'] = df_final.index.year
    iax = 0
    for year, df in df_final.groupby('year'):
        if year == 2021:
            break
        ax1 = axs[iax]
        x = df.index
        y1 = df['observed']
        ax1.plot(x, y1, '.')
        idx_event = df['event'] == True
        ax1.plot(x[idx_event], y1[idx_event], 'or', ms=5)
        ax1.set_xlim(datetime(year, 3, 1), datetime(year, 10, 31))
        ax1.set_ylim(19.0, 21.5)
        ax1.set_title(year, pad=0)
        ax2 = ax1.twinx()
        y2 = df[glofas_var]
        ax2.plot(x, y2, '-g')
        ax2.set_ylim(glofas_xlims)
        # Glofas detections
        for detection in get_glofas_detections(y2, thresh):
            a = detection - GLOFAS_DETECTION_WINDOW_BEHIND
            b = detection + GLOFAS_DETECTION_WINDOW_AHEAD
            ax2.plot(x[a:b], y2[a:b], '-y', lw=3)
            ax2.plot(x[detection], y2[detection], 'x', c='m')   
        if forecast_var is not None:
            y3 = df[forecast_var]
            ax2.plot(x, y3, '-g', alpha=0.5)
            for detection in get_glofas_detections(y3, thresh):
                a = detection - GLOFAS_DETECTION_WINDOW_BEHIND
                b = detection + GLOFAS_DETECTION_WINDOW_AHEAD
                ax2.plot(x[a:b], y3[a:b], '-c', lw=3, alpha=0.5)
                ax2.plot(x[detection], y3[detection], '^', c='m')
        iax += 1


plot_years(df_final, rp_dict[5])

```

### Forecasts

```python
leadtimes = [5, 10, 11, 12, 13, 14, 15, 20, 25, 30]
thresh_list = [
    80000,
    #5000,
    90000,
    #5000,
    100000
]
df_forecast_dict = {}
var = 'median'
#var = '1sig-'

for thresh in thresh_list:
    df_forecast = pd.DataFrame(data={'leadtime': leadtimes, 'TP': 0, 'FP': 0, 'FN': 0})
    for irow, row in df_forecast.iterrows():
        glofas_var_name = f'glofas_{row["leadtime"]}day_{var}'
        df_ds = get_detection_stats(df_final, glofas_var_name, [thresh], use_glofas_events=True)
        for q in ['TP', 'FP',  'FN']:
            df_forecast.at[irow, q] = df_ds[q][0]
    df_forecast['precision'] = df_forecast['TP'] / (df_forecast['TP'] + df_forecast['FP'])
    df_forecast['recall'] = df_forecast['TP'] / (df_forecast['TP'] + df_forecast['FN'])
    df_forecast_dict[thresh] = df_forecast

    
    
for ls_dict, offset_frac in zip([
    {'TP': '-', 'FP': '--', 'FN': ':'},
    {'precision': '-', 'recall': '--'}
],
    [0.1, 0.01]
):
    fig, ax = plt.subplots()
    i = 0
    for thresh, df in df_forecast_dict.items():
        for iq, q in enumerate(ls_dict.keys()):
            ax.plot(df['leadtime'], df[q] + offset_frac * i + offset_frac /2 * iq, 
                    ls=ls_dict[q], marker='o',  c=f'C{i}')
        ax.plot([], [], c=f'C{i}', label=f'{thresh} m$^3$ s$^-1$')
        i += 1
    for q in ls_dict.keys():
        ax.plot([], [], c='k', ls=ls_dict[q], label=q)
    ax.legend()
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel('Number')
```

```python
# Between 10 and 15 days, the number of TP and FN is the same.
# Only difference is number of FP:
df_forecast_dict[100000][['leadtime', 'FP']]
```

```python
thresh = 80000
fig, axs = plt.subplots(3, figsize=(8,6))
df = df_final[(df_final.index > datetime(1999, 1, 1)) & (df_final.index < datetime(2019, 1, 1))]
for i, q in enumerate(['glofas_observed', 'glofas_10day_median', 'glofas_15day_median'][::-1]):
    if i == 2:
        c1 = 'k'
        c2 = 'C3'
    else:
        c1 = 'C0'
        c2 = 'C1'
    ax = axs[i]
    x = df.index
    y = df[q]
    ax.plot(x, y, '-', c=c1, lw=0.5)
    ax.set_ylim(0, 120000)
    #ax.set_xlim(0, 120000)
    ax.minorticks_on()
    ax.axhline(thresh, c='r', lw=0.5)
    for detection in get_glofas_detections(y, thresh):
        a = detection - GLOFAS_DETECTION_WINDOW_BEHIND
        b = detection + GLOFAS_DETECTION_WINDOW_AHEAD
        #ax.plot(x[a:b], y[a:b], '-r', lw=1, alpha=0.5)
        ax.plot(x[detection], y[detection], '.', c=c2)

```

```python
df_final
```
