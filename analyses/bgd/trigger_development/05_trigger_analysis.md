```python
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, pearsonr
import numpy as np
import pandas as pd
import xarray as xr

from utils import utils
from importlib import reload
reload(utils)

mpl.rcParams['figure.dpi'] = 300

GLOFAS_VERSION = 2
LEADTIMES_V2 = [5, 10, 15, 20, 25, 30]
```

### Create GloFAS objects

```python
da_glofas_reanalysis =utils.get_glofas_reanalysis(version=GLOFAS_VERSION)
da_glofas_forecast =utils.get_glofas_forecast(version=GLOFAS_VERSION, leadtimes=LEADTIMES_V2)
da_glofas_forecast_summary =utils.get_da_glofas_summary(da_glofas_forecast)
da_glofas_reforecast =utils.get_glofas_reforecast(version=GLOFAS_VERSION, leadtimes=LEADTIMES_V2)
da_glofas_reforecast_summary =utils.get_da_glofas_summary(da_glofas_reforecast)

```

 ### Read in FFWC data

```python
df_ffwc_wl =utils.read_in_ffwc()
```

### Find the true positive events -- three days in a row above threshold

```python
WATER_THRESH = 19.5 + 0.85
#WATER_THRESH = 19.5 + 1

NDAYS_THRESH = 3

def get_groups_above_threshold(observations, threshold):
    return np.where(np.diff(np.hstack(([False], 
                                         observations > threshold, 
                                       [False]))))[0].reshape(-1, 2)
    
groups = get_groups_above_threshold(df_ffwc_wl['observed'], 
                                WATER_THRESH)

# Only take those that are 3 consecutive days
groups = [group for group in groups 
              if group[1] - group[0] >= NDAYS_THRESH]

# Mark the first date in each series as TP
events = [group[0] + NDAYS_THRESH - 1 for group in groups]
#eventss = np.concatenate([np.arange(group[0]+NDAYS_THRESH-1, 
#                                   group[1]) 
#                         for group in groups], axis=0)



df_ffwc_wl['event'] = False
df_ffwc_wl['event'][events] = True
```

### Add GloFAS to FFWC

```python
# Create final df
df_final = df_ffwc_wl.copy()

# Add glofas obs
df_glofas = da_glofas_reanalysis.to_dataframe()[[utils.STATION]].rename(columns={utils.STATION: 'glofas_observed'})
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

## Plots & info for slides


### Events

```python
# Print out events
df_final[['observed', 'glofas_observed']][df_final['event'] == True]
```

### GloFAS vs FFWC

```python
def plot_glofas_vs_ffwc(glofas_var_name, ylabel='GloFAS ERA5 river discharge [m$^3$ s$^{-1}$]'):
    xvar = 'observed'
    df = df_final[[xvar, glofas_var_name]].dropna()
    x = df[xvar]
    y = df[glofas_var_name]

    fig, ax = plt.subplots()
    ax.plot(x, y, '.', alpha=0.5)
    idx = df_final['event'] == True
    ax.plot(x[idx], y[idx], 'xr', ms=5, mfc='r', zorder=5)
    ax.set_xlabel('FFWC water level [m]')
    ax.set_ylabel(ylabel)

    split_val = 19.5
    idx = x < split_val
    print("Pearson's for above and below 19.5", pearsonr(x[idx], y[idx]),
    pearsonr(x[~idx], y[~idx]))
    ax.axvline(19.5, lw=0.3, c='k')
    
plot_glofas_vs_ffwc('glofas_observed')
```

### Glofas Event detection

```python
# Glofas event detection
GLOFAS_DETECTION_WINDOW = 7
def get_glofas_detections(glofas_var, thresh):
    groups = get_groups_above_threshold(glofas_var, thresh)
    return [group[0] for group in groups]

def get_detection_stats(glofas_var_name, thresh_array):
    nthresh = len(thresh_array)
    TP = np.empty(nthresh)
    FN = np.empty(nthresh)
    FP = np.zeros(nthresh)
    # Drop any NAs in the glofas columns
    df = df_final[['observed', 'event', glofas_var_name]]
    df = df[df[glofas_var_name].notna()]
    for ithresh, thresh in enumerate(thresh_array):
        detections = get_glofas_detections(df[glofas_var_name], thresh)
        detected_event_dates = []
        for detection in detections:
            detected_events = df['event'][detection:
                                    detection+GLOFAS_DETECTION_WINDOW]
            detected_event_dates += (
                list(detected_events[detected_events==True].index))
            if sum(detected_events) == 0:
                FP[ithresh] += 1
        # Get the unique event dates
        detected_event_dates = list(set(detected_event_dates))
        event_dates = df.index[df['event']]
        events_are_detected = [event_date in detected_event_dates for event_date in event_dates]
        TP[ithresh] = sum(events_are_detected)
        FN[ithresh] = sum([not event_detected for event_detected in events_are_detected])
    return TP, FP, FN

def get_more_stats(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / ((1/recall) + (1/precision))
    return precision, recall, f1
```

```python
def plot_stats(glofas_var_name, thresh_array, x_axis_units='[m$^3$ s$^{-1}$]'):
    TP, FP, FN = get_detection_stats(glofas_var_name,
                                thresh_array)
    precision, recall, f1 = get_more_stats(TP, FP, FN)

    # Plot results
    x = thresh_array
    fig, ax = plt.subplots()
    ax.plot(x, TP, label='TP')
    ax.plot(x, FP, label='FP')
    #ax.plot(x, FN, label='FN')
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
    
def get_thresh_and_max_precision_with_no_fn( glofas_var_name, thresh_array):
    TP, FP, FN = get_detection_stats(glofas_var_name,
                                thresh_array)
    precision, recall, f1 = get_more_stats(TP, FP, FN)
    print('threshold', thresh_array[FN==0][-5:])
    print('precision', precision[FN==0][-5:])
    print('FP', FP[FN==0][-5:])
    
def print_stats_for_val(glofas_var_name, thresh_array, trigger_val):
    TP, FP, FN = get_detection_stats(glofas_var_name,
                                thresh_array)
    i = np.argmin(np.abs(thresh_array - trigger_val))
    print(f'Stats for trigger value of {trigger_val}:')
    print(f'TP: {TP[i]:.0f}, FP: {FP[i]:.0f}, FN: {FN[i]:.0f}')
```

```python
thresh_array = np.arange(70000, 110000, 500)
glofas_var_name = 'glofas_observed'
plot_stats(glofas_var_name, thresh_array)
get_thresh_and_max_precision_with_no_fn(glofas_var_name, thresh_array)

# Print out last year's trigger stats -- 1 in 5 year val of 98000
print_stats_for_val(glofas_var_name, thresh_array, 98000)
```

From this plot, it looks like the highest possible threshold with no FN is around 83,000. 
thresh_array==83000


### Plot each year

```python
def plot_years(df_final, glofas_var, thresh, glofas_xlims):
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
            a = detection
            b = a + GLOFAS_DETECTION_WINDOW
            ax2.plot(x[a:b], y2[a:b], '-y', lw=3)
        iax += 1


plot_years(df_final, 'glofas_observed', 83000, (40000, 140000))


```

### Check advantage of using sum

```python

for n in range(2, 5):
    print(n)
    thresh_array = np.arange(n*70000, n*110000, n*500)
    df_final['glofas_observed_sum'] = df_final['glofas_observed'].rolling(n).sum()
    get_thresh_and_max_precision_with_no_fn('glofas_observed_sum', thresh_array)
    
    
df_final['glofas_observed_sum'] = df_final['glofas_observed'].rolling(3).sum()
plot_glofas_vs_ffwc('glofas_observed_sum', ylabel='GloFAS ERA5 3-day river volumne [m$^3$ s$^{-1}$]')
plot_years(df_final, 'glofas_observed_sum', 250000, (40000*3, 140000*3))

    
```

## Forecast eval

```python
# calculate the sums for the variables in question
for nday in [5, 10, 15, 20]:
    for var_suffix in ['1sig-', 'median', '1sig+']:
        var_name = f'glofas_{nday}day_{var_suffix}'
        df_final[f'{var_name}_sum'] = df_final[var_name].rolling(3).sum()
```

```python
from matplotlib.ticker import MaxNLocator

thresh_array = np.arange(60000, 110000, 500)
df_for_hassan = pd.DataFrame({'glofas_threshold': thresh_array})


#for var_type in ['', '_sum']:
for var_type in ['']: 
    fig, ax1 = plt.subplots()
    ax1.set_ylim(0, 20)
    #ax2 = ax1.twiny()
    i = 0
    ax1.axhline(5, c='k', lw=1)

    for nday in [5, 10, 15]:
        #for var_suffix in ['1sig-', 'median', '1sig+']:
        for var_suffix in ['median']:
            ax = ax1
            if var_type == '_sum':
                thresh_array *= 3
                ax = ax1
            var_name = f'glofas_{nday}day_{var_suffix}{var_type}'
            df = df_final[['observed', var_name]].dropna()
            x, y = df['observed'], df[var_name]
            idx = x > 19.5
            print(var_name, pearsonr(x[idx], y[idx]))
            TP, FP, FN = get_detection_stats(
                var_name, thresh_array)
            df_for_hassan[f'{nday}day_true_positive'] = TP
            df_for_hassan[f'{nday}day_false_positive'] = TP
            ax.plot(thresh_array, TP + i*0.05, label=f'{nday} day', c=f'C{i}')
            ax.plot(thresh_array, FP, '--', c=f'C{i}')
            i += 1
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)

df_for_hassan.to_csv('glofas_forecast_thresholds.csv', index=False)
```

```python
plot_years(df_final, 'glofas_10day_1sig-', 90000, (40000, 140000))
```

# Appendix


## GloFAS plot example

```python
year = "2020"
da_forecast_30 = (da_glofas_forecast
        .sel(time=slice(year, year),leadtime=30)
    )
da_ra = da_glofas_reanalysis.sel(time=slice(year, year))

fig, ax = plt.subplots(figsize=(15, 5))
for sigma in range(1,4):
    ax.fill_between(da_forecast_30.time, y1=np.percentile(da_forecast_30, norm.cdf(sigma) * 100, axis=0),
                    y2=np.percentile(da_forecast_30, (1 - norm.cdf(sigma)) * 100, axis=0),
                    alpha=0.3 / sigma, fc='b')
ax.plot(da_forecast_30.time, np.median(da_forecast_30, axis=0), c='b', label='forecast median')
ax.plot(da_ra.time, da_ra, c='k', label='reanalysis')
ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Water discharge (m^3 s^-1)')
plt.show()

```

## Is the GloFAS posterior Guassian?
Not really.

```python
from statsmodels.graphics.gofplots import qqplot
for data in da_glofas_forecast.sel(leadtime=5).values.T[:10]:
    qqplot((data - np.mean(data)) /np.std(data) , line='45')
    plt.xlim(-2.5, 2.5)
```

## Comparing the different FFWC datasets


### FFWC data
- Sent by Sazzad
- Goes from 2017-06-06 to 2019-10-15, but only in the rainy season (contains gaps)
- Only has observations at 06:00
- Has forecast (when available) for same date for lead time hours 24, 48, 72, 96, 120

```python
ffwc_dir =utils.FFWC_DIR
ffwc_discharge_filename = 'Bahadurabad_bsereved_discharge.xlsx'
ffwc_wl_filename = 'Bahadurabad_WL_forecast20172019.xlsx'
df_ffwc_discharge = pd.read_excel(ffwc_dir / ffwc_discharge_filename, 
                                 index_col='Date')

```

```python
ffwc_leadtimes = [1, 2, 3, 4, 5]

# For water level, need to combine the three sheets
df_ffwc_wl_dict = pd.read_excel(
        ffwc_dir / ffwc_wl_filename,
                                sheet_name=None,
                                header=[1], index_col='Date')
df_ffwc_wl = (df_ffwc_wl_dict['2017']
              .append(df_ffwc_wl_dict['2018'])
                        .append(df_ffwc_wl_dict['2019'])
                        .rename(columns={**{
                            f'{leadtime} hrs': leadtime
                            for leadtime in ffwc_leadtimes
                        }, **{'Observed WL': 'observed'}}
                        ))
# Convert date time to just date
df_ffwc_wl.index = df_ffwc_wl.index.floor('d')

# Reindex the date column and shift the predictions
# new_index = pd.date_range(df_ffwc_wl.index.min(), 
#              df_ffwc_wl.index.max() + np.timedelta64(5, 'D'))
# df_ffwc_wl = df_ffwc_wl.reindex(new_index)

# for leadtime in ffwc_leadtimes:
#     df_ffwc_wl[leadtime] = (
#         df_ffwc_wl[leadtime].shift(int(leadtime/24))
#     )
# df_ffwc_wl.dropna(how='all', inplace=True)

# df_ffwc_wl[df_ffwc_wl.index.year==2019]
```

### Old FFWC data 

- Used by Leonardo for initial trigger dev
- Goes from 1987-08-12 to 2019-07-29	
- Has observed daily average, but only when it is above 19.5 m (so there are many gaps)

```python
FFWC_RL_HIS_FILENAME='2020-06-07 Water level data Bahadurabad Upper danger level.xlsx'
ffwc_rl_name='{}/{}'.format(ffwc_dir,FFWC_RL_HIS_FILENAME)
df_ffwc_wl_old=pd.read_excel(ffwc_rl_name,index_col=0,header=0)
df_ffwc_wl_old.index=pd.to_datetime(df_ffwc_wl_old.index,format='%d/%m/%y')
df_ffwc_wl_old=df_ffwc_wl_old[['WL']].rename(columns={'WL': 
                                                    'observed'})
```

### Other FFWC data

- Hannah obtained this from Hassan
- observations recorded 5 times a day: 06:00, 09:00, 12:00, 15:00, 18:0
- other stations also available
- runs from 2016-01-01 to 2020-10-31

```python
# Not needed, doesn't contain Bahadarabad
#ffwc_full_data_filename = "FFWC_data.xls"
#ffwc_additional_data_filename = "SW46.9L_19-11-2020"
#df_ffwc_full_wl = pd.read_excel(ffwc_dir / ffwc_full_data_filename, 
#                                 index_col='DateTime')


ffwc_full_data_filename = 'SW46.9L_19-11-2020.xls'
df_ffwc_full_wl = (pd.read_excel(ffwc_dir / ffwc_full_data_filename, 
                                 index_col='DateTime')
                   .rename(columns={'WL(m)': 'observed'}))[['observed']]
```

### Comparison plot

```python
fig, ax = plt.subplots()
ax.plot(df_ffwc_full_wl.index, df_ffwc_full_wl['observed'], '.')
ax.plot(df_ffwc_wl.index, df_ffwc_wl['observed'], '.')
ax.plot(df_ffwc_wl_old.index, df_ffwc_wl_old['observed'], '.')
fig.autofmt_xdate()

ax.set_xlim(df_ffwc_wl.index[0], df_ffwc_wl.index[10])

```

### Full 30 year figure

```python
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df_final.index, df_final['observed'], 'o', 
        label='FFWC observations')
ax.plot(df_final.index, df_final['ffwc_5day'], 's', 
        alpha=0.5, label='FFWC 5 day forecast')
idx_event = df_final['event'] == True
ax.plot(df_final.index[idx_event], 
        df_final['observed'][idx_event], 
        'o', ms=10, c='r', label='Event')
ax.axhline(WATER_THRESH, c='k', lw=0.5)
fig.autofmt_xdate()
ax.set_ylim(19.5, None)
ax.set_xlim(df_final.index[50], None)
ax2 = ax.twinx()
ax2.plot(df_final['glofas_observed'], '-', alpha=0.5, c='g', 
         label='GloFAS observations')
ax.legend()
ax2.legend(loc=2)
```

### Attempting to really interpolate -- very slow

```python
from scipy import interpolate, integrate
y = df_final['glofas_observed']
n = len(y)
x = np.arange(n)
y = interpolate.interp1d(x, y)
integration_period = 3
#for i in range(n-integration_period):
#    #v = integrate.quad(y, i, i+integration_period)
    
    
y =  df_final['glofas_observed']
y.rolling(3).sum()
```

### Comparing differences

```python
df = df_final.copy()
x = df_final['observed'].values[1:] - df_final['observed'].values[:-1]
y= df_final['ffwc_5day'].values[1:] - df_final['ffwc_5day'].values[:-1]
plt.plot(x,y, '.')
```

### Exploring summed difference

```python
year = 2016
gvar = 'glofas_observed'
#gvar = 'glofas_15day_median'
df = df_final[['observed', gvar, 'event']].dropna()
y1 = df['observed'].diff(1).rolling(15).sum()
y2 = df[gvar].diff(1).rolling(15).sum()
x = df.index
fig, ax = plt.subplots()
ax.plot(x, y1)
ax.plot(x, y2/10000)
ax.set_xlim(datetime(year,1,1), datetime(year+1,1,1))
```

```python
idx2 = df['observed'] > 19.5

plt.plot(y2, y1, '.', alpha=0.5)
plt.plot(y2[idx2], y1[idx2], '.r', alpha=0.5)
l = np.polyfit(y2.dropna(), y1.dropna(), 1)
print(pearsonr(y2.dropna(), y1.dropna()))
x = np.arange(-20000, 20000)
plt.plot(x, l[1] + l[0]* x)
print(pearsonr(y2[idx2].dropna(), y1[idx2].dropna()))
plt.axhline(0, c='k')
```

```python
x = df['glofas_observed'].diff(1).values[1:]
y = df['observed'].diff(1).values[1:]
plt.plot(x,y,'.', alpha=0.2)
l = np.polyfit(x, y, 1)
pearsonr(y,x)

x = np.arange(-20000, 20000)
plt.plot(x, l[1] + l[0]* np.arange(-20000, 20000))
plt.xlim(-25500, 25000)
plt.ylim(-1,1)

```

```python
x = df['glofas_observed']
y = df['observed']
idx = df['event'] == True
plt.plot(x,y, '.')
plt.plot(x[idx], y[idx], 'ro')
```
