```python
from importlib import reload
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats import norm, pearsonr


from utils import utils
reload(utils)

mpl.rcParams['figure.dpi'] = 300
```

### Create GloFAS objects

```python
da_glofas_reanalysis = utils.get_glofas_reanalysis()
da_glofas_forecast = utils.get_glofas_forecast()
da_glofas_forecast_summary = utils.get_da_glofas_summary(da_glofas_forecast)
da_glofas_reforecast = utils.get_glofas_reforecast()
da_glofas_reforecast_summary = utils.get_da_glofas_summary(da_glofas_reforecast)

```

 ### Read in FFWC data

```python
df_ffwc_wl = utils.read_in_ffwc()
```

### Find the true positive events -- three days in a row above threshold

```python
WATER_THRESH = 19.5 + 0.85
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
                 .rename(columns={cname: f'glofas_{leadtime}day_{cname}' for cname in glofas_columns}))
    df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)

# Any event elements that are NA should be False    
df_final['event'] = df_final['event'].fillna(False)
    
# Don't bother starting before FFWC observations do
df_final = df_final[df_final.index >= df_ffwc_wl.index[0]]

```

### Read in FFWC discharge data

```python
ffwc_discharge_filename = 'bahadurabad_discharge_01.xlsx'
df_ffwc_discharge = pd.read_excel(utils.FFWC_DIR / ffwc_discharge_filename,
                                  index_col='Date').rename(
    columns={'Discharge (m3/s)': 'ffwc_discharge'})
df_discharge = pd.merge(
    df_final, df_ffwc_discharge,
    how='inner',
    left_index=True, right_index=True)
```

### Plot GloFAS vs FFWC discharge

```python
fig, ax = plt.subplots(figsize=(5,5))
y = df_discharge['ffwc_discharge']
x = df_discharge['glofas_observed']
ax.plot(x, y, '.', alpha=0.5)

l = np.arange(-10000, 200000, 10000)
ax.plot(l, l, c='r', label='Ideal relation')

m_gf, b_gf = np.polyfit(x, y, 1)
plt.plot(l,   m_gf * l+ b_gf, label='Line of best fit')


ax.set_xlim(0, 150000)
ax.set_ylim(0, 150000)
ax.set_ylabel('FFWC water discharge (m^3 s^-1)')
ax.set_xlabel('GloFAS water discharge (m^3 s^-1)')

ax.legend()

```

```python
x = df_discharge['ffwc_discharge']
y = df_discharge['glofas_observed']
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, (y-x)/x, '.', alpha=0.5)
ax.axhline(0, c='k')
ax.set_xlabel('FFWC water discharge (m$^3$ s$^{-1}$)')
ax.set_ylabel('Fractional difference [(G-F)/F]')

fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.plot(x, (y-x), '.', alpha=0.5)
ax2.axhline(0, c='k')
ax2.set_xlabel('FFWC water discharge (m$^3$ s$^{-1}$)')
ax2.set_ylabel('GloFAS - FFWC (m$^3$ s$^{-1}$)')
ax2.set_ylim(-70000, 70000)
```

```python
# For GloFAS discharge values > 50000:
thresh = 50000
df_discharge_high = df_discharge[df_discharge['ffwc_discharge'] > thresh]

bins = np.arange(-2, 2, 0.1)
y = df_discharge_high['glofas_observed'] 
x = df_discharge_high['ffwc_discharge']
z = (y-x)/x
plt.hist(z, bins=bins)
```

### Plot FFWC river discharge against water level

```python
xvar = 'observed'
yvar = 'ffwc_discharge'
df = df_discharge[[xvar, yvar]].dropna()
x = df[xvar]
y = df[yvar]

fig, ax = plt.subplots()
ax.plot(x, y, '.', alpha=0.5)
ax.set_xlabel('FFWC water level [m]')
ylabel='FFWC river discharge [m$^3$ s$^{-1}$]'
ax.set_ylabel(ylabel)

split_val = 19.5
idx = x < split_val
print("Pearson's for above and below 19.5", pearsonr(x[idx], y[idx]),
pearsonr(x[~idx], y[~idx]))

ax.axvline(19.5, lw=0.3, c='k')
m3, m2, m1, b = np.polyfit(x[idx], y[idx], 3)
xplot = np.arange(12, 22)
plt.plot(xplot, m3 * xplot**3 + m2 * xplot**2 +  m1*xplot + b)


# Get a function to convert from discharge to WL
from scipy.interpolate import interp1d
y = np.arange(11.6, 25, 0.01)
x = m3 * y **3 + m2 * y**2 + m1 * y + b
f_discharge_to_wl = interp1d(x, y)
plt.plot(f_discharge_to_wl(x), x)

```

Using the line of best fit in the figure above, derive a relation to convert river discharge values to water level values. Apply this relation to the GloFAS river discharge data to get a GloFAS-predicted water level. Then use the GloFAS water level to define flooding events as is done in FFWC, and compare.

```python
# Convert GloFAS to FFWC using line
def convert_glofas_discharge_to_ffwc_wl(glofas_discharge):
    return  f_discharge_to_wl(m_gf * glofas_discharge + b_gf)

df_final['wl_estimate'] = convert_glofas_discharge_to_ffwc_wl(df_final['glofas_observed'])

fig, ax = plt.subplots()
ax.plot(df_final['observed'], df_final['wl_estimate'], '.', alpha=0.5)

```

```python
# Glofas event detection
GLOFAS_DETECTION_WINDOW = 7
def get_glofas_detections(glofas_var, thresh):
    groups = get_groups_above_threshold(glofas_var, thresh)

    # Only take those that are 3 consecutive days
    groups = [group for group in groups 
                  if group[1] - group[0] >= NDAYS_THRESH]
    
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
        detection_window_min = 0
        detection_window_max = 7
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
```

```python
thresh_array = np.arange(19.0, 22.0, 0.05)
glofas_var_name = 'wl_estimate'
plot_stats(glofas_var_name, thresh_array, x_axis_units='[m]')
```

```python
def plot_years(df_final, glofas_var, thresh,):
    fig, axs = plt.subplots(17, 2, figsize=(10, 30))
    fig.autofmt_xdate()
    axs = axs.flat
    df_final['year'] = df_final.index.year
    iax = 0
    for year, df in df_final.groupby('year'):
        if year == 2021:
            break
        ax = axs[iax]
        x = df.index
        y1 = df['observed']
        ax.plot(x, y1, '.')
        idx_event = df['event'] == True
        ax.plot(x[idx_event], y1[idx_event], 'or', ms=5)
        ax.set_xlim(datetime(year, 3, 1), datetime(year, 10, 31))
        ax.set_ylim(19.0, 21.5)
        ax.set_title(year, pad=0)

        y2 = df[glofas_var]
        ax.plot(x, y2, '-g')
        # Glofas detections
        for detection in get_glofas_detections(y2, thresh):
            a = detection
            b = a + GLOFAS_DETECTION_WINDOW
            ax.plot(x[a:b], y2[a:b], '-y', lw=3)
        iax += 1
        
plot_years(df_final, 'wl_estimate', 20.35)

```

```python
# Figure out which river discharge value corresponds to the danger level
convert_glofas_discharge_to_ffwc_wl(89000)
```

## New event detection method

- We know that the discharge / WL relationship breaks down exactly above the danger level of 19.5 m
- New algorithm:
    - Use FFWC forecast to check if water is above danger level 5 days in advance
    - If yes, only then examine GloFAS forecast
    
With this method we can perhaps increase the correspondence between GloFAS and FFWC


First check how quickly WL increases during events. If FFWC does not reach the danger level around 10 or 15 days before, then we can't use this method

```python
for event_date in df_final[df_final['event'] == True].index:
    date_delta = timedelta(days=15)
    date_range = np.arange(event_date - date_delta, event_date + date_delta,
                          timedelta(days=1))
    df = df_final.loc[date_range]
    print(event_date)
    print(df['observed'])
```

How many days before event is danger level reached:
- 5
- 4
- 7
- 4
- 3
- 4
- (2nd event of 2020 stayed above danger level from previous event)

Unfortunately the DL is not reached early enough. We could potentially use a lower value, but we would need more FFWC data to perform this analysis.
