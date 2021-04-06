---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: pa-anticipatory-action
    language: python
    name: pa-anticipatory-action
---

```python
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding import glofas
from src.bangladesh import get_glofas_data as ggd

DATA_DIR = Path(os.environ['AA_DATA_DIR'])
GLOFAS_DIR = DATA_DIR / 'processed/bangladesh/GLOFAS_Data' 
STATION = 'Bahadurabad_glofas'
```

### Create GloFAS objects

```python
glofas_reanalysis = glofas.GlofasReanalysis(
    stations_lon_lat=ggd.FFWC_STATIONS
)
glofas_forecast = glofas.GlofasForecast(
    stations_lon_lat=ggd.FFWC_STATIONS, leadtime_hours=ggd.LEADTIME_HOURS
)
glofas_reforecast = glofas.GlofasReforecast(
    stations_lon_lat=ggd.FFWC_STATIONS, leadtime_hours=ggd.LEADTIME_HOURS
)
```

### Read in and interpolate data for station

```python
da_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3
    )[STATION]

def shift_dates(da_dict):
    return{leadtime_hour:
        da.assign_coords(time=da.time.values + np.timedelta64(
            int(leadtime_hour/24), 'D'))
        for leadtime_hour, da in da_dict.items()
        }

def interp_dates(da_dict):
    return {
        leadtime_hour:
    da.interp(
        time=pd.date_range(
           da.time.min().values, 
          da.time.max().values), 
          method='linear')
    for leadtime_hour, da
    in da_dict.items()
    }

da_glofas_forecast_dict = {leadtime_hour:
    glofas_forecast.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, 
        country_iso3=ggd.COUNTRY_ISO3, 
        leadtime_hour=leadtime_hour
    )[STATION]
    for leadtime_hour in ggd.LEADTIME_HOURS}
da_glofas_forecast_dict = shift_dates(da_glofas_forecast_dict)

da_glofas_reforecast_dict = {leadtime_hour:
    glofas_reforecast.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, 
        country_iso3=ggd.COUNTRY_ISO3, 
        leadtime_hour=leadtime_hour
    )[STATION]
    for leadtime_hour in ggd.LEADTIME_HOURS}
da_glofas_reforecast_dict = interp_dates(
    shift_dates(da_glofas_reforecast_dict))
```

### For forecast and reforecast, create summary data array
Contains median and 1,2,3 sigma centiles

```python
nsig_max = 3
percentile_dict = {
    **{'median': 50.},
    **{f'{n}sig+': norm.cdf(n) * 100 for n in range(1,nsig_max+1)},
    **{f'{n}sig-': (1-norm.cdf(n)) * 100 for n in range(1,nsig_max+1)},
}
coord_names = ["leadtime_hour", "time"]

def get_da_glofas_summary(da_glofas_dict):

    data_vars_dict = {var_name:
        (coord_names,
        np.array([
            np.percentile(da_glofas, percentile_value, axis=0)
            for da_glofas in da_glofas_dict.values()
        ]))
        for var_name, percentile_value in percentile_dict.items()}

    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(
            time=da_glofas_dict[120].time,
            leadtime_hour=ggd.LEADTIME_HOURS
        )
    )

da_glofas_forecast = get_da_glofas_summary(da_glofas_forecast_dict)
da_glofas_reforecast = get_da_glofas_summary(da_glofas_reforecast_dict)
```

 ### Read in FFWC data

```python
ffwc_dir = DATA_DIR / 'exploration/bangladesh/FFWC_Data'
```

```python
# Read in data from Sazzad that has forecasts
ffwc_wl_filename = 'Bahadurabad_WL_forecast20172019.xlsx'
ffwc_leadtime_hours = [24, 48, 72, 96, 120]

# Need to combine the three sheets
df_ffwc_wl_dict = pd.read_excel(
        ffwc_dir / ffwc_wl_filename,
                                sheet_name=None,
                                header=[1], index_col='Date')
df_ffwc_wl = (df_ffwc_wl_dict['2017']
              .append(df_ffwc_wl_dict['2018'])
                        .append(df_ffwc_wl_dict['2019'])
                        .rename(columns={
                            f'{leadtime_hour} hrs': f'ffwc_{int(leadtime_hour/24)}day'
                            for leadtime_hour in ffwc_leadtime_hours
                        })).drop(columns=['Observed WL']) # drop observed because we will use the mean later
# Convert date time to just date
df_ffwc_wl.index = df_ffwc_wl.index.floor('d')
```

```python
# Then read in the older data (goes back much futher)
FFWC_RL_HIS_FILENAME='2020-06-07 Water level data Bahadurabad Upper danger level.xlsx'
ffwc_rl_name='{}/{}'.format(ffwc_dir,FFWC_RL_HIS_FILENAME)
df_ffwc_wl_old=pd.read_excel(ffwc_rl_name,index_col=0,header=0)
df_ffwc_wl_old.index=pd.to_datetime(df_ffwc_wl_old.index,format='%d/%m/%y')
df_ffwc_wl_old
df_ffwc_wl_old=df_ffwc_wl_old[['WL']].rename(columns={'WL': 
                                                    'observed'})[df_ffwc_wl_old.index < df_ffwc_wl.index[0]]
df_ffwc_wl = pd.concat([df_ffwc_wl_old, df_ffwc_wl])
```

```python
# Read in the more recent file from Hassan
ffwc_full_data_filename = 'SW46.9L_19-11-2020.xls'
df_ffwc_wl_full = (pd.read_excel(ffwc_dir / ffwc_full_data_filename, 
                                 index_col='DateTime')
                   .rename(columns={'WL(m)': 'observed'}))[['observed']]

# Mutliple observations per day. Find mean and std
df_ffwc_wl_full['date'] = df_ffwc_wl_full.index.date
df_ffwc_wl_full = (df_ffwc_wl_full.groupby('date').agg(['mean', 'std'])
)['observed'].rename(columns={'mean': 'observed', 'std': 'obs_std'})
df_ffwc_wl_full.index = pd.to_datetime(df_ffwc_wl_full.index)


# Combine with first DF

df_ffwc_wl = pd.merge(df_ffwc_wl_full[['obs_std']], df_ffwc_wl, left_index=True, right_index=True, how='outer')
df_ffwc_wl.update(df_ffwc_wl_full, overwrite=False)
```

### Find the true positive events -- three days in a row above threshold

```python
WATER_THRESH = 19.5 + 0.85
NDAYS_THRESH = 3

def get_groups_above_threshold(observations, threshold):
    return np.where(np.diff(np.hstack(([False], observations > threshold, 
                                       [False]))))[0].reshape(-1, 2)
    
groups = get_groups_above_threshold(df_ffwc_wl['observed'], WATER_THRESH)
# Only take those that are 3 consecutive days
groups = [group for group in groups if group[1] - group[0] >= NDAYS_THRESH]

# Mark all the above dates as TP events
detections = np.concatenate([np.arange(group[0]+NDAYS_THRESH, group[1]) for group in groups], axis=0)
df_ffwc_wl['detection'] = False
df_ffwc_wl['detection'][detections] = True

```

```python
df_ffwc_wl.index[df_ffwc_wl['detection']].year.unique()
```

### Add GloFAS to FFWC

```python
glofas_columns = ['median', 
'1sig-', '2sig-', '3sig-', 
'1sig+', '2sig+', '3sig+']
df_final = df_ffwc_wl.copy()
for leadtime_hour in ggd.LEADTIME_HOURS:
    df_glofas_reforecast = da_glofas_reforecast.sel(leadtime_hour=leadtime_hour).to_dataframe()[glofas_columns]
    df_glofas_forecast = da_glofas_forecast.sel(leadtime_hour=leadtime_hour).to_dataframe()[glofas_columns]
    df_glofas = (pd.concat([df_glofas_reforecast, df_glofas_forecast])
                 .rename(columns={cname: f'glofas_{int(leadtime_hour/24)}day_{cname}' for cname in glofas_columns}))
    df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)
```

```python
d = df_final[df_final['detection']]
```

### Figure out important params

```python
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df_final.index, df_final['observed'], '.')
ax.plot(df_final.index, df_final['ffwc_5day'], 's', alpha=0.5)
idx_detection = df_final['detection'] == True
ax.plot(df_final.index[idx_detection], df_final['observed'][idx_detection], '.', c='r')
ax.axhline(WATER_THRESH, c='k', lw=0.5)
fig.autofmt_xdate()
ax.set_ylim(19.5, None)
ax.set_xlim(df_final.index[50], None)
ax2 = ax.twinx()
ax2.plot(df_final['glofas_5day_1sig-'], 'x', alpha=0.5, c='g')
```

```python
thresh=18
x = df_final['observed'].shift(-10)
idx = x > thresh
plt.errorbar(x[idx], df_final[idx]['glofas_15day_median'],
             yerr=(df_final[idx]['glofas_15day_median'] - df_final[idx]['glofas_15day_1sig-'], 
                            df_final[idx]['glofas_15day_1sig+'] - df_final[idx]['glofas_15day_median']),
                  xerr=df_final[idx]['obs_std'],
             ls='none', marker='.', c='y', alpha=0.5)
idx = (x > thresh) & (df_final['detection'])
plt.plot(x[idx], df_final[idx]['glofas_15day_median'], 'or', ms=5, mfc='r')
```

```python
df = df_final.copy()
x = df_final['observed'].values[1:] - df_final['observed'].values[:-1]
y= df_final['ffwc_5day'].values[1:] - df_final['ffwc_5day'].values[:-1]
plt.plot(x,y, '.')
```

# Appendix


## GloFAS plot example

```python
year = "2020"
da_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3
    )[STATION].sel(time=slice(year, year))
da_forecast_720 = (
        glofas_forecast.read_processed_dataset(
            country_name=ggd.COUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3, leadtime_hour=720
        )[STATION]
        .shift(time=30)
        .sel(time=slice(year, year))
    )
da_reforecast_720 = (
        glofas_reforecast.read_processed_dataset(
            country_name=ggd.COUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3, leadtime_hour=720
        )[STATION]
        .shift(time=30)
        .sel(time=slice(year, year)) 
    )
```

```python
fig, ax = plt.subplots(figsize=(15, 5))
for sigma in range(1,4):
    ax.fill_between(da_forecast_720.time, y1=np.percentile(da_forecast_720, norm.cdf(sigma) * 100, axis=0),
                    y2=np.percentile(da_forecast_720, (1 - norm.cdf(sigma)) * 100, axis=0),
                    alpha=0.3 / sigma, fc='b')
ax.plot(da_forecast_720.time, np.median(da_forecast_720, axis=0), c='b', label='forecast median')
ax.plot(da_reanalysis.time, da_reanalysis, c='k', label='reanalysis')
ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Water discharge (m^3 s^-1)')
plt.show()

```

## Is the GloFAS posterior Guassian?
Not really.

```python
from statsmodels.graphics.gofplots import qqplot
for data in da_glofas_forecast_dict[120].values.T[:10]:
    qqplot((data - np.mean(data)) /np.std(data) , line='45')
    plt.xlim(-2.5, 2.5)
```

```python

```

## Comparing the different FFWC datasets


### FFWC data
- Sent by Sazzad
- Goes from 2017-06-06 to 2019-10-15, but only in the rainy season (contains gaps)
- Only has observations at 06:00
- Has forecast (when available) for same date for lead time hours 24, 48, 72, 96, 120

```python
ffwc_dir = DATA_DIR / 'exploration/bangladesh/FFWC_Data'
ffwc_discharge_filename = 'Bahadurabad_bsereved_discharge.xlsx'
ffwc_wl_filename = 'Bahadurabad_WL_forecast20172019.xlsx'
df_ffwc_discharge = pd.read_excel(ffwc_dir / ffwc_discharge_filename, 
                                 index_col='Date')

```

```python
ffwc_leadtime_hours = [24, 48, 72, 96, 120]

# For water level, need to combine the three sheets
df_ffwc_wl_dict = pd.read_excel(
        ffwc_dir / ffwc_wl_filename,
                                sheet_name=None,
                                header=[1], index_col='Date')
df_ffwc_wl = (df_ffwc_wl_dict['2017']
              .append(df_ffwc_wl_dict['2018'])
                        .append(df_ffwc_wl_dict['2019'])
                        .rename(columns={**{
                            f'{leadtime_hour} hrs': leadtime_hour
                            for leadtime_hour in ffwc_leadtime_hours
                        }, **{'Observed WL': 'observed'}}
                        ))
# Convert date time to just date
df_ffwc_wl.index = df_ffwc_wl.index.floor('d')

# Reindex the date column and shift the predictions
# new_index = pd.date_range(df_ffwc_wl.index.min(), 
#              df_ffwc_wl.index.max() + np.timedelta64(5, 'D'))
# df_ffwc_wl = df_ffwc_wl.reindex(new_index)

# for leadtime_hour in ffwc_leadtime_hours:
#     df_ffwc_wl[leadtime_hour] = (
#         df_ffwc_wl[leadtime_hour].shift(int(leadtime_hour/24))
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
ffwc_rl_name='{}/{}'.format(FFWC_RL_FOLDER,FFWC_RL_HIS_FILENAME)
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
