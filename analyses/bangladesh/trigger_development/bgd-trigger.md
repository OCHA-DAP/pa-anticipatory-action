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

 ### Read in FFWC data

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
```

### Find the events -- three days in a row above threshold

```python
WATER_THRESH = 19.5 + 0.85
NDAYS_THRESH = 3

def get_groups_above_threshold(observations, threshold):
    return np.where(np.diff(np.hstack(([False], observations > threshold, 
                                       [False]))))[0].reshape(-1, 2)
    
groups = get_groups_above_threshold(df_ffwc_wl['observed'], WATER_THRESH)
group_len = [group[1] - group[0] for group in groups]
groups = [group for i, group in enumerate(groups) if group_len[i] >= NDAYS_THRESH]
```

```python
groups
```

```python
fig, ax = plt.subplots()
ax.plot(df_ffwc_wl.index, df_ffwc_wl['observed'], '.')
for group in groups:
    a, b = group[0], group[1]
    ax.plot(df_ffwc_wl.index[a:b], df_ffwc_wl['observed'][a:b], '.', c='r')
fig.autofmt_xdate()

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
