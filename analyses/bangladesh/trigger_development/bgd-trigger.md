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

# Reindex the date column and shift the predictions
# new_index = pd.date_range(df_ffwc_wl.index.min(), 
#              df_ffwc_wl.index.max() + np.timedelta64(5, 'D'))
# df_ffwc_wl = df_ffwc_wl.reindex(new_index)

# for leadtime_hour in ffwc_leadtime_hours:
#     df_ffwc_wl[leadtime_hour] = (
#         df_ffwc_wl[leadtime_hour].shift(int(leadtime_hour/24))
#     )
# df_ffwc_wl.dropna(how='all', inplace=True)

```

```python
df_ffwc_wl[df_ffwc_wl.index.year==2019]
```

# Appendix


## Plot example

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
            country_name=gg.dCOUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3, leadtime_hour=720
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

```python

```
