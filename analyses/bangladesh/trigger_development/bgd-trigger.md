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

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding import glofas
from src.bangladesh import get_glofas_data as ggd

DATA_DIR = os.environ['AA_DATA_DIR']
GLOFAS_DIR = os.path.join(DATA_DIR, 
                          'processed', 
                          'bangladesh', 
                          'GLOFAS_Data')
STATION = 'Bahadurabad_glofas'
LEADTIME_HOURS = [120, 240, 360]
COUNTRY_NAME = "bangladesh"
COUNTRY_ISO3 = "bgd"
```

```python
glofas_reanalysis = glofas.GlofasReanalysis(
    stations_lon_lat=ggd.FFWC_STATIONS
)
glofas_forecast = glofas.GlofasForecast(
    stations_lon_lat=ggd.FFWC_STATIONS, leadtime_hours=LEADTIME_HOURS
)
glofas_reforecast = glofas.GlofasReforecast(
    stations_lon_lat=ggd.FFWC_STATIONS, leadtime_hours=LEADTIME_HOURS
)
```

```python
year = "2020"
da_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_name=gCOUNTRY_NAME, country_iso3=COUNTRY_ISO3
    )[STATION].sel(time=slice(year, year))
da_forecast_720 = (
        glofas_forecast.read_processed_dataset(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3, leadtime_hour=720
        )[STATION]
        .shift(time=30)
        .sel(time=slice(year, year))
    )
da_reforecast_720 = (
        glofas_reforecast.read_processed_dataset(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3, leadtime_hour=720
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
