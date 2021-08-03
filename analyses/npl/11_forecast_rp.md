Because the forecasts at Chatara and Chisapani have a negative bias, 
we want to know what to know what a 2 year RP effectively corresponds
to for each lead time.

```python
from pathlib import Path
import os
import sys
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
reload(utils)
mpl.rcParams['figure.dpi'] = 200
```

```python
COUNTRY_ISO3 = 'npl'
LEADTIMES = [x+1 for x in range(7)]
STATIONS = [
    'Chatara_v3',
    'Chisapani_v3'
]
MAIN_RP = 2
FORECAST_PERCENTILE = 30

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)[STATIONS]
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES, 
    interp=False, shift_dates=False
)[STATIONS]
ds_glofas_forecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)

df_return_period =  pd.read_excel(GLOFAS_RP_FILENAME, index_col='rp')
```

### Calculate the return periods

```python
reload(utils)
from src.utils_general import statistics
reload(statistics)
method = "analytical"
years = np.arange(1.5, 500.5, 0.1)
extend_factor = 10
df_rps_obs = utils.get_return_periods(ds_glofas_reanalysis, years=years, method=method, extend_factor=extend_factor)
# Select observations only on the date of forecast
df_rps_obs_sub = utils.get_return_periods(
    ds_glofas_reanalysis.reindex({"time": ds_glofas_forecast_summary.dropna(dim="time").time}),
    years=years, method=method, extend_factor=extend_factor)

df_rps_forecast_dict = {}
for leadtime in LEADTIMES:
    ds = ds_glofas_forecast_summary.sel(leadtime=leadtime, percentile=FORECAST_PERCENTILE)
    df_rps_forecast_dict[leadtime] = utils.get_return_periods(ds, years=years, method=method, extend_factor=extend_factor)
```

### Plot them for comparison

```python
for station in STATIONS:
    fig, ax = plt.subplots()
    ax.plot(df_rps_obs[station], c='k', label='full model')
    ax.plot(df_rps_obs_sub[station], 'k--', label='subset model')

    for leadtime, df_rps_forecast in df_rps_forecast_dict.items():
        ax.plot(df_rps_forecast[station], label=leadtime, alpha=0.75)
    ax.set_title(station[:-3])
    rp_val = df_return_period.loc[MAIN_RP, station[:-3]]
    ax.axhline(rp_val, c='k', ls=':', lw=0.5, label=f"1 in {MAIN_RP} y")
    ax.axvline(2, c='k', ls=':', lw=0.5)
    ax.legend()
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')

```

### Compute corresponding forecast RP

```python
# We just need approximate so I will find the closest value in the dataframe
```

```python
for station in STATIONS:
    print(station)
    rp_val = df_return_period.loc[MAIN_RP, station[:-3]]
    for leadtime, df_rps_forecast in df_rps_forecast_dict.items():
        rp_equiv = df_rps_forecast.index[(df_rps_forecast[station] - rp_val).abs().argmin()]
        print(leadtime, f'{rp_equiv:.1f}')
```
