```python
from pathlib import Path
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils, glofas
```

```python
COUNTRY_ISO3 = 'npl'

RETURN_PERIODS_GLOFAS = {
    'Chatara': {
        1.5: 5920,
        2.0: 6350,
        5.0: 7400
    },
    'Chisapani': {
        1.5: 5850,
        2.0: 6300,
        5.0: 7425,
        20.0: 8900
    },
    'Asaraghat': {
        1.5: 1700,
        2.0: 1800,
        5.0: 2070,
        20.0: 2400
    }
}

```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
```

```python
def get_emperical_return_period(ds_reanalysis: xr.Dataset, station: str):
    df_rp = (
        ds_reanalysis.to_dataframe()[[station]]
        .rename(columns={station: "discharge"})
        .resample(rule="A", kind="period")
        .max()
        .sort_values(by="discharge", ascending=False)
    )
    df_rp["year"] = df_rp.index.year

    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return interp1d(df_rp["rp"], df_rp["discharge"])

def get_analytical_return_period(ds_reanalysis, station):
    df_rp = (
        ds_reanalysis.to_dataframe()[[station]]
        .rename(columns={station: "discharge"})
        .resample(rule="A", kind="period")
        .max()
        .sort_values(by="discharge", ascending=False)
    )
    df_rp["year"] = df_rp.index.year

    rvs = df_rp['discharge']
    shape, loc, scale = gev.fit(rvs, loc=1500, scale=800)
    x = df_rp["discharge"]
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1- y)
    return interp1d(y, x)


ds_reanalysis = ds_glofas_reanalysis
years = np.arange(1.5, 30.5, 0.5)
stations = list(ds_reanalysis.keys())
df_rps_empirical = pd.DataFrame(columns=stations, index=years)
df_rps_analytical = pd.DataFrame(columns=stations, index=years)
for station in stations:
    f_rp = get_emperical_return_period(ds_reanalysis=ds_reanalysis, station=station)
    df_rps_empirical[station] = f_rp(years) 
    
    f_rp_analytical = get_analytical_return_period(ds_reanalysis=ds_reanalysis, station=station)
    df_rps_analytical[station] = f_rp_analytical(years) 
```

```python

for i, station in enumerate(stations):
    fig, ax = plt.subplots()
    ax.plot(df_rps_empirical.index, df_rps_empirical[station])
    ax.plot(df_rps_empirical.index, df_rps_analytical[station])
    if station in RETURN_PERIODS_GLOFAS:
        for year, val in RETURN_PERIODS_GLOFAS[station].items():
            ax.plot(year, val, 'o')
    ax.set_title(station)
```

```python

```
