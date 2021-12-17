```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
iso3="tcd"
```

```python
config=Config()
data_processed_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR)
chirps_country_processed_dir = os.path.join(data_processed_dir,iso3,"chirps")
chirps_country_processed_path = os.path.join(chirps_country_processed_dir,"monthly",f"{iso3}_chirps_monthly.nc")
```

```python
# Read in monthly Chad dataset
ds = xr.load_dataset(chirps_country_processed_path)
```

```python
# Simplify into DA at a single datapoint
da = ds['precip'].isel(latitude=140, longitude=150)
```

```python
# Do the rolling seasonal sum
seas_len = 3
da_season = (
        da.rolling(time=seas_len, min_periods=seas_len)
        .sum()
        .dropna(dim='time', how='all')
)
```

```python
# Plot to make sure it makes sense
fig, ax = plt.subplots(figsize=(20,10))
da.plot(ax=ax,label="monthly precipitation")
da_season.plot(ax=ax,label="sum of 3month precipitation")
plt.legend()
```

```python
# Confirm that first point in da_season is sum of first three in da
da_season.isel(time=0).values
```

```python
da.isel(time=[0,1,2]).sum().values
```

```python
# Limit to IRI climate range
da_season_climate = da_season.sel(time=da_season.time.dt.year.isin(range(1982, 2011)))
```

```python
da_season_climate.groupby(
        da_season_climate.time.dt.month
    )
```

```python
group
```

```python
# Plot the preipitation in each month group
for i, group in da_season_climate.groupby(
        da_season_climate.time.dt.month
    ):
    #take log such that values are in comparable range
    group = np.log10(group)
    group.plot.hist(bins=np.arange(-5, 5, 0.1), histtype='step', label=i, alpha=0.5, lw=2)
plt.legend()
```

```python
# For each month, print out and plot the range of values
for i, group in da_season_climate.groupby(
        da_season_climate.time.dt.month
    ):
    print('month', i)
    print(group.values)
    fig, ax = plt.subplots()
    y = group.values.copy()
    y.sort()
    ax.plot(y, 'o')
    ax.set_yscale('log')
    print(group.quantile(0.33))
```

```python
ds = xr.load_dataset(chirps_country_processed_path)
seas_len = 3
ds_season = (
        ds.rolling(time=seas_len, min_periods=seas_len)
        .sum()
        .dropna(dim="time", how="all")
    )
ds_season_climate = ds_season.sel(
        time=ds_season.time.dt.year.isin(range(1982, 2011))
    )
ds_season_climate_quantile_old = ds_season_climate.groupby(
        ds_season_climate.time.dt.month
    ).quantile(0.33)
```

```python
ds_season_old = (
        ds.rolling(time=seas_len, min_periods=seas_len)
        .sum()
        .dropna(dim="time", how="all")
    )
```

```python
ds_season_climate_quantile_old = ds_season_climate.groupby(
        ds_season_climate.time.dt.month
    ).quantile(0.33)
```

```python
# This is how it's calculated in the code
list_ds_seass = []
for s in np.unique(ds_season_old.time.dt.month):
    ds_seas_sel = ds_season_old.sel(time=ds_season_old.time.dt.month == s)
    # keep original values of cells that are either nan or have
    # below average precipitation, all others are set to -666
    ds_seas_below = ds_seas_sel.where(
        (ds_seas_sel.isnull())
        | (ds_seas_sel <= ds_season_climate_quantile_old.sel(month=s)),
        -666,
    )
    list_ds_seass.append(ds_seas_below)
ds_season_below = xr.concat(list_ds_seass, dim="time")
```

```python
da_season_below=ds_season_below.precip
```

```python
da_season_below.where(da_season_below>-666).count()/da_season_below.count()
```

```python
da_season_below.where(da_season_below>-666).count()
```

```python
da_season_below
```

### Compute from scratch with new method

```python
# ds = xr.load_dataset(chirps_country_processed_path)
# seas_len = 3
# ds_season = (
#         ds.rolling(time=seas_len, min_periods=seas_len)
#         .sum()
#         .dropna(dim="time", how="all")
#     )
# ds_season_climate = ds_season.sel(
#         time=ds_season.time.dt.year.isin(range(1982, 2011))
#     )
# ds_season_climate_quantile = ds_season_climate.groupby(
#         ds_season_climate.time.dt.month
#     ).quantile(0.33)
```

```python
ds_season
```

```python
ds_lt_simple = ds_season.where(ds_season
        < ds_season_climate_quantile_old.sel(month=ds_season.time.dt.month)
    )
```

```python
ds_season.precip.count()
```

```python
ds_lt_simple.precip.count()
```

```python
ds_lt.precip.count()
```

```python
np.unique(ds_lt.precip)
```

```python
ds_season_climate_quantile = ds_season_climate.groupby(
        ds_season_climate.time.dt.month
    ).quantile([1/3,2/3])
```

#TODO: add season coord

```python
da_season_climate_quantile
```

```python

```

```python
# With new method, use groupby to create boolean mask
da_season = ds_season.assign_coords({'month': ds_season.time.dt.month}).precip
da_season_climate_quantile=ds_season_climate_quantile.precip
ds_bn = da_season.groupby('month').apply(lambda x: 
                x.where(x <= da_season_climate_quantile.sel(quantile=1/3,month=x.time.dt.month)))
ds_an = da_season.groupby('month').apply(lambda x: 
                x.where(x >= da_season_climate_quantile.sel(quantile=2/3,month=x.time.dt.month)))
ds_no = da_season.groupby('month').apply(lambda x: 
                x.where((x > da_season_climate_quantile.sel(quantile=1/3,month=x.time.dt.month))&(x < da_season_climate_quantile.sel(quantile=2/3,month=x.time.dt.month))))
# ds_season = ds_season.assign({'below_normal': ds_bn['precip'], 'normal': ds_no['precip'],'above_normal':ds_no['precip']})
```

```python
ds_bn.count()
```

```python
ds_bn.sum()+ds_no.sum()+ds_an.sum()
```

```python
ds_bn.precip.sum()+ds_no.precip.sum()+ds_an.precip.sum()
```

```python
da_season.sum()
```

```python
ds_bn.count()/ds_season.precip.count()
```

```python
# Confirm lower tercile is 1/3
lower = (~np.isnan(ds_season['below_normal'])).sum() 
total = (~np.isnan(ds_season.precip)).sum()
lower / total
```

```python
# Confirm lower tercile is 1/3
lower = (~np.isnan(ds_season['below_normal'])).sum() 
total = (~np.isnan(ds_season)).sum()
lower / total
```

```python
# Make a plot showing fraction of pixels in lower tercile (so "dryness")
n_lower = (~np.isnan(ds_season['lower'])).sum(dim=['longitude', 'latitude']) 
n_upper = (~np.isnan(ds_season['upper'])).sum(dim=['longitude', 'latitude'])
frac = n_lower / (n_lower + n_upper)

fig, ax = plt.subplots()
ax.axhline(0.33, c='k')
frac.plot(ax=ax)
```

```python
# Make a plot of a particular time point
time = 240
ds_season['lower'].isel(time=time).plot(vmin=-1, vmax=5)
```

```python
ds_season['upper'].isel(time=time).plot(vmin=-1, vmax=5)
```

```python

```
