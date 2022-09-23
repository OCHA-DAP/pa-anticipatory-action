Notebook to make sure the computations of precipitation terciles was done correctly and experiment with
different methods of computing. Best method is now implemented in `src`.

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import geopandas as gpd

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
iso3 = "tcd"
```

```python
config = Config()
data_processed_dir = os.path.join(
    config.DATA_DIR, config.PUBLIC_DIR, config.PROCESSED_DIR
)
chirps_country_processed_dir = os.path.join(data_processed_dir, iso3, "chirps")
chirps_country_processed_path = os.path.join(
    chirps_country_processed_dir, "monthly", f"{iso3}_chirps_monthly.nc"
)
```

```python
parameters = config.parameters(iso3)
country_data_raw_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
)
adm1_bound_path = os.path.join(
    country_data_raw_dir, config.SHAPEFILE_DIR, parameters["path_admin1_shp"]
)
```

```python
# Read in monthly Chad dataset
ds = xr.load_dataset(chirps_country_processed_path)
```

```python
# Simplify into DA at a single datapoint
da = ds["precip"].isel(latitude=140, longitude=150)
```

```python
# Do the rolling seasonal sum
seas_len = 3
da_season = (
    da.rolling(time=seas_len, min_periods=seas_len)
    .sum()
    .dropna(dim="time", how="all")
)
```

```python
# Plot to make sure it makes sense
fig, ax = plt.subplots(figsize=(20, 10))
da.plot(ax=ax)
da_season.plot(ax=ax)
```

```python
# Confirm that first point in da_season is sum of first three in da
da_season.isel(time=0).values
```

```python
da.isel(time=[0, 1, 2]).sum().values
```

```python
# Limit to IRI climate range
da_season_climate = da_season.sel(
    time=da_season.time.dt.year.isin(range(1982, 2011))
)
```

```python
# Plot the preipitation in each month group
for i, group in da_season_climate.groupby(da_season_climate.time.dt.month):
    group = np.log10(group)
    group.plot.hist(
        bins=np.arange(-5, 5, 0.1), histtype="step", label=i, alpha=0.5, lw=2
    )
```

```python
# For each month, print out and plot the range of values
for i, group in da_season_climate.groupby(da_season_climate.time.dt.month):
    print("month", i)
    print(group.values)
    fig, ax = plt.subplots()
    y = group.values.copy()
    y.sort()
    ax.plot(y, "o")
    ax.set_yscale("log")
    print(group.quantile(0.33))
```

### Compare methods of computation
One method is implemented in the code. This gives correct results but is not an optimal method of computation and saving. 
Thus, test a new method and see if the results are the same

```python
# load data
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
ds_season_climate_quantile = ds_season_climate.groupby(
    ds_season_climate.time.dt.month
).quantile(0.33)
```

```python
da_season = ds_season.precip
da_season_climate = ds_season_climate.precip
da_season_climate_quantile = ds_season_climate_quantile.precip
```

#### Old method

```python
# This is how it's calculated in the code
list_ds_seass = []
for s in np.unique(da_season.time.dt.month):
    da_seas_sel = da_season.sel(time=da_season.time.dt.month == s)
    # keep original values of cells that are either nan or have
    # below average precipitation, all others are set to -666
    da_seas_below = da_seas_sel.where(
        (da_seas_sel.isnull())
        | (da_seas_sel <= da_season_climate_quantile.sel(month=s)),
        -666,
    )
    list_ds_seass.append(da_seas_below)
da_season_below = xr.concat(list_ds_seass, dim="time")
```

```python
# Check that it corresponds to 1/3
da_season_below.where(da_season_below > -666).count() / da_season.count()
```

#### New method

```python
# With new method, use groupby to create boolean mask
ds_season = ds_season.assign_coords({"month": ds_season.time.dt.month})
ds_lt = ds_season.groupby("month").apply(
    lambda x: x.where(
        x <= ds_season_climate_quantile.sel(month=x.time.dt.month)
    )
)
ds_gt = ds_season.groupby("month").apply(
    lambda x: x.where(
        x > ds_season_climate_quantile.sel(month=x.time.dt.month)
    )
)
ds_season_lowup = ds_season.assign(
    {"lower": ds_lt["precip"], "upper": ds_gt["precip"]}
)
```

```python
# Confirm lower tercile is 1/3
lower = (~np.isnan(ds_season_lowup["lower"])).sum()
upper = (~np.isnan(ds_season_lowup["upper"])).sum()
lower / (lower + upper)
```

#### Even newer method
Yet another option is to compute each of the three terciles, so the below normal, normal and above average. 
Moreover, we don't need the groupby and thus can simplify a bit

```python
da_season_climate_quantile_list = da_season_climate.groupby(
    da_season_climate.time.dt.month
).quantile([0.33, 0.66])
ds_bn = da_season.where(
    da_season
    <= da_season_climate_quantile_list.sel(
        quantile=0.33, month=da_season.time.dt.month
    )
)
ds_an = da_season.where(
    da_season
    >= da_season_climate_quantile_list.sel(
        quantile=0.66, month=da_season.time.dt.month
    )
)
ds_no = da_season.where(
    (
        da_season
        > da_season_climate_quantile_list.sel(
            quantile=0.33, month=da_season.time.dt.month
        )
    )
    & (
        da_season
        < da_season_climate_quantile_list.sel(
            quantile=0.66, month=da_season.time.dt.month
        )
    )
)
ds_season_terc = ds_season.assign(
    {
        "below_normal": ds_bn.drop("quantile"),
        "normal": ds_no,
        "above_normal": ds_an.drop("quantile"),
    }
)
```

```python
# Confirm lower tercile is 1/3
lower = (~np.isnan(ds_season_terc["below_normal"])).sum()
total = (~np.isnan(ds_season_terc.precip)).sum()
lower / total
```

As we can see all of the three methods align. We therefore go with the third method, which is the most complete and cleanest


Below we do a bit more analysis to make sure the outcomes are correct

```python
# Make a plot showing fraction of pixels in lower tercile (so "dryness")
n_lower = (~np.isnan(ds_season_terc["below_normal"])).sum(
    dim=["longitude", "latitude"]
)
n_tot = (~np.isnan(ds_season_terc["precip"])).sum(
    dim=["longitude", "latitude"]
)
frac = n_lower / n_tot

fig, ax = plt.subplots()
ax.axhline(0.33, c="k")
frac.plot(ax=ax)
```

Check that the division of terciles look correct

```python
# Make a plot of a particular time point
# points with below normal precip
time = 340
# large figsize needed to see indeed the two plots align
g = (
    ds_season_terc["below_normal"]
    .isel(time=time)
    .plot(cmap=matplotlib.colors.ListedColormap(["blue"]), figsize=(10, 15))
)
gdf_adm = gpd.read_file(adm1_bound_path)
gdf_adm.boundary.plot(ax=g.axes);
```

```python
# points with normal or above normal precip
# large figsize needed to see indeed the two plots align
g = (
    ds_season_terc["normal"]
    .isel(time=time)
    .plot(cmap=matplotlib.colors.ListedColormap(["blue"]), figsize=(10, 15))
)
ds_season_terc["above_normal"].isel(time=time).plot(
    cmap=matplotlib.colors.ListedColormap(["blue"]),
    ax=g.axes,
    add_colorbar=False,
)
gdf_adm.boundary.plot(ax=g.axes);
```

```python

```
