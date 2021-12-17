```python
%load_ext autoreload
%autoreload 2

import os
from pathlib import Path
import sys
from datetime import date
from rasterio.enums import Resampling

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells
from src.utils_general.raster_manipulation import compute_raster_statistics
import geopandas as gpd
from rasterio.enums import Resampling
import xarray as xr
import pandas as pd
import cftime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

poly_path = os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'cod_ab',
    'mwi_drought_adm2.gpkg'
)

arc2_centr = DrySpells(
    country_iso3 = "mwi",
    monitoring_start = "2018-01-07",
    monitoring_end = "2018-03-07",
    range_x = ("32E", "36E"),
    range_y = ("20S", "5S")
)

arc2_centr.download_data()

arc2_touch = DrySpells(
    country_iso3 = "mwi",
    monitoring_start = "2018-01-07",
    monitoring_end = "2018-03-07",
    range_x = ("32E", "36E"),
    range_y = ("20S", "5S"),
    agg_method = "touching"
)

arc2_approx = DrySpells(
    country_iso3 = "mwi",
    monitoring_start = "2018-01-07",
    monitoring_end = "2018-03-07",
    range_x = ("32E", "36E"),
    range_y = ("20S", "5S"),
    agg_method = "approximate_mask"
)
```

Below we process and calculate dry spells using the variety of possible methods.

```python
arc2_centr.downsample_data(poly_path, "ADM2_PCODE", reprocess=True)
arc2_centr.calculate_rolling_sum(True)
arc2_centr.identify_dry_spells()

arc2_touch.downsample_data(poly_path, "ADM2_PCODE", reprocess=True)
arc2_touch.calculate_rolling_sum(True)
arc2_touch.identify_dry_spells()

arc2_approx.downsample_data(poly_path, "ADM2_PCODE", reprocess=True)
arc2_approx.calculate_rolling_sum(True)
arc2_approx.identify_dry_spells()
```

We also want to compare to calculating dry spells for each raster cell and then looking at % of raster cells in the southern region in dry spell.

```python
ds = arc2_centr.load_raw_data()
gdf_adm1 = gpd.read_file(filename = os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'raw',
    'mwi',
    'cod_ab',
    'mwi_adm_nso_20181016_shp',
    'mwi_admbnda_adm1_nso_20181016.shp'
))
gdf_adm1 = gdf_adm1[gdf_adm1.ADM1_PCODE == "MW3"]
ds = ds.rio.reproject(
    ds.rio.crs,
    shape = (ds.rio.height * 4, ds.rio.width * 4),
    resampling = Resampling.mode
)
ds = ds.rio.clip(gdf_adm1.geometry)
ds_rolling = ds.rolling(T=14, min_periods=14).sum().dropna(dim="T", how="all")

# Only calculate for time of interest between January 7th and March 7th

months = np.array([x.month for x in ds_rolling["T"].values]) * 100
days = np.array([x.day for x in ds_rolling["T"].values])
months_days = months + days

ds_rolling = ds_rolling[(months_days >= 107) & (months_days <= 307), :, :]

# convert to pandas dataframe for saving

df = pd.DataFrame({
    "time":ds_rolling.coords["T"].values,
    "percent_area":xr.where(ds_rolling <= 2, 1, 0).mean(dim=["x", "y"]) * 100
})

df.to_csv(
    os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'arc2',
    'arc2_pct_area_dry_spells.csv'
))


ds_rolling.to_netcdf(
    os.path.join(
        os.getenv("AA_DATA_DIR"),
        "public",
        "processed",
        "mwi",
        "arc2",
        "arc2_raster_dry_spells.nc"
    )
)

# find areas that have ever been in dry spell that year
ds_year = xr.where(ds_rolling <= 2, 1, 0).groupby("T.year").max(dim="T")

df_year = pd.DataFrame({
    "year":ds_year.coords["year"].values,
    "percent_area":ds_year.mean(dim=["x", "y"]) * 100
})

df_year.to_csv(
    os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'arc2',
    'arc2_pct_area_dry_spells_cumulative.csv'
))

ds_year.to_netcdf(
    os.path.join(
        os.getenv("AA_DATA_DIR"),
        "public",
        "processed",
        "mwi",
        "arc2",
        "arc2_raster_dry_spells_cum.nc"
    )
)
```

```python
arc2_centr.find_longest_runs()
```

```python
arc2_centr.dry_spell_count()
```

```python
arc2_centr.load_rolling_sum_data(True)
```

```python
df = arc2_centr.days_under_threshold()
gdf_merged = gdf_adm2.merge(df, on="ADM2_PCODE")

alt.Chart(gdf_merged).mark_geoshape(
).encode(
    color="days_under_threshold"
)
```

```python
ds = arc2_centr.load_raw_data()
```

```python
ds2 = ds[ds.indexes["T"].to_datetimeindex().date >= arc2_centr.date_min,:,:]
ds2 = ds2.reindex(T=list(reversed(ds2.indexes["T"])))
xr.where(ds2.cumsum(dim="T") <= 200, 0, 1).argmax(dim="T").plot()
```

```python

```

```python
# twoslopenorm class
divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=14, vmax = 28)
ds = arc2_centr.days_under_threshold()
gdf_adm3 = gpd.read_file(os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'cod_ab',
    'mwi_drought_adm2.gpkg'
)
f, ax = plt.subplots()
ds = ds.rio.clip(gdf_adm2.geometry)
ds = ds.where(ds.values >= 0, np.NaN)
ds.plot(ax = ax,
        cmap='Greys',
        norm=divnorm)
gdf_adm2.plot(ax=ax, facecolor="none")

plt.title("Consecutive days under 2mm cumulative rainfall")
```

```python
 ds.where(ds.values < 0, np.NaN)
```

```python
gdf_adm2 = gpd.read_file(
os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'cod_ab',
    'mwi_drought_adm2.gpkg'
))

gdf_adm2 = gdf_adm2[gdf_adm2.ADM1_PCODE == "MW3"]
```

```python
f, ax = plt.subplots()

ds.plot(ax = ax)
```

```python
np.unique(ds)
```

```python

```
