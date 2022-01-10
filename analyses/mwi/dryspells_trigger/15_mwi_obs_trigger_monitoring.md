```python
%load_ext autoreload
%autoreload 2

import os
from pathlib import Path
import sys
from datetime import date

path_mod = f"{Path.cwd().parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells

import geopandas as gpd
import xarray as xr
import numpy as np
# for plotting
import matplotlib.pyplot as plt
from matplotlib import colors
```

The first we will do is setup the 3 monitors, which are going to use the 3 separate types of aggregation. We only need to download the data once for each because they all depend on the same raw data from ARC2. Currently, I've kept them using the 2018 monitoring period as an example because we have not reached the period we want to monitor. To monitor in 2022, all we have to do is set `monitoring_start = "2021-01-07"` and `monitoring_end = date.today()`.

```python
## Global variables for all monitoring

POLY_PATH = Path(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'cod_ab',
    'mwi_drought_adm2.gpkg'
)

# Admin3 file for mapping
gdf_adm3 = gpd.read_file(Path(
    os.getenv('AA_DATA_DIR'),
    'public',
    'raw',
    'mwi',
    'cod_ab',
    'mwi_adm_nso_20181016_shp',
    'mwi_admbnda_adm3_nso_20181016.shp'
))

gdf_adm3 = gdf_adm3[gdf_adm3.ADM1_PCODE == "MW3"]

MONITORING_START = "2022-01-07"
MONITORING_END = date.today()
RANGE_X = ("32E", "36E")
RANGE_Y = ("20S", "5S")

# centroid method
arc2_centr = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = MONITORING_START,
    monitoring_end = MONITORING_END,
    range_x = RANGE_X,
    range_y = RANGE_Y
)

arc2_centr.download()

# touching method
arc2_touch = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = MONITORING_START,
    monitoring_end = MONITORING_END,
    range_x = RANGE_X,
    range_y = RANGE_Y,
    agg_method = "touching"
)

# approximate mask
arc2_approx = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = MONITORING_START,
    monitoring_end = MONITORING_END,
    range_x = RANGE_X,
    range_y = RANGE_Y,
    agg_method = "approximate_mask"
)
```

Now with each of these, we can just re-process our data and calculate rolling sums and dry spells.

```python
arc2_centr.aggregate_data()
arc2_centr.calculate_rolling_sum()
arc2_centr.identify_dry_spells()

arc2_touch.aggregate_data()
arc2_touch.calculate_rolling_sum()
arc2_touch.identify_dry_spells()

arc2_approx.aggregate_data()
arc2_approx.calculate_rolling_sum()
arc2_approx.identify_dry_spells()
```

How many dry spells have we detected thus far in our monitoring period?

```python
print(
    f"Centroid method: {arc2_centr.count_dry_spells()}\n"
    f"Touching method: {arc2_touch.count_dry_spells()}\n"
    f"Approx m method: {arc2_approx.count_dry_spells()}"
)
```

And are we getting close to triggering? This is useful during monitoring. Let's check the number of areas that have gotten 12 days or more cumulative sum below 2mm rainfall. We can also check the current cumulative sum

```python
DAYS = 12

print(
    f"Centroid method: {arc2_centr.count_days_under_threshold(DAYS)}\n"
    f"Touching method: {arc2_touch.count_days_under_threshold(DAYS)}\n"
    f"Approx m method: {arc2_approx.count_days_under_threshold(DAYS)}"
)
```

```python
arc2_centr.days_under_threshold(raster=False)
```

So, if we have triggered, we also want to plot the cumulative rainfall across our area of interest and monitoring period, as well as numbers of days without rain above 2mm. We can plot with any of the methods of aggregation, but chose to use centroid for below.

```python
# get number of consecutive days under threshold per area
da_days = arc2_centr.days_under_threshold()
f, ax = plt.subplots()
da_days = da_days.rio.clip(gdf_adm3.geometry)
da_days = da_days.where(da_days.values >= 0, np.NaN)
divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=14, vmax = 28)
da_days.plot(ax = ax,
             cmap='Greys',
             norm = divnorm)
gdf_adm3.plot(ax=ax, facecolor="none", alpha=0.5)

plt.title("Most consecutive days under 2mm cumulative rainfall")
```

```python
da_cum = arc2_centr.cumulative_rainfall()
f, ax = plt.subplots()
da_cum = da_cum.rio.clip(gdf_adm3.geometry)
da_cum = da_cum.where(da_cum.values >= 0, np.NaN)
da_cum.plot(ax = ax,
            cmap='Greys')
gdf_adm3.plot(ax=ax, facecolor="none", alpha=0.5)
plt.title(f"Cumulative rainfall from {arc2_centr.date_min}")
```

And lastly, how many days total in each admin 2, over our monitoring period, have had cumulative sums under our threshold. Let's just look at the centroid method for easier visualization.

```python
arc2_centr.find_longest_runs()
```

```python

```

```python

```
