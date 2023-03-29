```python
import os
from pathlib import Path
import sys
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.flooding.floodscan import floodscan
from src.utils_general.raster_manipulation import compute_raster_statistics
from src.utils_general.statistics import get_return_periods_dataframe

iso3="ssd"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
country_data_exploration_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
country_data_public_exploration_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / iso3
adm2_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin2_shp"]
country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "processed" / iso3
gdf_rivers=gpd.read_file(country_data_public_exploration_dir/"rivers"/"ssd_main_rivers_fao_250k"/"ssd_main_rivers_fao_250k.shp")
gdf_bentiu=gpd.read_file(country_data_processed_dir / "bentiu" / "bentiu_bounding_box.gpkg")
```

```python
ds = xr.open_dataset(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan.nc')
da = ds.SFED_AREA

da2019 = da.sel(time="2019").max(dim='time')
da2020 = da.sel(time="2020").max(dim='time')
da2020_min = da.sel(time="2020").min(dim='time')
da2021 = da.sel(time="2021").max(dim='time')
da2021_min = da.sel(time="2021").min(dim='time')
da2022 = da.sel(time="2022").max(dim='time')
da2022_min = da.sel(time="2022").min(dim='time')
da_diff2020 = da2020-da2019
da_diff = da2022 - da2021
da_incr = (da.shift(time=-1)-da).sel(time=slice('2022-05', '2022-10')).mean(dim='time', skipna=True)
da_decr = (da.shift(time=-1)-da).sel(time=slice('2021-12', '2022-04')).mean(dim='time', skipna=True)

# average minumum across 1988 to 2019
da_min_avg = da.groupby("time.year").min().sel(year=slice(1988, 2019)).mean(dim="year")

# time to drain relative to 2021
drain_time = (da2022 - da_min_avg) / (da2021-da2022_min)

# recovery
da_recovery=((da2021 - da2022_min)/(da2021-da2021_min))
# saving to netcdf
da_incr.to_netcdf(country_data_exploration_dir/'floodscan'/'ssd_2022_may_oct_avg_increase.nc')
da_decr.to_netcdf(country_data_exploration_dir/'floodscan'/'ssd_2022_nov_2022_apr_avg_decrease.nc')
da_diff.to_netcdf(country_data_exploration_dir/'floodscan'/'ssd_2022_2021_diff.nc')
drain_time.to_netcdf(country_data_exploration_dir/'floodscan'/'ssd_2022_time_to_drain.nc')
da_recovery.to_netcdf(country_data_exploration_dir/'floodscan'/'ssd_2022_recovery.nc')
```

```python
fig, ax = plt.subplots(figsize=(15, 10))
da_diff2020.plot(ax=ax)
#gdf_rivers.plot(ax=ax)
```

```python
fig, ax = plt.subplots(figsize=(15, 10))
da_diff.plot(ax=ax, cbar_kwargs = {"label": "Flood level difference"})
#gdf_rivers[gdf_rivers.NAVIGATION == 'Navigable all year'].plot(ax=ax)
# gdf_bentiu.centroid.plot(ax=ax, color='grey', linewidth=3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Flood level difference, 2022 maximum - 2021 maximum", weight="bold", size = 16)
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(15, 10))
da_incr.plot(ax=ax)
gdf_rivers.plot(ax=ax)
```

```python
fig, ax = plt.subplots(figsize=(15, 10))
da_decr.plot(ax=ax, cbar_kwargs = {"label": "Floodwater change"})
#gdf_rivers[gdf_rivers.NAVIGATION == 'Navigable all year'].plot(ax=ax)
# gdf_bentiu.centroid.plot(ax=ax, color='grey', linewidth=3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Floodwater change, December 2021 to April 2022", weight="bold", size = 16)
plt.show()
```

```python
]
```

```python
L.get_texts()
```

```python
fig, ax = plt.subplots(figsize=(15, 10))
np.log(drain_time).plot(ax=ax)
gdf_rivers.plot(ax=ax)
gdf_bentiu.centroid.plot(ax=ax, color='black', linewidth=3)
```

```python
da_recovery.plot(size=8)
```

```python

```
