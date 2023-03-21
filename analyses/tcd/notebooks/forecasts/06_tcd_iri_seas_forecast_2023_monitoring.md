# IRI forecast for 2023 monitoring in Chad
Using a threshold of 42.5% over 20% of the area as this is expected to be reached from time to time but not too often. The 42.5% is specifically set to match with the bins of [IRI's graphics](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/). 

#### Load libraries and set global constants


```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```


```python
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import cftime
import calendar
from dateutil.relativedelta import relativedelta
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
import hvplot.xarray
import altair as alt

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import (
    get_iri_data,
    get_iri_data_dominant,
)
from src.utils_general.raster_manipulation import compute_raster_statistics
```


```python
mon_date = "2023-03-16"
lead_time = 4
```


```python
hdx_blue = "#007ce0"
```


```python
iso3 = "tcd"
config = Config()
parameters = config.parameters(iso3)
country_data_raw_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
)
data_processed_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR
)
adm1_bound_path = (
    country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin1_shp"]
)
adm2_path = (
    data_processed_dir
    / iso3
    / config.SHAPEFILE_DIR
    / "tcd_adm2_area_of_interest.gpkg"
)
```

#### Set variables


```python
gdf_adm1 = gpd.read_file(adm1_bound_path)
gdf_adm2 = gpd.read_file(adm2_path)
incl_adm_col = "area_of_interest"
gdf_aoi = gdf_adm2[gdf_adm2[incl_adm_col] == True]
```

## Inspect forecasts

### Threshold


Due to the limited data availability it is very hard to determine the threshold objectively. We do advise against the 60% threshold since even globally this phenomenon that seems too rare for our purpose. 

However a threshold anywhere between 40 and 50 could be reasonable. We experimented with these different thresholds. For now we propose a threshold of 42.5%. This because we estimate it to be already quite rare, in combination with the 20% of the area requirement, but at the same time we estimate it to be possible to occur. The reason we set it to 42.5 specifically is because this matches the IRI bins. Thus people can easily inspect the forecasts themselves on the maproom.


```python
# C indicates the tercile (below-average, normal, or above-average).
# F indicates the publication month, and L the leadtime
ds_iri = get_iri_data(config, download=True)
ds_iri = ds_iri.rio.write_crs("EPSG:4326", inplace=True)
da_iri = ds_iri.prob
# select all cells touching the region
da_iri_allt = da_iri.rio.clip(gdf_aoi["geometry"], all_touched=True)
# C=0 indicates the below average tercile
da_iri_allt_bavg = da_iri_allt.sel(C=0)
```


```python
# check that all touching is done correctly
g = da_iri_allt.sel(F="2023-03-16", L=4, C=0).plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# check that all touching is done correctly
g = da_iri_allt.sel(F="2023-04-16", L=3, C=0).plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# check that all touching is done correctly
g = da_iri_allt.sel(F="2023-05-16", L=2, C=0).plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# check that all touching is done correctly
g = da_iri_allt.sel(F="2023-06-16", L=1, C=0).plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# upsample the resolution in order to create a mask of our aoi
resolution = 0.05
mask_list = []
for terc in da_iri_allt.C.values:
    for lt in da_iri_allt.L.values:
        da_terc_lt = da_iri_allt.sel(C=terc, L=lt)
        da_terc_lt_mask = da_terc_lt.rio.reproject(
            da_terc_lt.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        mask_list.append(da_terc_lt_mask.expand_dims({"C": [terc], "L": [lt]}))
da_iri_mask = (
    xr.combine_by_coords(mask_list)
    .rio.clip(gdf_aoi["geometry"], all_touched=False)
    .prob
)
# reproject changes longitude and latitude name to x and y
# so change back here
da_iri_mask = da_iri_mask.rename({"x": "longitude", "y": "latitude"})
da_iri_mask_bavg = da_iri_mask.sel(C=0)
```


```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2023-03-16", L=4, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2023-04-16", L=3, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2023-05-16", L=2, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
```


```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2023-06-16", L=1, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
```

#### Compute stats
We can now compute the statistics of the region of interest. We use the approximate mask to define the cells included for the computation of the statistics. 

We have to set two parameters: the minimum probability of below average, and the percentage of the area that should have this minimum probability assigned. 

As discussed above we set the probability of below average to 42.5% (but experimentation with other thresholds has been done). 

For now we set the minimum percentage of the area that should reach the threshold to 20% as that was proposed by the Atelier. This seems reasonable to us as it is a substantial area thus possibly indicating widespread drought. At the same time requiring a larger percentage significantly lowers the chances of meeting the trigger, as we often see that extreme values are only forecasted in a smaller area.


```python
# % probability of bavg
threshold = 42.5
# min percentage of the area that needs to reach the threshold
perc_area = 20
```


```python
adm0_col = "admin0Name"
pcode0_col = "admin0Pcod"
```


```python
# compute stats
# dissolve the region to one polygon
gdf_aoi_dissolved = gdf_aoi.dissolve(by=adm0_col)
gdf_aoi_dissolved = gdf_aoi_dissolved[[pcode0_col, "geometry"]]

df_stats_reg_bavg = compute_raster_statistics(
    gdf=gdf_aoi_dissolved,
    bound_col=pcode0_col,
    raster_array=da_iri_mask_bavg,
    lon_coord="longitude",
    lat_coord="latitude",
    stats_list=["min", "mean", "max", "std", "count"],
    # computes value where 20% of the area is above that value
    # at most 80% of the area should be below threshold
    percentile_list=[100 - perc_area],
    all_touched=True,
)
da_iri_mask_thresh = da_iri_mask_bavg.where(da_iri_mask_bavg >= threshold)
df_stats_reg_bavg_thresh = compute_raster_statistics(
    gdf=gdf_aoi_dissolved,
    bound_col=pcode0_col,
    raster_array=da_iri_mask_thresh,
    lon_coord="longitude",
    lat_coord="latitude",
    stats_list=["count"],
    all_touched=True,
)

df_stats_reg_bavg["perc_thresh"] = (
    df_stats_reg_bavg_thresh[f"count_admin0Pcod"]
    / df_stats_reg_bavg[f"count_admin0Pcod"]
    * 100
)
df_stats_reg_bavg["F"] = pd.to_datetime(
    df_stats_reg_bavg["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
)
df_stats_reg_bavg["month"] = df_stats_reg_bavg.F.dt.month
```


```python
df_stats_reg_bavg
```
