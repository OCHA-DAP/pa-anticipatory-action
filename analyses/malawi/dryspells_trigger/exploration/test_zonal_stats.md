```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import plotly.express as px 
import seaborn as sns
import calendar
import math
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'size'   : 16}

mpl.rc('font', **font)
```

```python
#set plot colors
hdx_blue='#66B0EC'
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(country_data_exploration_dir,'chirps')
cod_ab_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3,"cod_ab")
adm2_shp_path=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3,"cod_ab",parameters["path_admin2_shp"])
adm1_shp_path=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3,"cod_ab",parameters["path_admin1_shp"])
```

### Read in forecast and observational data

```python
gdf_adm1=gpd.read_file(adm1_shp_path)
```

```python
da_for=processing.get_ecmwf_forecast_by_leadtime("mwi")
```

```python
da_for_clip=da_for.rio.set_spatial_dims(x_dim="longitude",y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
#get the global dataset, such that we can clip it with a buffer
da_obs_glb=xr.open_dataset(config.CHIRPS_MONTHLY_RAW_PATH)
```

```python
bb=gdf_adm1.total_bounds
buf=1
```

```python
da_obs_mwi=da_obs_glb.rio.write_crs("EPSG:4326").rio.clip_box(bb[0]-buf,bb[1]-buf,bb[2]+buf,bb[3]+buf).precip
```

```python
da_obs_interp=da_obs_mwi.interp(latitude=da_for["latitude"],longitude=da_for["longitude"],method="linear")
da_obs_clip=da_obs_interp.rio.clip(gdf_adm1["geometry"],all_touched=True)
```

```python
da_for_clip=da_for.rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"],all_touched=True)
```

```python
#only include times that are present in both datasets (assuming no missing data)
da_for_clip=da_for_clip.sel(time=slice(da_obs_clip.time.min(), da_obs_clip.time.max()))
da_obs_clip=da_obs_clip.sel(time=slice(da_for_clip.time.min(), da_for_clip.time.max()))
```

### Compute with zonal stats
This is how stats are currently computed, but this is super slow

```python
from rasterstats import zonal_stats

def compute_zonal_stats(
    ds,
    raster_transform,
    adm_path,
    adm_col,
    percentile_list=None,
):
    # compute statistics on level in adm_path for all dates in ds
    df_list = []
    for leadtime in ds.leadtime.values:
        for number in ds.number.values:
            df = gpd.read_file(adm_path)[[adm_col, "geometry"]]
            ds_date = ds.sel(number=number, leadtime=leadtime)

            df[["mean_cell", "max_cell", "min_cell"]] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date.values,
                    affine=raster_transform,
                    nodata=np.nan,
                    all_touched=False
                )
            )[["mean", "max", "min"]]

            if percentile_list: 
                df[
                    [f"percentile_{str(p)}" for p in percentile_list]
                ] = pd.DataFrame(
                    zonal_stats(
                        vectors=df,
                        raster=ds_date.values,
                        affine=raster_transform,
                        nodata=np.nan,
                        stats=" ".join(
                            [f"percentile_{str(p)}" for p in percentile_list]
                        ),
                    )
                )[
                    [f"percentile_{str(p)}" for p in percentile_list]
                ]

            df["number"] = number
            df["leadtime"] = leadtime

            df_list.append(df)
        df_hist = pd.concat(df_list)
        # drop the geometry column, else csv becomes huge
        df_hist = df_hist.drop("geometry", axis=1)

    return df_hist
```

```python
bla=compute_zonal_stats(da_for.sel(time="2020-12-01"),
                da_for.rio.transform(),
                adm1_shp_path,
                parameters[f"shp_adm1c"],
            )
```

```python
bla[(bla.ADM1_EN=="Southern")&(bla.leadtime==1)].head()
```

```python
bla[(bla.ADM1_EN=="Southern")&(bla.leadtime==1)].mean()
```

### Compute with rioxarray
Another method is to clip the raster to the area of interest and then compute the stats
From quick testing this gives the same results and is waaay faster. 

My question is if you think this is a valid method or anything should be improved? Else I will implement it as a proper function in `general_utils`

```python
from shapely.geometry import mapping

def compute_zonal_stats_xarray(
    raster, shapefile, lon_coord="lon", lat_coord="lat",all_touched=False
):
    raster_clip = raster.rio.set_spatial_dims(
        x_dim=lon_coord, y_dim=lat_coord
    ).rio.clip(
        shapefile.geometry.apply(mapping), raster.rio.crs, all_touched=all_touched
    )
    grid_mean = raster_clip.mean(dim=[lon_coord, lat_coord]).rename("mean_cell")

    grid_min = raster_clip.min(dim=[lon_coord, lat_coord]).rename("min_cell")
    grid_max = raster_clip.max(dim=[lon_coord, lat_coord]).rename("max_cell")
    grid_std = raster_clip.std(dim=[lon_coord, lat_coord]).rename("std_cell")
    zonal_stats_xr = xr.merge(
        [grid_mean, grid_min, grid_max, grid_std]
    ).drop("spatial_ref")
    zonal_stats_df = zonal_stats_xr.to_dataframe()
    zonal_stats_df = zonal_stats_df.reset_index()
    return zonal_stats_df
```

```python
stats_df_list=[]
for a in gdf_adm1.ADM1_EN.unique():
    gdf_adm=gdf_adm1[gdf_adm1.ADM1_EN==a]
    stats_adm=compute_zonal_stats_xarray(da_for.rio.write_crs("EPSG:4326"),gdf_adm,lon_coord="longitude",lat_coord="latitude")
    stats_adm["ADM1_EN"]=a
    stats_df_list.append(stats_adm)
stats_per_adm=pd.concat(stats_df_list)
```

```python
stats_per_adm[(stats_per_adm.ADM1_EN=="Southern")&(stats_per_adm.time=="2020-12-01")&(stats_per_adm.leadtime==1)].head()
```
