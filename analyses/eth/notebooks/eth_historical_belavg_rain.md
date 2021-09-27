```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import xarray as xr
import cftime
import rioxarray
from shapely.geometry import mapping
import seaborn as sns
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp
from src.utils_general.plotting import plot_raster_boundaries_clip
from src.indicators.drought.chirps_rainfallobservations import clip_chirps_monthly_bounds, compute_seasonal_lowertercile_raster, \
get_filepath_seasonal_lowertercile_raster, get_filepath_chirps_monthly
```

#### Set config values

```python
iso3="eth"
config=Config()
parameters = config.parameters(iso3)

public_data_dir = os.path.join(config.DATA_DIR, config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(public_data_dir,config.RAW_DIR,iso3)
country_data_processed_dir = os.path.join(public_data_dir,config.PROCESSED_DIR,iso3)

chirps_glb_dir=os.path.join(public_data_dir,config.RAW_DIR,config.GLOBAL_ISO3,"chirps")
chirps_mwi_dir=os.path.join(country_data_processed_dir,"chirps")
chirps_glb_monthly_path=os.path.join(chirps_glb_dir,"chirps_global_monthly.nc")
chirps_monthly_mwi_path=os.path.join(chirps_mwi_dir,"chirps_mwi_monthly.nc")
```

```python
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```python
#date to make plots for to test values. To be sure this is consistent across the different plots
test_date=cftime.DatetimeGregorian(2020, 1, 1, 0, 0, 0, 0)
test_date_dtime="2020-1-1"
```

```python
## only needed if data needs to be updated
# get_chirps_data_monthly(config,iso3, use_cache=False)
# compute_seasonal_lowertercile_raster(config, iso3, use_cache=False)
```

```python
ds = rioxarray.open_rasterio(get_filepath_seasonal_lowertercile_raster(iso3,config)).sortby("time")
```

```python
da=ds.precip
da.attrs["units"]="mm/month"
```

```python
da_ondmam=da.where(da.time.dt.month.isin([12,5]), drop=True)
```

```python
da_ondmam.sel(time=slice('2000', '2021')).plot(    
    col="time",
    col_wrap=4,
#     row="L",
#     cmap=mpl.cm.YlOrRd, #mpl.cm.RdORYlBu_r,
#     robust=True,
    levels=[-666,0],
    colors=['#cccccc','#f2645a'],
#     cmap="YlOrRd",
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
    },)
```

```python
da_ondmam_perc_bavg=da_ondmam.where(da_ondmam>=0).count(dim=["x","y"])/da_ondmam.count(dim=["x","y"])*100
```

```python
df_ondmam_perc_bavg=da_ondmam_perc_bavg.to_dataframe()
df_ondmam_perc_bavg.drop("spatial_ref",axis=1,inplace=True)
df_ondmam_perc_bavg.rename(columns={"precip":"perc_bavg"},inplace=True)
df_ondmam_perc_bavg.reset_index(inplace=True)
df_ondmam_perc_bavg.time=pd.to_datetime(df_ondmam_perc_bavg.time.apply(lambda x: x.strftime('%Y-%m-%d')))
```

```python
df_ondmam_perc_bavg[df_ondmam_perc_bavg.time.dt.year>=2010]
```

```python
perc_thresh=20
```

```python
df_ondmam_perc_bavg[f"ge_{perc_thresh}"]=np.where(df_ondmam_perc_bavg.perc_bavg>=perc_thresh,1,0)
```

```python
df_ondmam_perc_bavg['consecutive'] = df_ondmam_perc_bavg[f"ge_{perc_thresh}"].groupby( \
    (df_ondmam_perc_bavg[f"ge_{perc_thresh}"] != df_ondmam_perc_bavg[f"ge_{perc_thresh}"].shift()).cumsum()).transform('size') * \
    df_ondmam_perc_bavg[f"ge_{perc_thresh}"]

```

```python
df_ondmam_perc_bavg[df_ondmam_perc_bavg.consecutive>=2]
```

### Same analysis but only for Somali ADMIN1 region

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
```

```python
gdf_som=gdf_adm1[gdf_adm1.ADM1_EN=="Somali"]
```

```python
da_ondmam_som=da_ondmam.rio.clip(gdf_som["geometry"])
```

```python
da_ondmam_som.sel(time=slice('2000', '2021')).plot(    
    col="time",
    col_wrap=4,
#     row="L",
#     cmap=mpl.cm.YlOrRd, #mpl.cm.RdORYlBu_r,
#     robust=True,
    levels=[-666,0],
    colors=['#cccccc','#f2645a'],
#     cmap="YlOrRd",
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
    },)
```

```python
da_ondmam_som_perc_bavg=da_ondmam_som.where(da_ondmam>=0).count(dim=["x","y"])/da_ondmam_som.count(dim=["x","y"])*100
```

```python
da_ondmam_som_perc_bavg=da_ondmam_som_perc_bavg.to_dataframe()
da_ondmam_som_perc_bavg.drop("spatial_ref",axis=1,inplace=True)
da_ondmam_som_perc_bavg.rename(columns={"precip":"perc_bavg"},inplace=True)
da_ondmam_som_perc_bavg.reset_index(inplace=True)
da_ondmam_som_perc_bavg.time=pd.to_datetime(da_ondmam_som_perc_bavg.time.apply(lambda x: x.strftime('%Y-%m-%d')))
```

```python
da_ondmam_som_perc_bavg[da_ondmam_som_perc_bavg.time.dt.year>=2010]
```

```python
perc_thresh=20
```

```python
da_ondmam_som_perc_bavg[f"ge_{perc_thresh}"]=np.where(da_ondmam_som_perc_bavg.perc_bavg>=perc_thresh,1,0)
```

```python
da_ondmam_som_perc_bavg['consecutive'] = da_ondmam_som_perc_bavg[f"ge_{perc_thresh}"].groupby( \
    (da_ondmam_som_perc_bavg[f"ge_{perc_thresh}"] != da_ondmam_som_perc_bavg[f"ge_{perc_thresh}"].shift()).cumsum()).transform('size') * \
    da_ondmam_som_perc_bavg[f"ge_{perc_thresh}"]

```

```python
da_ondmam_som_perc_bavg[da_ondmam_som_perc_bavg.consecutive>=2][["time","perc_bavg",f"ge_{perc_thresh}","consecutive"]]
```

```python
da_ondmam_som_perc_bavg[da_ondmam_som_perc_bavg.consecutive>=3][["time","perc_bavg",f"ge_{perc_thresh}","consecutive"]]
```
