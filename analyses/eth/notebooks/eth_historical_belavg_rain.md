```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.chirps_rainfallobservations import compute_seasonal_lowertercile_raster, \
get_filepath_seasonal_lowertercile_raster
```

#### Set config values

```python
iso3="eth"
config=Config()
parameters = config.parameters(iso3)

public_data_dir = os.path.join(config.DATA_DIR, config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(public_data_dir,config.RAW_DIR,iso3)
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
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
    levels=[-666,0],
    colors=['#cccccc','#f2645a'],
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
    levels=[-666,0],
    colors=['#cccccc','#f2645a'],
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
