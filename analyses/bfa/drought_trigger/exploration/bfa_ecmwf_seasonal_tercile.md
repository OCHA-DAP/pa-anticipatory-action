---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: antact_toolbox
    language: python
    name: antact_toolbox
---

### Create ecmwf's seasonal tercile forecast

**Note: this notebook was made a long time ago in a half-state of understanding and was not finished. So I am not sure if it makes any sense, but leaving it here for reference. Feel free to delete as well**
We would like to have ecmwf's forecast in tercile format. As this is the source with the longest historical record so it will help in the trigger development. 
They don't publish their data in this tercile format. 
They do publish a graphical product [here](https://apps.ecmwf.int/webapps/opencharts/products/seasonal_system5_standard_rain?area=AFRI&base_time=202012010000&stats=terc-1&valid_time=202101010000), so our goal would be to get the data in that format. 

We have a scripts to retrieve the monthly data, in `src/indicators/drought/ecmwf_seasonal` 
This notebook attempts to convert this data to tercile data. However, the results don't come close, which worries me. 
Even using that script but retrieving the info in two different ways results in different results while I would expect them to be the same. Note that this was quickly put together, so there might be a mistake but this should be understood. 

We also have a script that is not actively used but I added now to directly retrieve the tercile probabilities when downloading the data. This was checked by ECMWF. 
The reason I would prefer to retrieve the terciles from the monthly data instead of using these terciles, is because we then need to save less raw data. 

However the data from this script does produce a different outcome again, which is more similair to the one on the website, though it still seems to differ.

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import pandas as pd
import numpy as np
import xarray as xr
import rioxarray

import math
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
reload(processing)
```

```python
country="bfa"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)

ecmwf_country_data_processed_dir = os.path.join(country_data_processed_dir,"ecmwf")
ecmwf_country_data_raw_dir = os.path.join(country_data_raw_dir,"ecmwf")
ecmwf_glb_data_exploration_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / config.GLOBAL_ISO3 / "ecmwf"

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

For now we are focussing on leadtime=1

```python
leadtime=1
seas_len=3
```

### Load data how we normally do it

```python
da_for=processing.get_ecmwf_forecast_by_leadtime(country_iso3)
```

```python
da_for
```

```python
da_for_lt=da_for.sel(leadtime=leadtime)
```

```python
#compute total 3monthly rainfall
da_for_lt_seas = (
        da_for_lt.rolling(time=seas_len, min_periods=seas_len)
        .sum()
        .dropna(dim="time", how="all")
    )#.dropna("number")
```

```python
# define the years that are used to define the climatology
da_for_lt_seas_climate = da_for_lt_seas.sel(
    time=da_for_lt_seas.time.dt.year.isin(range(1993, 2017))
)
```

```python
#compute below average cap
da_for_lt_seas_climate_quant=da_for_lt_seas_climate.groupby(da_for_lt_seas_climate.time.dt.month).quantile(0.33,skipna=True,dim=["time","number"])
```

```python
#all cells with below should be 1
#all cells with above should be 0
#all cells with nan should be nan
list_ds_seass=[]
for s in np.unique(da_for_lt_seas.time.dt.month):
    da_for_lt_seas_selm=da_for_lt_seas.sel(time=da_for_lt_seas.time.dt.month==s)
    bavg_th=da_for_lt_seas_climate_quant.sel(month=s)
    #keep original values of cells above bavg th or are nan, others set to -666
    #i.e. indicating those received below average
    da_for_lt_seas_onlybelow=da_for_lt_seas_selm.where((da_for_lt_seas_selm.isnull())|(da_for_lt_seas_selm>bavg_th),-666)
                                                      
    
    #set cells receiving normal/below average to 0
    da_for_lt_seas_below=da_for_lt_seas_onlybelow.where((da_for_lt_seas_onlybelow.isnull())|(da_for_lt_seas_onlybelow==-666),0)
    #set cells below average to 1
    da_for_lt_seas_below=da_for_lt_seas_below.where((da_for_lt_seas_onlybelow!=-666),1)
    list_ds_seass.append(da_for_lt_seas_below)
da_for_lt_seas_below=xr.concat(list_ds_seass,dim="time")
```

```python
#compute probability of below average
da_for_lt_seas_below_prob=da_for_lt_seas_below.sum(dim="number")/da_for_lt_seas_below.count(dim="number")*100
```

Aaahh the results from the different steps look as expected, but the probabilities of below average are waaay higher than show on the graphical product on ecmwf's website. 
Also it is not normal that probabilities are so high, so it seems something is going wrong

```python
da_for_lt_seas.sel(time="2021-02-01",latitude=da_for_lt_seas.latitude[0],longitude=da_for_lt_seas.longitude[0])
```

```python
da_for_lt_seas_climate_quant.sel(month=2,latitude=da_for_lt_seas.latitude[0],longitude=da_for_lt_seas.longitude[0])
```

```python
da_for_lt_seas_below.sel(time="2021-02-01",latitude=da_for_lt_seas.latitude[0],longitude=da_for_lt_seas.longitude[0])
```

```python
da_for_lt_seas_below_prob.sel(time="2021-02-01",latitude=da_for_lt_seas.latitude[0],longitude=da_for_lt_seas.longitude[0])
```

```python
da_for_lt_seas_below_prob.max()
```

```python
g=da_for_lt_seas_below_prob.sel(time="2021-02-01").squeeze().plot(
    levels=[0,10,20,40,50,60,70,100],
    colors=['#3054FF','#80FFFF','#FFFFFF','#FFF730','#FFAC00','#FF4701','#CD011E'])
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red");
```

### Load data using old method

```python
ds=processing.get_ecmwf_forecast(country_iso3)
```

```python
leadtime=1
seas_len=3
ds_ltsel=ds.sel(step=ds.step.isin(range(leadtime+1,leadtime+1+seas_len)))
```

```python
leadtime=1
seas_len=3
ds_ltsel=ds.sel(step=ds.step.isin(range(leadtime+1,leadtime+1+seas_len)))
```

```python
#compute sum of rainfall across the season
ds_ltseas=ds_ltsel.sum(dim="step",skipna=True,min_count=seas_len)
```

```python
#select the years on which we want to compute the climatological bounds
ds_ltseas_climate=ds_ltseas.sel(time=ds_ltseas.time.dt.year.isin(range(1993,2017))).dropna("number")
```

```python
ds_ltseas_climate
```

```python
ds_ltseas_climate_quantile=ds_ltseas_climate.dropna("number").groupby(ds_ltseas_climate.time.dt.month).quantile(0.33,skipna=True,dim=["time","number"])
```

```python
ds_ltseas_climate_quantile
```

```python tags=[]
list_ds_seass=[]
for s in np.unique(ds_ltseas.time.dt.month):
    ds_ltseas_selm=ds_ltseas.sel(time=ds_ltseas.time.dt.month==s)
    bavg_th=ds_ltseas_climate_quantile.sel(month=s)
    #keep original values of cells above bavg th or are nan, others set to 1
    #i.e. indicating those received below average
    ds_ltseas_onlybelow=ds_ltseas_selm.where((ds_ltseas_selm.isnull())|(ds_ltseas_selm>bavg_th),1)
    #set cells receiving normal/below average to 0
    ds_ltseas_below=ds_ltseas_onlybelow.where((ds_ltseas_onlybelow.isnull())|(ds_ltseas_onlybelow<=bavg_th),0)
    list_ds_seass.append(ds_ltseas_below)
ds_ltseas_below=xr.concat(list_ds_seass,dim="time")
```

```python
ds_ltseas_below_prob=ds_ltseas_below.sum(dim="number")/ds_ltseas_below.count(dim="number")*100
ds_ltseas_below_prob=ds_ltseas_below_prob.rename({"precip":"prob_bavg"})
```

```python
ds_ltseas_below_prob.sel(time="2021-01-01")
```

```python
ds_ltseas_below_prob.prob_bavg.max()
```

```python
g=ds_ltseas_below_prob.sel(time="2021-01-01").squeeze().prob_bavg.plot(
    levels=[0,10,20,40,50,60,70,100],
    colors=['#3054FF','#80FFFF','#FFFFFF','#FFF730','#FFAC00','#FF4701','#CD011E'])
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red");
```

```python
ds_ltseas_below_prob.sel(time="2021-01-01")
```

### Open data computed with tercile download script

```python
ds_web=rioxarray.open_rasterio(ecmwf_glb_data_exploration_dir / "ecmwf_seasonal_bavg_202101.nc")
```

```python
ds_web
```

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
```

```python
ds_web_clip=ds_web.rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
ds_web_clip
```

```python
ds_ltseas_below_prob_clip=ds_ltseas_below_prob.sel(time="2021-01-01").rio.set_spatial_dims(x_dim="longitude",y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
ds_ltseas_below_prob_clip
```

```python
g=ds_web_clip.data.plot( 
    levels=[0,10,20,40,50,60,70,100],
    colors=['#3054FF','#80FFFF','#FFFFFF','#FFF730','#FFAC00','#FF4701','#CD011E']
)
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red")
# ax.axis("off")
```
