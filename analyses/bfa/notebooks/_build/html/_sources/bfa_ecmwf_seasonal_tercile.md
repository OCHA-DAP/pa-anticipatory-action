---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: antact
  language: python
  name: antact
---

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
from importlib import reload
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
import glob
import itertools

import math
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {
#         'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
```

```{code-cell} ipython3
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR, "exploration", country_iso3,'chirps')

chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")
ecmwf_country_data_processed_dir = os.path.join(country_data_processed_dir,"ecmwf")
ecmwf_country_data_raw_dir = os.path.join(country_data_raw_dir,"ecmwf")
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
all_dry_spells_list_path=os.path.join(country_data_processed_dir,"dry_spells","full_list_dry_spells.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation_admin1.csv")
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

```{code-cell} ipython3
ds=processing.get_ecmwf_forecast(country_iso3)
```

```{code-cell} ipython3
ds
```

```{code-cell} ipython3
#not using the tprate variable, so drop
ds=ds.drop("tprate")
```

```{code-cell} ipython3
# ds.sel(latitude=ds.latitude[5],longitude=ds.longitude[5],time=ds.time[0],number=50)
```

```{code-cell} ipython3
leadtime=1
seas_len=3
ds_ltsel=ds.sel(step=ds.step.isin(range(leadtime+1,leadtime+1+seas_len)))
```

```{code-cell} ipython3
#compute sum of rainfall across the season
ds_ltseas=ds_ltsel.sum(dim="step",skipna=True,min_count=seas_len)
```

```{code-cell} ipython3
#select the years on which we want to compute the climatological bounds
ds_ltseas_climate=ds_ltseas.sel(time=ds_ltseas.time.dt.year.isin(range(1993,2017))).dropna("number")
```

```{code-cell} ipython3
ds_ltseas_climate
```

```{code-cell} ipython3
ds_ltseas_climate_quantile=ds_ltseas_climate.dropna("number").groupby(ds_ltseas_climate.time.dt.month).quantile(0.33,skipna=True,dim=["time","number"])
```

```{code-cell} ipython3
ds_ltseas_climate_quantile
```

```{code-cell} ipython3
:tags: []

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

```{code-cell} ipython3
ds_ltseas_below_prob=ds_ltseas_below.sum(dim="number")/ds_ltseas_below.count(dim="number")*100
ds_ltseas_below_prob=ds_ltseas_below_prob.rename({"precip":"prob_bavg"})
```

```{code-cell} ipython3
ds_ltseas_below_prob.sel(time="2021-01-01")
```

```{code-cell} ipython3
ds_ltseas_below_prob.prob_bavg.max()
```

```{code-cell} ipython3
g=iri_clip.where(iri_clip.F.dt.month.isin([3]), drop=True).sel(L=1,C=0).prob.plot(
    col="F",
    col_wrap=3,
    cmap=mpl.cm.YlOrRd,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
    },
    figsize=(20,20)
)
df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

```{code-cell} ipython3
# import xarray as xr 
# from matplotlib import pyplot as plt
# plt.ion()
# import cartopy.crs as ccrs
# # fnc='ed8f355c-5823-453c-bbdc-d4c64a038b61.nc'
# # ds=xr.open_dataset(fnc)
# plt.figure()
# ax=plt.axes(projection=ccrs.PlateCarree())
g=ds_ltseas_below_prob.sel(time="2021-01-01").squeeze().prob_bavg.plot( levels=[0,10,20,40,50,60,70,100],colors=['#3054FF','#80FFFF','#FFFFFF','#FFF730','#FFAC00','#FF4701','#CD011E'])
# ax.coastlines()
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red")
# ax.axis("off")
```

```{code-cell} ipython3
ds_ltseas_below_prob.sel(time="2021-01-01")
```

```{code-cell} ipython3
ds_web=xr.open_dataset("../../../Experiments/drought/data/ecmwf/ecmwf_seasonal_bavg_202101.nc")
```

```{code-cell} ipython3
import rioxarray
```

```{code-cell} ipython3
ds_web=rioxarray.open_rasterio("../../../Experiments/drought/data/ecmwf/ecmwf_seasonal_bavg_202101.nc")
```

```{code-cell} ipython3
gdf_adm1=gpd.read_file(adm1_bound_path)
```

```{code-cell} ipython3
ds_web_clip=ds_web.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```{code-cell} ipython3
ds_web_clip=ds_web.rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```{code-cell} ipython3
ds_web_clip
```

```{code-cell} ipython3
ds_ltseas_below_prob_clip=ds_ltseas_below_prob.sel(time="2021-01-01").rio.set_spatial_dims(x_dim="longitude",y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```{code-cell} ipython3
ds_ltseas_below_prob_clip
```

```{code-cell} ipython3
g=ds_ltseas_below_prob_clip.squeeze().prob_bavg.plot( levels=[0,20,22,24,26,28,30,32,34,36,38,40])
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red")
# ax.axis("off")
```

```{code-cell} ipython3
g=ds_web_clip.data.plot( levels=[0,20,22,24,26,28,30,32,34,36,38,40])
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red")
# ax.axis("off")
```

```{code-cell} ipython3
g=ds_web_clip.data.plot( levels=[0,10,20,40,50,60,70,100],colors=['#3054FF','#80FFFF','#FFFFFF','#FFF730','#FFAC00','#FF4701','#CD011E'])
df_bound = gpd.read_file(adm1_bound_path)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="red")
# ax.axis("off")
```

```{code-cell} ipython3

```
