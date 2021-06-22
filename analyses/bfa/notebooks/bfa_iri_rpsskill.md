---
jupyter:
  jupytext:
    formats: ipynb,md,Rmd
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: antact
    language: python
    name: antact
---

# RPSS across Burkina Faso
This notebook explores the Ranked Probability Skill Score (RPSS) for IRI's seasonal forecast. 

Interactive tool several skill measurements: https://iri.columbia.edu/our-expertise/climate/forecasts/verification/   
Data description RPSS: http://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/skill_precip_seasonal.html   
Data files RPSS: http://iridl.ldeo.columbia.edu/home/.jingyuan/.NMME_seasonal_hindcast_verification/.monthly_RPSS_seasonal_hindcast_precip_ELR/.lead3/RPSS/datafiles.html

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import xarray as xr
from pathlib import Path
import os
import sys
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.plotting import plot_raster_boundaries_clip
from src.utils_general.raster_manipulation import fix_calendar
```

```python
country="bfa"
country_iso3="bfa"
config = Config()
parameters = config.parameters(country)
country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3)
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

```python
leadtimes=range(1,5)
```

```python
#load the data
data=[]
#separate data file per leadtime, so load all and combine
for l in leadtimes:
    #using OpenDAP to load the data --> don't have to save it
    remote_data = xr.open_dataset(f"http://iridl.ldeo.columbia.edu/home/.jingyuan/.NMME_seasonal_hindcast_verification/.monthly_RPSS_seasonal_hindcast_precip_ELR/.lead{l}/RPSS/dods",decode_times=False,)
    remote_data=remote_data.rename({"X":"lon","Y":"lat"})
    remote_data=fix_calendar(remote_data,timevar="T")
    remote_data = xr.decode_cf(remote_data)
    data.append(remote_data.RPSS)
comb_lt=xr.DataArray(data=data,dims=["leadtime","T","lat","lon"],
            coords=dict(
            T=remote_data.T,
            lon=remote_data.lon,
            lat=remote_data.lat,
            leadtime=leadtimes))
comb_lt=comb_lt.to_dataset(name="RPSS")
```

```python
comb_lt
```

```python
#clip to area
df_bound=gpd.read_file(adm1_bound_path)
ds_clipped=comb_lt.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping),all_touched=True)
```

```python
ds_clipped
```

```python
#create bins for plotting
bins_clipped=np.linspace(ds_clipped["RPSS"].min(),ds_clipped["RPSS"].max(),10)
#season mapping for plot titles
seasons={1:"JFM",2:"FMA",3:"MAM",4:"AMJ",5:"MJJ",6:"JJA",7:"JAS",8:"ASO",9:"SON",10:"OND",11:"NDJ",12:"DJF"}
```

```python
#plot for specific months of interest
start_months=[6,8]
for l in ds_clipped.leadtime.values:
    start_months_lt=[m-l for m in start_months]
    ds_sel=ds_clipped.sel(leadtime=l,T=ds_clipped.T.dt.month.isin(start_months_lt))
    ds_list_sel=[ds_sel.sel(T=m) for m in ds_sel["T"]]
    list_seasons=[(m+l)%12 if (m+l)%12!=0 else 12 for m in ds_sel["T"].dt.month.values]
    title_list_clipped=[f'{seasons[s]} (issued in {m})' for s,m in zip(list_seasons,ds_sel["T"].dt.strftime("%b").values)]
    fig_clip = plot_raster_boundaries_clip(ds_list_sel, 
                                       adm1_bound_path, 
                                       figsize=(20,3),
                                       title_list=title_list_clipped, 
                                       forec_val="RPSS", 
                                       colp_num=4,
                                       clipped=False,
                                       predef_bins=bins_clipped,
                                       cmap="YlOrRd",
                                       suptitle=f"Leadtime = {l} months")
```

```python
#plot for all months
start_months=ds_clipped["T"].dt.month.values
for l in ds_clipped.leadtime.values:
    start_months_lt=[(m-l)%12 if (m-l)%12!=0 else 12 for m in start_months]
    ds_sel=ds_clipped.sel(leadtime=l,T=ds_clipped.T.dt.month.isin(start_months_lt))
    ds_list_sel=[ds_sel.sel(T=m) for m in ds_sel["T"]]
    list_seasons=[(m+l)%12 if (m+l)%12!=0 else 12 for m in ds_sel["T"].dt.month.values]
    title_list_clipped=[f'{seasons[s]} (issued in {m})' for s,m in zip(list_seasons,ds_sel["T"].dt.strftime("%b").values)]
    fig_clip = plot_raster_boundaries_clip(ds_list_sel, 
                                       adm1_bound_path, 
                                       figsize=(20,10),
                                       title_list=title_list_clipped, 
                                       forec_val="RPSS", 
                                       colp_num=4,
                                       clipped=False,
                                       predef_bins=bins_clipped,
                                       cmap="YlOrRd",
                                       suptitle=f"Leadtime = {l} months")
```

### Archive

```python
# #to check if crs is aligned
# from src.utils_general.plotting import plot_raster_boundaries
# ds_list=[comb_lt.sel(T=m) for m in comb_lt["T"]]
# bins=np.arange(comb_lt['RPSS'].min(),comb_lt['RPSS'].max(),0.05)
# fig_clip = plot_raster_boundaries_clip(ds_list, adm1_bound_path, figsize=(20,30),title_list=comb_lt["T"].values, forec_val="RPSS", colp_num=2,clipped=False,predef_bins=bins)
```
