### ECMWF realtime data
Exploratory notebook to understand the structure of ecmwfs realtime data. 
We thereafter also compute the trigger metric for different methodologies. 

ECMWFs realtime data is shared with us through an Amazon bucket. For testing I have transferred the relevant files to our private GDrive. 

The filenames are cryptic and are explained here https://confluence.ecmwf.int/pages/viewpage.action?pageId=111155348
For the Malawi trigger we are interested in the TL4 files

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import pandas as pd
import numpy as np

import xarray as xr
import fnmatch
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from matplotlib.colors import ListedColormap

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
from src.utils_general.raster_manipulation import compute_raster_statistics
reload(processing);
```

#### Set config values

```python
country_iso3="mwi"
config=Config()
parameters = config.parameters(country_iso3)

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
ecmwf_realtime_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / config.RAW_DIR / "glb" / "ecmwf"
```

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
```

We select all files that contain the seasonal forecast with monthly mean, i.e. that Have T4L in their filename

```python
filepath_list=list(ecmwf_realtime_dir.glob('*T4L*1'))
```

We first load one file without any concatenation and processing to understand the fileformat.  

The files are very cryptic but they contain a dataType which can be "em" or "fcmean". As far as I understand "em" is the mean across all ensembles while "fcmean" is the value per ensemble 

Then there also is the "numberOfPoints". This has to do with the geographical area, since we receive the forecast for several geographical areas. 
Just by testing and looking at the latitude/longitude range I figured 384 is the one matching Malawi but no idea if there is a better way to set this


Lastly, cfgrib automatically loads a coordinate names `valid_time` which is then loaded as the date after the end date of the forecast validity, which is very confusing. Due to [this Github issue](https://github.com/ecmwf/cfgrib/issues/97) we discovered you can set backend kwargs on which time dimensions to load. Now `forecastMonth` indicates the leadtime (ranging from 1 to 7) and `verifying_time` indicates the start date of the month the forecast is valid for. 

```python
xr.load_dataset(filepath_list[0],engine="cfgrib",filter_by_keys={'numberOfPoints': 384, 'dataType': 'fcmean'},backend_kwargs=dict(time_dims=('time', 'forecastMonth','verifying_time')))
```

Next we load all files at once

```python
def _preprocess_monthly_mean_dataset(ds_date: xr.Dataset):
        
         return (
            ds_date
            .expand_dims("time")
            #surface is empty
             .drop_vars("surface")
        )
```

```python
#i would think setting concat_dim=["time","step"] makes more sense 
#but get an error "concat_dims has length 2 but the datasets passed are nested in a 1-dimensional structure"
#it seems to work thought when using concat_dim="time" but would have to test once we have data from several dates.. 
ds=xr.open_mfdataset(filepath_list, engine = "cfgrib",filter_by_keys={'numberOfPoints': 384, 'dataType': 'fcmean'},concat_dim=["forecastMonth"],
                     combine="nested",
                     preprocess=lambda d: _preprocess_monthly_mean_dataset(d),
                     #TODO: we currently don't use "verifying_time" so might want to remove that
                    backend_kwargs=dict(time_dims=('time', 'forecastMonth','verifying_time')))
```

```python
#rename to have consistent naming with the API data (though actually maybe we would want to change the API data as well.. )
ds=ds.rename({"forecastMonth":"step"})
```

```python
ds
```

```python
ds=processing.convert_tprate_precipitation(ds)
ds=ds.rio.write_crs("EPSG:4326",inplace=True)
da=ds.precip
```

```python
da
```

Now that we have the raster data, we can compute the statistics. 
We also take the 50% probability value as this is what is being used in the trigger. 

For the trigger we are interested in the values of the forecast published in December which is predicting February, and specifically for the Southern region. So we select on this data. 

```python
def compute_raster_stats_test(da,prob=0.5,resolution=None,all_touched=False,pcode_col="ADM1_EN",add_col=["ADM1_PCODE"]):
    for t in da.time.values:
        df_lt_list=[]
        for lt in da.step.values:
            ds_sel_lt = da.sel(step=lt,time=t)
            # reproject only working on 2D&3D arrays
            # so only do after selecting the date and leadtime..
            if resolution is not None:
                ds_sel_lt = ds_sel_lt.rio.reproject(
                    ds_sel_lt.rio.crs,
                    resolution=resolution,
                    resampling=Resampling.nearest,
                    nodata=np.nan,
                )
                # we use longitude and latitude in other places
                # so stick to those namings
                ds_sel_lt = ds_sel_lt.rename(
                    {"x": "longitude", "y": "latitude"}
                )
            df_lt = compute_raster_statistics(
                gdf_adm1,
                pcode_col,
                ds_sel_lt,
                lon_coord="longitude",
                lat_coord="latitude",
                all_touched=all_touched,
            )
            df_lt_list.append(df_lt)
        df = pd.concat(df_lt_list)
        df = df.merge(
            gdf_adm1[[pcode_col] + add_col], on=pcode_col, how="left"
        )
        df["date"] = t
        df_quant = df.groupby(
        ["date", pcode_col, "step"]+add_col, as_index=False
    ).quantile(prob)
    return df_quant
```

```python
df_center=compute_raster_stats_test(da)
df_center[(df_center.step==3)&(df_center.ADM1_EN=="Southern")]
```

```python
df_mask=compute_raster_stats_test(da,resolution=0.05)
df_mask[(df_mask.step==3)&(df_mask.ADM1_EN=="Southern")]
```

```python
df_allt=compute_raster_stats_test(da,all_touched=True)
df_allt[(df_allt.step==3)&(df_allt.ADM1_EN=="Southern")]
```

From the statistics below we can see that the "mean_ADM1_EN" column is above the cap of 210 for each of the three methods. 
This means that for none of the methods the trigger is reached. 

However, they are all very close to 210 so we want to make sure we dont make mistakes :) 


### Plotting
Lastly we plot the data to understand the patterns better. We can see that approximately half of the southern region sees cells below the threshold. 

```python
da_feb=da.sel(step=3,time="2021-12-01").squeeze().quantile(0.5, dim="number")
```

```python
#select part of latitude values for nicer plot
da_feb_plt=da_feb.where(da_feb.latitude<=-9,drop=True)
```

```python
#set bins
bins=[0, 50, 100, 150, 210.1, 230, 250, 300, 350]
cmap=ListedColormap(
    [
        "#c25048",
        "#f2645a",
        "#f7a29c",
        "#fce0de",
        "#dbedfb",
        "#cce5f9",
        "#66b0ec",
        "#007ce0",
        "#0063b3",
    ])
```

We experiment with different color scales and bins to get the clearest visual

```python
g=da_feb_plt.plot(levels=bins,cmap="Blues",figsize=(10,15),)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");

g.axes.set_title(
    f"Forecasted monthly precipitation \n with 50% "
    f"probability for February 2022",
    size=14,
)
plt.figtext(0, 0.05, f"Forecast published on 5 December 2021",size=14);
```

```python
g=da_feb_plt.plot(levels=bins,cmap=cmap,figsize=(10,15),)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");

g.axes.set_title(
    f"Forecasted monthly precipitation \n with 50% "
    f"probability for February 2022",
    size=14,
)
plt.figtext(0, 0.05, f"Forecast published on 5 December 2021",size=14);
```

```python
#set bins with grey area
bins_grey=[0, 50, 100, 150, 190,  210.1, 230, 250, 300, 350]
cmap_grey=ListedColormap(
    [
        "#c25048",
        "#f2645a",
        "#f7a29c",
        "#fce0de",
        "#d1d1d1",
        "#eeeeee",
        "#cce5f9",
        "#66b0ec",
        "#007ce0",
        "#0063b3",
    ])
```

```python
g=da_feb_plt.plot(levels=bins_grey,cmap=cmap_grey,figsize=(10,15),)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");

g.axes.set_title(
    f"Forecasted monthly precipitation \n with 50% "
    f"probability for February 2022",
    size=14,
)
plt.figtext(0, 0.05, f"Forecast published on 5 December 2021",size=14);
```

```python
g=da_feb_plt.plot.contourf(levels=bins,cmap=cmap,figsize=(10,15),)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");

g.axes.set_title(
    f"Forecasted monthly precipitation \n with 50% "
    f"probability for February 2022",
    size=14,
)
plt.figtext(0, 0.05, f"Forecast published on 5 December 2021",size=14);
```

```python
g=da_feb_plt.plot.contourf(levels=bins,cmap=cmap,figsize=(10,15),)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");

g.axes.set_title(
    f"Forecasted monthly precipitation \n with 50% "
    f"probability for February 2022",
    size=14,
)
plt.figtext(0, 0.05, f"Forecast published on 5 December 2021",size=14);
```

Inspect values of cells with their center in the Southern region

```python
gdf_reg=gdf_adm1[gdf_adm1.ADM1_EN=="Southern"]
```

```python
da_feb_clip=da_feb.rio.write_crs("EPSG:4326").rio.clip(gdf_reg["geometry"])
```

```python
da_feb_clip.values
```

```python
da_feb_clip.mean().values
```

```python
(da_feb_clip.where(da_feb_clip<=210).count()/da_feb_clip.count()).values
```

```python
g=da_feb_clip.plot.imshow(levels=bins,cmap=cmap,figsize=(10,15),extend="max")
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
```

```python
g=da_feb_clip.plot.imshow(levels=bins_grey,cmap=cmap_grey,figsize=(10,15),extend="max")
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
```

```python
g=da_feb_clip.plot.imshow(levels=bins,cmap="Blues",figsize=(10,15),extend="max")
gdf_adm1.boundary.plot(ax=g.axes,color="grey");

g.axes.set_title(
    f"Forecasted monthly precipitation \n with 50% "
    f"probability for February 2022",
    size=14,
)
plt.figtext(0, 0.05, f"Forecast published on 5 December 2021",size=14);
```
