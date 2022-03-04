### Flooded fraction in Bentiu, Rubkona, and Fangak
Based on conversations, we explore the flooded fraction at more specific areas. 
Namely Bentiu IDP Camp, Rubkona county, and Fangak county. 

To account for wetlands which get flooded each year we also look at anomalies. 

We can see large differences between the two counties. Fangak sees more recurrent annual flooding. Both counties saw the largest flooding in 2021. However, for example a large part of Fangak got already flooded in 2020 while that wasn't the case for Rubkona. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import sys
from datetime import timedelta
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import hvplot.xarray
import matplotlib as mpl
import numpy as np
from scipy import stats
from functools import reduce
import altair as alt
import panel.widgets as pnw
import calendar
import rioxarray

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.flooding.floodscan import floodscan
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
%load_ext rpy2.ipython
```

```R tags=[]
library(tidyverse)
library(lubridate)
```

#### define functions

```R
plotFloodedFraction <- function (df,y_col,facet_col,title){
    df %>%
    ggplot(
        aes_string(
            x = "time",
            y = y_col
        )
    ) +
    stat_smooth(
        geom = "area",
        span = 1/4,
        fill = "#ef6666"
    ) +
    scale_x_date(
        date_breaks = "3 months",
        date_labels = "%b"
    ) +
    facet_wrap(
        as.formula(paste("~", facet_col)),
        scales="free_x",
        ncol=5
    ) +
    ylab(
        "Flooded fraction"
    )+
    xlab(
        "Month"
    )+
    labs(
        title = title
    )+
    theme_minimal()
}
```

```python
iso3="ssd"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
country_data_exploration_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "processed" / iso3
country_data_public_exploration_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / iso3
bentiu_bound_path=country_data_processed_dir / "bentiu" / "bentiu_bounding_box.gpkg"
adm2_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin2_shp"]
```

```python
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
fs_clip=xr.load_dataset(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan.nc')
#I dont fully understand why, these grid mappings re-occur and what they mean
#but if having them, later on getting crs problems when computing stats
fs_clip.SFED_AREA.attrs.pop('grid_mapping')
fs_clip.NDT_SFED_AREA.attrs.pop('grid_mapping')
fs_clip.LWMASK_AREA.attrs.pop('grid_mapping')
fs_clip=fs_clip.rio.write_crs("EPSG:4326",inplace=True)
```

```python
da_clip=fs_clip.SFED_AREA
```

### Stats on Bentiu IDP Camp
We first compute the statistics for Bentiu only. Note that this is a very small area where there are only 2 cells touching the region and none with the centre in the region.  

```python
gdf_bentiu=gpd.read_file(bentiu_bound_path)
```

```python
#check how many cells are included in the region
#NOTE: no cells have their centre within the area.. Only 2 touching it
da_clip.rio.clip(gdf_bentiu.geometry, all_touched = True)
```

```python
df_floodscan_bent=compute_raster_statistics(
        gdf=gdf_bentiu,
        bound_col="id",
        raster_array=da_clip,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_floodscan_bent['year']=df_floodscan_bent.time.dt.year
df_floodscan_bent['month'] = pd.DatetimeIndex(df_floodscan_bent['time']).month
df_floodscan_bent['mean_rolling']=df_floodscan_bent.sort_values('time').mean_id.rolling(10,min_periods=10).mean()
```

We can plot the data over all years. 
We see that it is pretty rare for Bentiu to get flooded. However, in 2007, 2008, and 2021 a large part of Bentiu did see flooding.  
Also noteabily the water in 2008 and 2022 didn't recede during the beginning of the year. 

```python
#note: the 2022 data x-axis is not correct but too hard to fix for purpose
```

```R magic_args="-i df_floodscan_bent -w 40 -h 20 --units cm"
df_plot <- df_floodscan_bent %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_id = mean_id*100)
plotFloodedFraction(df_plot,'mean_id','year',"Flooded fraction of Bentiu")
```

```python
# df_floodscan_bent.to_csv(
#     country_data_exploration_dir / "floodscan" / "bentiu_flood.csv"
# )
```

### Stats on Rubkona
Rubkona is the county of which Bentiu is part. We also analyze the flooding in this county as we might target the whole county instead of only Bentiu

```python
gdf_rub=gdf_adm2[gdf_adm2.ADM2_EN=='Rubkona']
```

```python
#where is Rubkona situated
g=gdf_adm2.boundary.plot()
gdf_rub.plot(ax=g.axes,color="red")
g.axes.axis("off");
```

```python
da_rub=da_clip.rio.clip(gdf_rub.geometry, all_touched = True)
```

```python
#how much data do we have and where is Bentiu
g=da_rub.mean(axis=2).plot()
gdf_rub.boundary.plot(ax=g.axes)
gdf_bentiu.boundary.plot(ax=g.axes,color="red");
```

```python
#check how many cells are included in the region
da_rub.mean(axis=2).count().values
```

```python
bound_col="ADM2_PCODE"
```

```python
df_floodscan_rub=compute_raster_statistics(
        gdf=gdf_rub,
        bound_col=bound_col,
        raster_array=da_clip,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_floodscan_rub['year']=df_floodscan_rub.time.dt.year
df_floodscan_rub['month'] = pd.DatetimeIndex(df_floodscan_rub['time']).month
df_floodscan_rub['mean_rolling']=df_floodscan_rub.sort_values('time')[f"mean_{bound_col}"].rolling(10,min_periods=10).mean()
```

From the graph below we can see quite a different pattern in Rubkona than in Bentiu alone. A small part of the county gets flooded almost every year around June. Later on, we will see that this annual flooding occurs in the south-west of Rubkona. 

Moroever, we do see the flooding of 2007/2008 as well, but this was relatively small compared to the flooding in 2014/2015. Wheras in 2014 Bentiu saw less flooding during 2014 compared to 2007. 

```R magic_args="-i df_floodscan_rub -w 40 -h 20 --units cm"
df_plot <- df_floodscan_rub %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100)
plotFloodedFraction(df_plot,'mean_ADM2_PCODE','year',"Flooded fraction of Rubkona")
```

With the above plot we convert the 2D situation to a 1D situation. We therefore next plot an animation of the 2D situation for a few years. 

We first look at the mean value per month across all years. We can see from there that the south-west is more regularly flooded, though the mean values are really low (note that the axis only goes to 0.5). 

We can also inspect individual years. We animate 2021, 2014, and 2007. We can see from here that in 2021 and 2014 the flooding clearly started from the left bottom corner and spread slowly further in the county. This spreading is quite slow and takes several weeks. 

In 2007 part of the flooding also started from the left bottom, but part also came from the middle-right. 

When we compare 2014 and 2021, we can also see that the flood in 2021 was both wider spread, as well as more intens in certain areas. 

```python
da_rub.groupby('time.month').mean().interactive(loc='bottom').isel(
    month=pnw.Player(name='month', start=0, end=11, 
                  step=1,
                  loop_policy='loop')).plot(
    vmin=0,vmax=0.5)
```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_rub.sel(time=(da_rub.time.dt.year==2021)&(da_rub.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_rub.sel(time=(da_rub.time.dt.year==2014)&(da_rub.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_rub.sel(time=(da_rub.time.dt.year==2007)&(da_rub.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

Based on the above analysis, it could be an idea to look at the anomaly instead of the absolute value. With this, you filter out areas that are commonly flooded. 

For now we just substract the median from the absolute value. If we decide to use this method as part of the trigger, we probably want a more robust method to compute the median. 
The simplest method we apply here is to remove years during which we saw extremely high values by eyeballing the above graphs. More rigorous options could be a curve fit with sigma clipping for positive outliers, or some kind of fourier fit

We could also use percentage-based methods or metrics like the z-score. However, the struggle with these is that our data has very small numbers, and is not normallly distributed. This causes that these scores have a wide range of values. 

Another option could be to based on this analysis mask out certain areas. I.e. if the median (during the rainy season) is above x

```python
#data is not normally distributed
da_rub.plot(bins=np.arange(0.01,1,0.01));
```

```python
#compute the median per pixel and day of the year
#remove the most extreme years from the median computation. 
#This is not very objective!
da_rub_sel=da_rub.where(~da_rub.time.dt.year.isin([2014,2015,2021,2022]))
da_med_rub=da_rub_sel.groupby(da_rub_sel.time.dt.dayofyear).median()
```

We can see that only in the bottom-left there are pixels that are often flooded. Note that even here the numbers are really small (max 2.5% of the pixel flooded), but this is also partly caused by the fact that we take the mean over the full year. 

```python
da_med_rub.mean("dayofyear").plot()
```

```python
#substract median for the given day of the year and pixel
da_anom_rub=xr.apply_ufunc(lambda x,m:x-m,da_rub.groupby('time.dayofyear'),da_med_rub)
```

```python
df_floodscan_anom_rub=compute_raster_statistics(
        gdf=gdf_rub,
        bound_col=bound_col,
        raster_array=da_anom_rub.rio.write_crs("EPSG:4326"),
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_floodscan_anom_rub['year']=df_floodscan_anom_rub.time.dt.year
df_floodscan_anom_rub['month'] = pd.DatetimeIndex(df_floodscan_anom_rub['time']).month
df_floodscan_anom_rub['mean_rolling']=df_floodscan_anom_rub.sort_values('time')[f"mean_{bound_col}"].rolling(10,min_periods=10).mean()
```

```R magic_args="-i df_floodscan_anom_rub -w 40 -h 20 --units cm"
df_plot <- df_floodscan_anom_rub %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100)
plotFloodedFraction(df_plot,'mean_ADM2_PCODE','year',"Flooded fraction minus the median flooded fraction,Rubkona")
```

In the above graph we can see that substracting the median does flatten the curvesournd jun/july while we continue to see a very clear peak towards the end of 2021. 


We should understand the wetlands a little better. Why are they only flooded around July, and then substract again around September? Does this mean that when they are flooded in Sep-Dec, this causes humanitarian needs? 

We need to know this in order to understand if it makes sense to apply a mask on them. 

Another idea in the light of the trigger is to find a better way to understand the pattern of the spatial spread.

Lastly, do we need a better sense of the impact of spatial spread, vs intensity in smaller areas? 


### Stats on Fangak
Other county of interest

```python
gdf_fan=gdf_adm2[gdf_adm2.ADM2_EN=='Fangak']
```

```python
#where is Rubkona situated
g=gdf_adm2.boundary.plot()
gdf_fan.plot(ax=g.axes,color="red")
g.axes.axis("off");
```

```python
da_fan=da_clip.rio.clip(gdf_fan.geometry, all_touched = True)
```

```python
#how much data do we have and where is Bentiu
g=da_fan.mean(axis=2).plot()
gdf_fan.boundary.plot(ax=g.axes)
gdf_bentiu.boundary.plot(ax=g.axes,color="red");
```

```python
#check how many cells are included in the region
da_fan.mean(axis=2,skipna=True).count().values
```

```python
bound_col="ADM2_PCODE"
```

```python
df_floodscan_fan=compute_raster_statistics(
        gdf=gdf_fan,
        bound_col=bound_col,
        raster_array=da_clip,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_floodscan_fan['year']=df_floodscan_fan.time.dt.year
df_floodscan_fan['month'] = pd.DatetimeIndex(df_floodscan_fan['time']).month
df_floodscan_fan['mean_rolling']=df_floodscan_fan.sort_values('time')[f"mean_{bound_col}"].rolling(10,min_periods=10).mean()
```

From the graph below we can see quite a different pattern in Fangak than in Rubkona. A larger part of the county gets flooded almost every year. The peak of the annual flooding is around September whereas in Rubkona this was around July. 
Moroever, we see a larger flooding in 2016 and 2020, while we didn't see this in Rubkona. Contrary the flodings that were seen in 2007 and 2014 in Rubkona, are not out of the extraordinary in Fangak. 

We can still clearly see that also in Fangak, 2021 was the most extreme and the waters havent receded so far in 2022. 

```R magic_args="-i df_floodscan_fan -w 40 -h 20 --units cm"
df_plot <- df_floodscan_fan %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100)
plotFloodedFraction(df_plot,'mean_ADM2_PCODE','year',"Flooded fraction of Fangak")
```

We next plot the raster data again. 
In 2021 we can see that around the whole county there was flooding, with different intensity per pixel. 
In 2021 already a large part of the country was flooded at the beginning of the season. When we look at 2020, we can see that the flooding clearly starts from the bottom-left bottom and then spreads. 

```python
#Mean flooded fraction across all years per month for Fangak county
da_fan.groupby('time.month').mean().drop("spatial_ref").interactive(loc='bottom').isel(
    month=pnw.Player(name='month', start=0, end=11, 
                  step=1,
                  loop_policy='loop')).plot(
    vmin=0,vmax=0.5,cbar_kwargs={"label":"flooded fraction"})
```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_fan.sel(time=(da_fan.time.dt.year==2021)&(da_fan.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_fan.sel(time=(da_fan.time.dt.year==2020)&(da_fan.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_fan.sel(time=(da_fan.time.dt.year==2007)&(da_fan.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

Next we look at the anomaly compared to the median again. The same issues of non-normal distribution apply as to Rubkona

```python
#data is not normally distributed
da_fan.plot(bins=np.arange(0.01,1,0.01));
```

```python
#compute the median per pixel and day of the year
#remove the most extreme years from the median computation. 
#This is not very objective!
da_fan_sel=da_fan.where(~da_fan.time.dt.year.isin([2016,2020,2021,2022]))
da_med_fan=da_fan_sel.groupby(da_fan_sel.time.dt.dayofyear).median()
```

We can again see that mainly the pixels in the bottom-left are regularly flooded. However, this does cover a substantially larger area than in Rubkona. 

```python
da_med_fan.mean("dayofyear").plot()
```

```python
#substract median for the given day of the year and pixel
da_anom_fan=xr.apply_ufunc(lambda x,m:x-m,da_fan.groupby('time.dayofyear'),da_med_fan)
```

```python
df_floodscan_anom_fan=compute_raster_statistics(
        gdf=gdf_fan,
        bound_col=bound_col,
        raster_array=da_anom_fan.rio.write_crs("EPSG:4326"),
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_floodscan_anom_fan['year']=df_floodscan_anom_fan.time.dt.year
df_floodscan_anom_fan['month'] = pd.DatetimeIndex(df_floodscan_anom_fan['time']).month
df_floodscan_anom_fan['mean_rolling']=df_floodscan_anom_fan.sort_values('time')[f"mean_{bound_col}"].rolling(10,min_periods=10).mean()
```

```R magic_args="-i df_floodscan_anom_fan -w 40 -h 20 --units cm"
df_plot <- df_floodscan_anom_fan %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100)
plotFloodedFraction(df_plot,'mean_ADM2_PCODE','year',"Flooded fraction minus the median flooded fraction, Fangak")
```
