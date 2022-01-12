### NDVI Ethiopia - end of 2021
This notebook explores the NDVI at the end of 2021 and beginning of 2022 in Ethiopia. The reason for this exploration is a request to map the drought conditions at admin3 level. 

As measure we use the percentage of the median NDVI. We use this instead of the absolute NDVI as drought conditions should be seen as relative to a standard. 

We explore how this measure of NDVI differs during three dekads: 21-30 Nov 2021, 1-10 Dec 2021, and 1-10 Jan 2022. The last one is the most current data at the point of writing. However, we are not in the rainy season anymore, so the comparison to the other two is done to see if the NDVI changed since then. They did and thus it is unclear to me which of them best captures the actual drought conditions. 
The NDVI's of [21-30 Nov](https://fews.net/east-africa/seasonal-monitor/november-2021) and [1-10 Dec](https://fews.net/east-africa/alert/december-29-2021) have been used in the linked FN reports. 

We first inspect the raster data but thereafter aggregate to admin3. As aggregation method the median is chosen but there is the classic problem of differences in admin sizes. 

Three open questions: 
- Do we think the map at admin3 level will have added value for the HNO? 
- Should we use the NDVI of 1-10 Dec 2021 or 1-10 Jan 2022? 
- Is the median the most appropiate method of aggregation? 

```python
%load_ext autoreload
%autoreload 2

import geopandas as gpd
from pathlib import Path
import sys
import os
import math

import xarray as xr
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.ndvi import (download_ndvi, process_ndvi, load_processed_ndvi, 
load_dekad_ndvi,_date_to_dekad,_dekad_to_date)
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
iso3="eth"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
adm2_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin2_shp"]
adm3_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin3_shp"]
```

```python
gdf_adm3=gpd.read_file(adm3_bound_path)
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
pcode2_col="ADM2_PCODE"
pcode3_col="ADM3_PCODE"
```

```python
ndvi_colors=["#724c04","#d86f27","#f0a00f","#f7c90a","#fffc8b","#e0e0e0","#86cb69","#3ca358","#39a458","#197d71","#146888","#092c7d"]
ndvi_bins=[0,60,70,80,90,95,105,110,120,130,140,200]
```

### NDVI rasters


The NDVI raster files are downloaded manuallly from [here](https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/africa/east/dekadal/emodis/ndvi_c6/percentofmedian/downloads/dekadal/). Code is almost ready to do this automatically but due to some issues and time pressure, this was preferred


From the plots we can conclude two main points:
1) During all three dekads there were significant areas that saw below median NDVI (brown-yellow)
2) The pattern is different for the dekads in 2021 than the latest dekad. Where for the first two the worst NDVI conditions are seen in the south and East, while in the latest dekad the worst conditions are in the South but also more up north in the middle of the country. 

```python
dekad_list=[[2021,33],[2021,34],[2022,1]]
```

```python
da_dekad_list=[]
for year_dekad in dekad_list:
    da_dekad=load_raw_dekad_ndvi("east",year_dekad[0],year_dekad[1])
    da_dekad_clip=da_dekad.rio.clip(gdf_adm3.geometry, drop=True, from_disk=True)
    da_dekad_clip=da_dekad_clip.assign_coords({"year":year_dekad[0],"dekad":year_dekad[1],"date":_dekad_to_date(year_dekad[0],year_dekad[1])}).expand_dims(["date"])
    da_dekad_list.append(da_dekad_clip)
da=xr.concat(da_dekad_list,"date")
```

```python
da.plot.imshow(
    col="date",
    levels=ndvi_bins,
    cmap=ListedColormap(ndvi_colors),
    figsize=(30,10),
    cbar_kwargs={
    "orientation": "horizontal",
    "shrink": 0.8,
    "aspect": 40,
    "pad": 0.1,
    'ticks': ndvi_bins,
    },
)
```

## Aggregation


Next we aggregate the data to the admin level. In general it is not recommended to aggregate to all these admins without a clear goal. But since requested we do an attempt. 

There are different methods of aggregation:
- min
- max
- mean
- median
- perc of area

Due to high fluctuations in the data, we estimate the min and max to not represent the situation accurately. 
The percentage of area brings an extra complexity as we then have to set a threshold. We therefore choose to not do that at this point. However, an option could be to set a threshold, e.g. <=100, and look at the perc of each adm being below that threshold. 

Based on the above we suggest to use the mean or median. Due to relatively high fluctuations, I suggest to use the median.


#### Adm3 vs Adm2
It was requested to aggregate the data to adm3. We quickly inspect how they look compared to the admin2's.We can see that there are more than 1000 admin3's. However, due to the high resolution of the data (what is the exact resolution?) and the large size of the country, we still expect it to be okay to aggregate to admin3 if needed.  

```python
print(f"Number of adm3s: {len(gdf_adm3)}")
print(f"Number of adm2s: {len(gdf_adm2)}")
```

```python
gdf_adm3.boundary.plot()
```

```python
gdf_adm2.boundary.plot()
```

### Define functions

```python
def aggregate_admin(da,gdf,pcode_col,bins=None):
    da_clip = da.rio.clip(gdf.geometry, drop=True, from_disk=True)
    df_stats=compute_raster_statistics(
        gdf=gdf,
        bound_col=pcode_col,
        raster_array=da_clip,
        lon_coord="x",
        lat_coord="y",
        stats_list=["min","median","mean","max","std","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=False,
    )
    #find better way to do this
    #dont understand why but df_stats_adm2.convert_dtypes() is not working
    df_stats[f"mean_{pcode_col}"]=df_stats[f"mean_{pcode_col}"].astype("float64")
    df_stats[f"median_{pcode_col}"]=df_stats[f"median_{pcode_col}"].astype("float64")
    if bins is not None: 
        df_stats["mean_binned"]=pd.cut(df_stats[f"mean_{pcode_col}"],bins)
        df_stats["median_binned"]=pd.cut(df_stats[f"median_{pcode_col}"],bins)
    gdf_stats=gdf[[pcode_col,"geometry"]].merge(df_stats,on=pcode_col)
    return gdf_stats
```

```python
def plt_ndvi_dates(gdf_stats,data_col,colp_num=3):
    num_plots = len(gdf_stats.date.unique())
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(30,10))
    for i,d in enumerate(gdf_stats.date.unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        gdf_stats[gdf_stats.date==d].plot(ax=ax, column=data_col,
                             legend=True,# if i==num_plots-1 else False,
                cmap=ListedColormap(ndvi_colors)#,levels=ndvi_bins
         )
        ax.set_title(f"{data_col} of % NDVI of median for {pd.to_datetime(str(d)).strftime('%Y-%m-%d')}")
        ax.axis("off")
```

#### Aggregated to admin3
Below the values per admin3 are shown. We use the same bins as [those used by USGS/FewsNet](https://earlywarning.usgs.gov/fews/product/448). 
We can see the same pattern as we saw with the raw data, which is a good sign. The dissimilarity between the pattern beginning of December and beginning of January is even more clear from these plots. It remains an open question which we should use.. 

```python
#this takes a couple of minutes to compute
gdf_stats_adm3=aggregate_admin(da,gdf_adm3,pcode3_col,bins=ndvi_bins)
```

```python
plt_ndvi_dates(gdf_stats_adm3,"median_binned")
```

To check, we do the same computations for the adm2 to see if the general patterns are the same. As we can see they indeed generally match though as expected the values at admin2 are more smoothed

```python
gdf_stats_adm2=aggregate_admin(da,gdf_adm2,pcode2_col,bins=ndvi_bins)
```

```python
plt_ndvi_dates(gdf_stats_adm2,"median_binned")
```
