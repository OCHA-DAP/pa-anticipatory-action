### NDVI Ethiopia - end of 2021
This notebook explores the NDVI at the end of 2021 and beginning of 2022 in Ethiopia. The reason for this exploration is a request to map the drought conditions at admin3 level. 

As measure we use the percentage of the median NDVI. We use this instead of the absolute NDVI since this allows us to measure the conditions relative to a standard. 

We explore how this measure of NDVI differs from Oct 2021 till beginning of Jan 2022. The first dekad of Jan 2022 is the most current data at the point of writing. We include these months since Oct-Dec is a rainy season for part of the country so we can see if the NDVI changed. There is many uncertainties on how we should interpret the NDVI, e.g. how we should take into account the [seasonal calendar](https://fews.net/file/113527). This is thus solely a statement of the NDVI at the given moments and not directly of drought. 

NDVI, and the percentage of median as we use it, is commonly reported by FewsNet, the most recent ones including NDVI of [21-30 Nov](https://fews.net/east-africa/seasonal-monitor/november-2021), [1-10 Dec](https://fews.net/east-africa/alert/december-29-2021), and [1-10 Jan](https://fews.net/east-africa/seasonal-monitor/december-2021).

We first inspect the raster data but thereafter aggregate to admin3. As aggregation method the median is chosen but there is the classic problem of differences in admin sizes. 

While we first inspect the data per dekad, it was requested to create a graph of cumulative NDVI over the Oct-Dec season. While we are not NDVI experts, it didn't seem sensible to take a sum or median. We therefore instead look at the percentage of dekads where the NDVI is below x% of the median. 

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
from dateutil.relativedelta import relativedelta

path_mod = f"{Path.cwd().parents[3]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.ndvi import (download_ndvi,load_raw_dekad_ndvi,_dekad_to_date)
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
iso3="eth"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
country_data_exploration_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / iso3
ndvi_exploration_dir = country_data_exploration_dir / "ndvi"
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
ndvi_bins=[0,60,70,80,90,95,105,110,120,130,140]
ndvi_labels=["<60","60-70","70-80","90-95","95-105","105-110","110-120","120-130","130-140",">140"]
```

### NDVI rasters


The NDVI raster files can be found [here](https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/africa/east/dekadal/emodis/ndvi_c6/percentofmedian/downloads/dekadal/) and can be automatically downloaded using the code


From the plots we can conclude two main points:
1) During all three dekads there were significant areas that saw below median NDVI (brown-yellow)
2) The pattern is different for the dekads in 2021 than the latest dekad. Where for the first two the worst NDVI conditions are seen in the south and East, while in the latest dekad the worst conditions are in the South but also more up north in the middle of the country. 

```python
#all dekads from first of oct 2021
dekad_list=[[2021,28],[2021,29],[2021,30],[2021,31],[2021,32],[2021,33],[2021,34],[2021,35],[2021,36],[2022,1]]
```

```python
# #example download
# download_ndvi("east",[2021,35])
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
#arghh I just want to select the first dekads of each month but somehow normally selecting the dates
#is not working so a bit of a hack
da_sel=xr.concat([da.sel(date="2021-10-01"),da.sel(date="2021-11-01"),da.sel(date="2021-12-01"),da.sel(date="2022-01-01")],"date")
```

```python
da_sel.plot.imshow(
    col="date",
    levels=ndvi_bins,
    cmap=ListedColormap(ndvi_colors),
    figsize=(40,10),
    cbar_kwargs={
    "orientation": "horizontal",
    "shrink": 0.8,
    "aspect": 40,
    "pad": 0.1,
    'ticks': ndvi_bins,
    },
);
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
fig,axs=plt.subplots(1,2,figsize=(10,20))
gdf_adm3.boundary.plot(ax=axs[0])
axs[0].set_title("ADMIN3 boundaries")
gdf_adm2.boundary.plot(ax=axs[1])
axs[1].set_title("ADMIN2 boundaries");
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
        stats_list=["min","median","mean","max"],
        all_touched=False,
    )
    #would like better way to do this
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
def plt_ndvi_dates(gdf_stats,data_col,colp_num=3,caption=None):
    num_plots = len(gdf_stats.date.unique())
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(10*colp_num,10*rows))
    for i,d in enumerate(gdf_stats.date.unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        gdf_stats[gdf_stats.date==d].plot(ax=ax, column=data_col,
                             legend=True,#if i==num_plots-1 else False,
                            categorical=True,
                cmap=ListedColormap(ndvi_colors)
         )
        ax.set_title(f"{pd.to_datetime(str(d)).strftime('%d-%m-%Y')} till "
                     f"{(pd.to_datetime(str(d))+relativedelta(days=9)).strftime('%d-%m-%Y')}")
        ax.axis("off")
    if caption:
        plt.figtext(0.7, 0.2,caption)
    plt.suptitle("Percent of median NDVI",size=24,y=0.9)
    return fig

```

#### Aggregated to admin3
Below the values per admin3 are shown. We use the same bins as [those used by USGS/FewsNet](https://earlywarning.usgs.gov/fews/product/448). 
We can see the same pattern as we saw with the raw data, which is a good sign. We see that in the beginning of October most of the country saw median conditions. This moved to below median NDVI conditions in the South-East. Towards the end of December the conditions return to median in the east, but below median conditions are seen in the Middle of the country. However, these plots should only be seen as the NDVI and not perse drought conditions as this e.g. depends on the rainy seasons. 

```python
# #this takes a couple of minutes to compute
# #only needed first time, else can load the file below
# gdf_stats_adm3=aggregate_admin(da,gdf_adm3,pcode3_col,bins=ndvi_bins)
```

```python
# #save file
# gdf_stats_adm3[["ADM3_PCODE","date","year","dekad","median_ADM3_PCODE","median_binned_str"]].rename(
#     columns={"median_binned_str":"median_binned_ADM3_PCODE"}).to_csv(
#     ndvi_exploration_dir/"eth_ndvi_adm3_102021-012022.csv")
#read file
df_stats_adm3=pd.read_csv(ndvi_exploration_dir/"eth_ndvi_adm3_102021-012022.csv")
gdf_stats_adm3=gdf_adm3[["geometry",pcode3_col]].merge(df_stats_adm3,how="right")
```

```python
#str for plotting of labels
gdf_stats_adm3["median_binned_str"]=pd.cut(gdf_stats_adm3[f"median_{pcode3_col}"],ndvi_bins,labels=ndvi_labels)
fig=plt_ndvi_dates(gdf_stats_adm3,"median_binned_str",
                   caption="Data is aggregated from raster to admin3 by taking the median",
                  )
# fig.savefig(country_data_exploration_dir / "plots" / "eth_ndvi_adm3_20212022_all.png", facecolor="white", bbox_inches="tight")
```

### "Cumulative" NDVI
We were asked to combine the data from Oct-Dec to one graph. This can be slightly tricky as NDVI can be seen as a cumulative indicator by itself. We chose to report the percentage of dekads that thad a NDVI that was below x% of the median. We think this is a sensible metric, but in the future we would advise to get an opinion on this from expert, e.g. at FAO or USGS

```python
thresh=80
```

```python
#only select the 2021 dates
gdf_2021_adm3=gdf_stats_adm3[gdf_stats_adm3.year==2021]
#count the number of dekads with median below thresh
df_medb_count=gdf_2021_adm3.loc[gdf_2021_adm3.median_ADM3_PCODE<=thresh,
                      [pcode3_col,"median_ADM3_PCODE","date"]].groupby(pcode3_col,as_index=False).count()
#compute percent
df_medb_count["percent"]=df_medb_count.median_ADM3_PCODE/len(gdf_2021_adm3.date.unique())*100
#create gdf again
gdf_medb_count=gdf_adm3[[pcode3_col,"geometry"]].merge(df_medb_count,on=pcode3_col,how="outer")
#nan value means none of the dekads were below thresh so fill them with 0
gdf_medb_count=gdf_medb_count.fillna(0)
```

#### Create map
We create a map with quintile bins as we think this gives enough granularity. We can see that most admin3's in the north didn't see NDVI values that were a lot below median. However, in the south we see this was a common occurence. 

```python
perc_bins=[0,20,40,60,80,100]
#select subset of the original ndvi colors
perc_colors=["#724c04","#d86f27","#f7c90a","#3ca358","#197d71"]
perc_colors.reverse()
perc_labels=["0-20","20-40","40-60","60-80","80-100"]
```

```python
#bin the values
gdf_medb_count["perc_binned"]=pd.cut(gdf_medb_count["percent"],perc_bins,include_lowest=True,labels=perc_labels)
```

```python
g=gdf_medb_count.plot(
    "perc_binned",
    legend=True,
    cmap=ListedColormap(perc_colors)
)
g.set_title(f"Percentage of dekads Oct-Dec 2021 NDVI \n was <={thresh}% of median NDVI");
g.axis("off");
```

### Clip to (agro)pastoral
In Ethiopia the Oct-Dec is not the relevant season for each part of the country and each type of livelihood. We were asked to only focus on the (agro)pastoral areas, and thus below we clip out the admin3's that are not (partially) (agro)pastoral

A dataset of livelihood zones was shared with us. After inspection it turns out this is the same as the [2009 livelihood zone map by FewsNet](https://fews.net/data_portal_download/download?data_file_path=http://shapefiles.fews.net.s3.amazonaws.com/LHZ/ET_LHZ_2009.zip) which is publicly available. We thus use this one. Note that FewsNet also has a [2018 update](https://fews.net/data_portal_download/download?data_file_path=http://s3.amazonaws.com/shapefiles.fews.net/LHZ/ET_LHZ_2018.zip), but since the 2009 data was shared with us, we sticked to that one. 

```python
gdf_lz_fn=gpd.read_file(country_data_exploration_dir/"ET_LHZ_2009/ET_LHZ_2009.shp")
```

```python
#quick plot of livelihood zones
g=gdf_lz_fn.plot("LZTYPE",legend=True)
g.axes.axis("off");
```

```python
#clip removes the admin3s that are not fully covered by (agro)pastoral livelihood zone
gdf_medb_count_clip=gpd.clip(gdf_medb_count,gdf_lz_fn[gdf_lz_fn.LZTYPE.isin(["Pastoral","Agropastoral"])])
```

```python
#determine admin3's that are (partially) (agro)pastoral
gdf_medb_count_clip["include"]=True
gdf_medb_count_include=gdf_medb_count.merge(gdf_medb_count_clip[[pcode3_col,"include"]],on=pcode3_col,how="left")
gdf_medb_count_include["include"]=np.where(gdf_medb_count_include.include.isnull(),False,True)
```

```python
gdf_stats_mask=gdf_medb_count_include.copy()
#set values to nan for not included areas
#makes plotting easier
gdf_stats_mask.loc[gdf_stats_mask.include==False,"perc_binned"]=np.nan
```

```python
g=gdf_stats_mask.plot(
    "perc_binned",
    legend=True,
    figsize=(10,10),
    cmap=ListedColormap(perc_colors),
    missing_kwds={"color": "lightgrey",
                  "label": "non-pastoralist area"}
)
g.set_title(f"Percentage of dekads Oct-Dec 2021 NDVI \n was <={thresh}% of median NDVI");
g.axis("off");
```
