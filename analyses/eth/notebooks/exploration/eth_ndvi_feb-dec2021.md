### NDVI Ethiopia - JJAS 2021
Same stuff as `eth_ndvi.md` but with no explanation and also including the JJAS season. 

Some things take long to compute, but most intermediate products are saved and can thus be loaded. 
Will clean up later. 

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

from src.indicators.drought.ndvi import (download_ndvi,load_raw_dekad_ndvi,
                                         _dekad_to_date, _date_to_dekad)
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
#feb-may 2021
dekad_list_fmam=[[2021,d] for d in range(4,16)]
```

```python
#jun-sep 2021
dekad_list_jjas=[[2021,d] for d in range(16,28)]
```

```python
#oct-dec 2021
dekad_list_ond=[[2021,d] for d in range(28,37)]
```

```python
#jan 2022
dekad_list_jan=[[2022,1]]#,[2022,2]]
```

```python
# # #example download
# download_ndvi("east",[2021,16])
```

```python
all_dekads_dict={"fmam":dekad_list_fmam,"jjas":dekad_list_jjas,"ond":dekad_list_ond,"jan":dekad_list_jan}
```

```python
all_dekads_list=[]
for v in all_dekads_dict.values():
    all_dekads_list +=v
```

```python
# #only compute when adding new data
# #takes around 20 mins
# da_dekad_list=[]
# for year_dekad in all_dekads_list:
#     da_dekad=load_raw_dekad_ndvi("east",year_dekad[0],year_dekad[1])
#     da_dekad_clip=da_dekad.rio.clip(gdf_adm3.geometry, drop=True, from_disk=True)
#     da_dekad_clip=da_dekad_clip.assign_coords({"year":year_dekad[0],"dekad":year_dekad[1],"date":_dekad_to_date(year_dekad[0],year_dekad[1])}).expand_dims(["date"])
#     da_dekad_list.append(da_dekad_clip)
# da=xr.concat(da_dekad_list,"date")
```

```python
# da.to_netcdf(ndvi_exploration_dir/"eth_raster_feb2021_jan2022.nc")
da=xr.load_dataset(ndvi_exploration_dir/"eth_raster_feb2021_jan2022.nc")
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
    gdf_stats["median_binned_str"]=pd.cut(gdf_stats_adm3[f"median_{pcode_col}"],ndvi_bins,labels=ndvi_labels)
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
# # #this takes a couple of minutes to compute
# # #only needed first time, else can load the file below
# gdf_stats_adm3=aggregate_admin(da.rio.write_crs("EPSG:4326"),gdf_adm3,pcode3_col,bins=ndvi_bins)
# #save file
# gdf_stats_adm3[["ADM3_PCODE","date","year","dekad","median_ADM3_PCODE","median_binned_str"]].rename(
#     columns={"median_binned_str":"median_binned_ADM3_PCODE"}).to_csv(
#     ndvi_exploration_dir/"eth_ndvi_adm3_022021-052021.csv")
```

```python
#read files
df_stats_adm3_junjan=pd.read_csv(ndvi_exploration_dir/"eth_ndvi_adm3_062021-012022.csv",parse_dates=['date'])
df_stats_adm3_fmam=pd.read_csv(ndvi_exploration_dir/"eth_ndvi_adm3_022021-052021.csv",parse_dates=['date'])
df_stats_adm3=pd.concat([df_stats_adm3_fmam,df_stats_adm3_junjan])
gdf_stats_adm3=gdf_adm3[["geometry",pcode3_col]].merge(df_stats_adm3,how="right")
```

### "Cumulative" NDVI
We were asked to combine the data from Oct-Dec to one graph. This can be slightly tricky as NDVI can be seen as a cumulative indicator by itself. We chose to report the percentage of dekads that thad a NDVI that was below x% of the median. We think this is a sensible metric, but in the future we would advise to get an opinion on this from expert, e.g. at FAO or USGS

```python
thresh=80
```

```python
def compute_dekads_below_thresh(gdf,pcode_col,value_col):
    #count the number of dekads with median below thresh
    df_medb_count=gdf.loc[gdf[value_col]<=thresh,
                          [pcode_col,value_col,"date"]].groupby(pcode_col,as_index=False).count()
    #compute percent
    df_medb_count["percent"]=df_medb_count[value_col]/len(gdf.date.unique())*100
    #create gdf again
    gdf_uniq=gdf[[pcode3_col,"geometry"]].drop_duplicates()
    gdf_medb_count=gdf_uniq.merge(df_medb_count,on=pcode_col,how="outer")
    #nan value means none of the dekads were below thresh so fill them with 0
    gdf_medb_count=gdf_medb_count.fillna(0)
    #bin the values
    gdf_medb_count["perc_binned"]=pd.cut(gdf_medb_count["percent"],perc_bins,include_lowest=True,labels=perc_labels)
    return gdf_medb_count
```

```python
def clip_lz(gdf,lztype):
    gdf_lz_fn=gpd.read_file(country_data_exploration_dir/"ET_LHZ_2009/ET_LHZ_2009.shp")
    #clip removes the admin3s that are not fully covered by (agro)pastoral livelihood zone
    gdf_clip=gpd.clip(gdf,gdf_lz_fn[gdf_lz_fn.LZTYPE.isin(lztype)])
    #determine admin3's that are (partially) (agro)pastoral
    gdf_clip["include"]=True
    gdf_include=gdf.merge(gdf_clip[[pcode3_col,"include"]],on=pcode3_col,how="left")
    gdf_include["include"]=np.where(gdf_include.include.isnull(),False,True)
    gdf_include.loc[gdf_include.include==False,"perc_binned"]=np.nan
    return gdf_include                      
```

```python
def plot_ndvi(gdf,title):
    #use when no missing data
    g=gdf.plot(
        "perc_binned",
        legend=True,
        figsize=(10,10),
        cmap=ListedColormap(perc_colors),
    )
    g.set_title(title);
    g.axis("off");
    return g
```

```python
def plot_mask(gdf,label_missing,title):
    g=gdf.plot(
        "perc_binned",
        legend=True,
        figsize=(10,10),
        cmap=ListedColormap(perc_colors),
        missing_kwds={"color": "lightgrey",
                      "label": label_missing}
    )
    g.set_title(title);
    g.axis("off");
    return g
```

```python
perc_bins=[0,20,40,60,80,100]
#select subset of the original ndvi colors
perc_colors=["#724c04","#d86f27","#f7c90a","#3ca358","#197d71"]
perc_colors.reverse()
perc_labels=["0-20","20-40","40-60","60-80","80-100"]
```

```python
gdf_stats_adm3_ond=gdf_stats_adm3[gdf_stats_adm3.date.isin(
    [_dekad_to_date(dek[0],dek[1]) for dek in dekad_list_ond])]
gdf_medb_count_ond=compute_dekads_below_thresh(gdf_stats_adm3_ond,pcode3_col,"median_ADM3_PCODE")
gdf_stats_mask_ond=clip_lz(gdf_medb_count_ond,['Pastoral','Agropastoral'])
g=plot_mask(gdf_stats_mask_ond,
          label_missing='non-pastoralist area',
          title=f"Percentage of dekads Oct-Dec 2021 NDVI \n was <={thresh}% of median NDVI")
# g.figure.savefig(country_data_exploration_dir / "plots" / "eth_ndvi_adm3_ond2021.png", 
#                  facecolor="white", 
#                  bbox_inches="tight",dpi=200)
```

```python
gdf_stats_adm3_jjas=gdf_stats_adm3[gdf_stats_adm3.date.isin(
    [_dekad_to_date(dek[0],dek[1]) for dek in dekad_list_jjas])]
gdf_medb_count_jjas=compute_dekads_below_thresh(gdf_stats_adm3_jjas,pcode3_col,"median_ADM3_PCODE")
#this takes long due to the large number of cropping areas
gdf_stats_mask_jjas=clip_lz(gdf_medb_count_jjas,['Cropping','Agropastoral'])
g=plot_mask(gdf_stats_mask_jjas,
          label_missing='non-cropping area',
          title=f"Percentage of dekads June-Sep 2021 NDVI \n was <={thresh}% of median NDVI")
# g.figure.savefig(country_data_exploration_dir / "plots" / "eth_ndvi_adm3_jjas2021.png", facecolor="white", bbox_inches="tight")
```

```python
gdf_stats_mask_jjas_sel=gdf_stats_mask_jjas.drop(['geometry','date'],axis=1).rename(
    columns={'median_ADM3_PCODE':'num_dekad_below80',
             'percent':'perc_dekad_below80','perc_binned':'perc_dekad_below80_bin',
            'include':'cropping_lz'})
# gdf_stats_mask_jjas_sel.to_csv(ndvi_exploration_dir / "eth_ndvi_adm3_jjas2021_perc80.csv",index=False)
```

```python
#unclear what we should mask, so not applying any atm
gdf_stats_adm3_fmam=gdf_stats_adm3[gdf_stats_adm3.date.isin(
    [_dekad_to_date(dek[0],dek[1]) for dek in dekad_list_fmam])]
gdf_medb_count_fmam=compute_dekads_below_thresh(gdf_stats_adm3_fmam,pcode3_col,"median_ADM3_PCODE")
#this takes long due to the large number of cropping areas
g=plot_ndvi(gdf_medb_count_fmam,title=f"Percentage of dekads Feb-May 2021 NDVI \n was <={thresh}% of median NDVI");
# g.figure.savefig(country_data_exploration_dir / "plots" / "eth_ndvi_adm3_fmam2021.png", facecolor="white", bbox_inches="tight")
```

```python
gdf_stats_adm3_fmam_sel=gdf_stats_adm3_fmam.drop(['geometry','date'],axis=1).rename(
    columns={'median_ADM3_PCODE':'num_dekad_below80',
             'percent':'perc_dekad_below80','perc_binned':'perc_dekad_below80_bin',
            'include':'cropping_lz'})
# gdf_stats_adm3_fmam_sel.to_csv(ndvi_exploration_dir / "eth_ndvi_adm3_fmam2021_perc80.csv",index=False)
```
