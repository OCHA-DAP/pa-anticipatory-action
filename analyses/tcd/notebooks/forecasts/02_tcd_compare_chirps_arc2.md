## Compare CHIRPS and ARC2 data in Chad
This notebook compares the values of observed precipitation as reported by CHIRPS and ARC2. The goal of this comparison is to know if there are large differences between the sources. If the differences are large it might be beneficial to understand which of the two better represents the actual situation and thus which we should stick to. 

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import os
from pathlib import Path
import sys
from datetime import date
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import rioxarray
import xarray as xr
from scipy.stats import zscore

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells
from src.utils_general.raster_manipulation import compute_raster_statistics
from src.indicators.drought.config import Config

config=Config()

data_processed_dir=Path(config.DATA_DIR) /config.PUBLIC_DIR/config.PROCESSED_DIR
```

```python
hdx_blue="#007ce0"
```

```python
def plt_raster_facet(da,facet_col,vmax=None,title=None):
    g=da.plot.imshow(
    col=facet_col,
    col_wrap=6,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation (mm)"
    },
    levels=20,
    vmax=vmax,
    cmap='Blues',
)

    for ax in g.axes.flat:
        gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
        ax.axis("off")
    if title is not None:
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.9,bottom=0.3,hspace=0.3)
```

```python
iso3="tcd"
```

```python
chirps_country_processed_dir = Path(data_processed_dir)/iso3/"chirps"
chirps_country_processed_path = Path(chirps_country_processed_dir)/"monthly"/f"{iso3}_chirps_monthly.nc"
parameters=config.parameters(iso3)
adm1_path = Path(os.getenv("AA_DATA_DIR"))/"public"/"raw"/iso3/"cod_ab"/parameters["path_admin1_shp"]
gdf_adm1=gpd.read_file(adm1_path)
adm2_path=data_processed_dir / iso3 / config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"
gdf_adm2=gpd.read_file(adm2_path)
gdf_aoi = gdf_adm2[gdf_adm2.area_of_interest == True]
```

```python
#months and region of interest for the trigger
months_sel=[6,7,8,9]
```

```python
#years that overlap in the two datasets
#should optimally be set dynamically
min_year=2000
max_year=2021
```

### Load CHIRPS data

```python
#load the data
#when using rioxarray some of the processing takes very long. Possibly to do with the type of `time` but leaving for now
ds_chirps_monthly=xr.load_dataset(chirps_country_processed_path)
ds_chirps_monthly=ds_chirps_monthly.where(ds_chirps_monthly.time.dt.year.isin(range(min_year,max_year+1)),drop=True)
da_chirps_monthly=ds_chirps_monthly.precip.rio.set_crs("EPSG:4326",inplace=True)
```

```python
#select the region and months of interest
da_chirps_monthly_reg=da_chirps_monthly.rio.clip(gdf_aoi["geometry"])
da_chirps_monthly_sel=da_chirps_monthly_reg.where(da_chirps_monthly_reg.time.dt.month.isin(months_sel),drop=True)
```

```python
#group to yearly data for comparison
da_chirps_yearly_sel=da_chirps_monthly_sel.groupby(da_chirps_monthly_sel.time.dt.year).sum().rio.clip(gdf_aoi["geometry"])
```

### Load ARC2 data

We first download the ARC2 data. ARC2 is daily data. We also group it to monthly and yearly data to compare with CHIRPS. 

```python
# #get bounds to define range to download
# gdf_adm1.total_bounds
#define class
arc2 = DrySpells(
    country_iso3 = iso3,
    monitoring_start = "2000-01-01",
    monitoring_end = "2021-11-25",
    range_x = ("13E", "25E"),
    range_y = ("7N", "24N"),
    polygon_path = adm2_path,
    bound_col = "admin2Pcod",
)

# #download data, only needed if not downloaded yet
# arc2.download_data(main=True)
```

```python
da_arc = arc2.load_raw_data(convert_date=False)
```

```python
#rename because .T is taking the transpose so naming it time makes sure there is no confusion
da_arc=da_arc.rename({"T":"time"})
#units attrs is very long list of "mm/day" so set to just "mm/day", mainly for plotting
da_arc.attrs["units"]="mm/day"
```

```python
da_arc_country=da_arc.rio.clip(gdf_adm2["geometry"])
da_arc_country=da_arc_country.where(da_arc_country.time.dt.year.isin(range(min_year,max_year+1)),drop=True)
```

```python
#for some reason the resample sets the nan values to zero
#so clipping again to the country but there should be a better solution for it
da_arc_monthly=da_arc_country.resample(time='MS',skipna=True).sum().rio.clip(gdf_adm2["geometry"])
```

```python
da_arc_monthly_reg=da_arc_monthly.rio.clip(gdf_aoi["geometry"])
da_arc_monthly_sel=da_arc_monthly_reg.where(da_arc_monthly_reg.time.dt.month.isin(months_sel),drop=True)
```

```python
#group to yearly data for comparison
#skipna is annoyingly not working so clip to region again, but shouldn't work like this.. 
da_arc_yearly_sel=da_arc_monthly_sel.groupby(da_arc_monthly_sel.time.dt.year).sum(skipna=True).rio.clip(gdf_aoi["geometry"])
```

### Compare CHIRPS and ARC2


We start by just plotting the total monthly precipitation for each month in 2020 for CHIRPS and ARC2 separately. 

```python
#chirps monthly precip 2020
plt_raster_facet(da_chirps_monthly.sel(time=da_chirps_monthly.time.dt.year.isin([2020])),"time",vmax=380,title="CHIRPS: Monthly precipitation in 2020")
```

```python
#arc2 monthly precip 2020
plt_raster_facet(da_arc_monthly.sel(time=da_arc_monthly.time.dt.year.isin([2020])),"time",vmax=380,
                title="ARC2: Monthly precipitation in 2020")
```

From these plots we can already see large differences:
1) in absolute values. ARC2 is clearly showing higher values
2) in patterns. For example in August we can see that ARC2 registers relatively higher values towards the north than CHIRPS. 

These differences are surprisingly large and thus would require further investigation on where they originate from. I have no answer to this as of now.. 


We also simply compare the mean of all values within the selected region and selected months. Again we can see a large difference, where the values of ARC2 are on average a lot higher

```python
da_chirps_monthly_sel.mean()
```

```python
da_arc_monthly_sel.mean()
```

To understand the direction of difference, i.e. whether CHIRPS or ARC2 reports higher values we use the yearly sum of data and use the mean across all cells within the region. We then plot the difference between the yearly numbers of the two data sources

```python
df_chirps_yearly=da_chirps_yearly_sel.mean(dim=["longitude","latitude"]).to_dataframe().drop(
    "spatial_ref",axis=1).rename(columns={"precip":"chirps"})
df_arc_yearly=da_arc_yearly_sel.mean(dim=["X","Y"]).to_dataframe().drop("spatial_ref",axis=1).rename(columns={"est_prcp":"arc"})
```

```python
df_comb=pd.concat([df_chirps_yearly,df_arc_yearly],axis=1)
df_comb["diff"]=df_comb.chirps-df_comb.arc
```

```python
df_comb.sort_values("diff",inplace=True,ascending=False)
#remove years that are not covered by both data sources
df_comb=df_comb.dropna()
df_comb=df_comb.reset_index()
```

```python
# Plotting the horizontal lines
fig,ax=plt.subplots(figsize=(12,8))
plt.hlines(y=df_comb.index
        , xmin=0, xmax=df_comb["diff"],
           linewidth=5)

# Decorations
# Setting the labels of x-axis and y-axis
plt.gca().set(ylabel='year', xlabel=f'Difference (mm),CHIRPS minus ARC2')

# Setting Date to y-axis
plt.yticks(df_comb.index, df_comb.year, fontsize=12)
ax.xaxis.label.set_size(16)

plt.title(f'Yearly difference CHIRPS minus ARC2 (mm)', fontdict={
          'size': 20});
```

From the divergent bar plot above we can see that ARC2 is always giving higher yearly values than CHIRPS. The most extreme differences were in 2020 and 2021. And generally there seems to be a pattern that the differences have been more extreme in recent years. 


I don't know why there are these large differences. I was first planning to use CHIRPS for some part of the analyses and ARC2 for other parts of the analyses, but that might not be a smart idea looking at these numbers.. I have no idea yet how to figure out where these differences come from and what to do with them.   


### Relative differences
So far we looked at absolute differences. It is interesting to compare relative differences and maybe even more important for our purposes. I.e is the year that arc2 observed the most rainfall the same as that CHIRPS received the most rainfall. One very simple method is to look at the rank, i.e. a rank of one means that that was the year with the most rainfall. We do this below and then plot the difference in rank for the two data sets for each year. 

For now we only compare the rank of the total rainfall over the whole area of interest. However this could be extended both temporally and spatially by e.g. looking at monthly patterns and by looking at the raster cell level. 

```python
df_comb["arc_rank"]=df_comb.arc.rank(ascending=False)
df_comb["chirps_rank"]=df_comb.chirps.rank(ascending=False)
df_comb["diff_rank"]=df_comb.chirps_rank-df_comb.arc_rank
```

```python
# Plotting the horizontal lines
df_comb=df_comb.sort_values("diff_rank").reset_index(drop=True)
fig,ax=plt.subplots(figsize=(12,8))
plt.hlines(y=df_comb.index
        , xmin=0, xmax=df_comb["diff_rank"],
           linewidth=5)

# Decorations
# Setting the labels of x-axis and y-axis
plt.gca().set(ylabel='year', xlabel=f'Rank difference,CHIRPS minus ARC2')

# Setting Date to y-axis
plt.yticks(df_comb.index, df_comb.year, fontsize=12)
ax.xaxis.label.set_size(16)

plt.title(f'Rank difference CHIRPS minus ARC2', fontdict={
          'size': 20});
```

From the graph above we can see that for quite some years the rank was the same or close. However for other years there was a large difference in rank. For example 2021 was according to CHIRPS the 8th most rainfall while according to ARC2 it was the 2th on the rank. This is slightly worrisome and I don't know what the cause of the difference is.

Interestingly we do see that the years that had a large absolute difference (e.g. 2020) don't necessarily have a large difference in rank (in the case of 2020 a difference of 2). The opposite is also true, e.g. in 2017 we didn't see that much of an absolute difference while in terms of rank there is a big difference. 

```python
df_comb[["year","diff_rank","chirps_rank","arc_rank","arc","chirps"]]
```

Another commonly used relative measure is the zscore. Again we compare only the yearly data now but this could be extended to monthly or raster data. 

The z-score is defined as $Z=\frac{X-\mu}{\sigma}$ where X is the datapoint, $\mu$ the mean and $\sigma$ the standard deviation

```python
df_comb["arc_zscore"]=zscore(df_comb.arc)
df_comb["chirps_zscore"]=zscore(df_comb.chirps)
df_comb["diff_zscore"]=df_comb.chirps_zscore-df_comb.arc_zscore
```

```python
# Plotting the horizontal lines
df_comb=df_comb.sort_values("diff_zscore").reset_index(drop=True)
fig,ax=plt.subplots(figsize=(12,8))
plt.hlines(y=df_comb.index
        , xmin=0, xmax=df_comb["diff_zscore"],
           linewidth=5)

# Decorations
# Setting the labels of x-axis and y-axis
plt.gca().set(ylabel='year', xlabel=f'Relative difference (z-score),CHIRPS minus ARC2')

# Setting Date to y-axis
plt.yticks(df_comb.index, df_comb.year, fontsize=12)
ax.xaxis.label.set_size(16)

plt.title(f'Relative difference CHIRPS minus ARC2 (z-score)', fontdict={
          'size': 20});
```

```python
# Plotting the horizontal lines
df_comb=df_comb.sort_values("chirps_zscore").reset_index(drop=True)
fig,axes=plt.subplots(1,2,figsize=(20,8))
for ax,ds in zip(axes,["chirps","arc"]):
    df_comb=df_comb.sort_values(f"{ds}_zscore").reset_index(drop=True)
    ax.hlines(y=df_comb.index
            , xmin=0, xmax=df_comb[f"{ds}_zscore"],
               linewidth=5)#,ax=ax)

    # Decorations
    # Setting the labels of x-axis and y-axis
#     plt.gca().set(ylabel='year', xlabel=f'Difference (mm),CHIRPS minus ARC2',ax=ax)
    ax.set_xlabel("Z-score")
    ax.set_ylabel("year")
    # Setting Date to y-axis
    ax.set_yticks(df_comb.index)
    ax.set_yticklabels(df_comb.year,fontsize=12)
    ax.xaxis.label.set_size(16)

    ax.set_title(f'Z-score {ds}', fontdict={
              'size': 20});
```

From the plots above we can also see that the z-score between the two sources differs significantly. Both in absolute terms (the max z-score of chirps is around 2 while that of arc2 is around 3) and in relative differences. it is still unclear where these differences originate from. 


In a desparate atttempt to understand the differences between the two datasources we quickly check the difference of included cells due to the resolution. The resolution of CHIRPS is 0.05 and ARC2 0.1. From the plot below we can see that the included area is large compared to the resolution and thus there is not much difference in the included area. 

```python
g=da_chirps_yearly_sel.sel(year=2018).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
```

```python
g=da_arc_yearly_sel.sel(year=2018).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
```
