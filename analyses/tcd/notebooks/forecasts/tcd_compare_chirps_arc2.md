## Compare CHIRPS and ARC2 data in Chad
This notebook compares the values of observed precipitation as reported by CHIRPS and ARC2. The goal of this comparison is to know if there are large differences between the sources. If the differences are large it might be beneficial to understand which of the two better represents the actual situation and thus which we should stick to. 

```python
%load_ext autoreload
%autoreload 2
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
```

```python
#months and region of interest for the trigger
months_sel=[6,7,8,9]
adm_sel=['Barh-El-Gazel','Batha','Kanem','Lac','Ouaddaï','Sila','Wadi Fira']
```

```python
#years that overlap in the two datasets
#should optimally be set dynamically
min_year=2000
max_year=2021
```

```python
gdf_reg=gdf_adm1[gdf_adm1.admin1Name.isin(adm_sel)]
```

### Load CHIRPS data

```python
#load the data
#when using rasterio, it doesn't read the 'time' coord correctly
ds_chirps_monthly=xr.load_dataset(chirps_country_processed_path)
ds_chirps_monthly=ds_chirps_monthly.where(ds_chirps_monthly.time.dt.year.isin(range(min_year,max_year+1)),drop=True)
da_chirps_monthly=ds_chirps_monthly.precip.rio.set_crs("EPSG:4326",inplace=True)
```

```python
#select the region and months of interest
da_chirps_monthly_reg=da_chirps_monthly.rio.clip(gdf_reg["geometry"])
da_chirps_monthly_sel=da_chirps_monthly_reg.where(da_chirps_monthly_reg.time.dt.month.isin(months_sel),drop=True)
```

```python
#group to yearly data for comparison
da_chirps_yearly_sel=da_chirps_monthly_sel.groupby(da_chirps_monthly_sel.time.dt.year).sum().rio.clip(gdf_reg["geometry"])
```

### Load ARC2 data

We first download the ARC2 data. ARC2 is daily data. We also group it to monthly and yearly data to compare with CHIRPS. 

```python
# #get bounds to define range to download
# gdf_adm1.total_bounds
#define class
arc2 = DrySpells(
    country_iso3 = "tcd",
    monitoring_start = "1983-01-01",#"2000-01-01",
    monitoring_end = "2021-11-25",
    range_x = ("13E", "25E"),
    range_y = ("7N", "24N")
)

# #download data, only needed if not downloaded yet
# arc2.download_data(master=True)
```

```python
da_arc = arc2.load_raw_data()
```

```python
#rename because .T is taking the transpose so naming it time makes sure there is no confusion
da_arc=da_arc.rename({"T":"time"})
#units attrs is very long list of "mm/day" so set to just "mm/day", mainly for plotting
da_arc.attrs["units"]="mm/day"
```

```python
da_arc_country=da_arc.rio.clip(gdf_adm1["geometry"])
da_arc_country=da_arc_country.where(da_arc_country.time.dt.year.isin(range(min_year,max_year+1)),drop=True)
```

```python
#for some reason the resample sets the nan values to zero
#so clipping again to the country but there should be a better solution for it
da_arc_monthly=da_arc_country.resample(time='MS',skipna=True).sum().rio.clip(gdf_adm1["geometry"])
```

```python
da_arc_monthly_reg=da_arc_monthly.rio.clip(gdf_reg["geometry"])
da_arc_monthly_sel=da_arc_monthly_reg.where(da_arc_monthly_reg.time.dt.month.isin(months_sel),drop=True)
```

```python
#group to yearly data for comparison
#skipna is annoyingly not working so clip to region again, but shouldn't work like this.. 
da_arc_yearly_sel=da_arc_monthly_sel.groupby(da_arc_monthly_sel.time.dt.year).sum(skipna=True).rio.clip(gdf_reg["geometry"])
```

### Compare CHIRPS and ARC2


We start by just plotting the total monthly precipitation for each month in 2020 for CHIRPS and ARC2 separately. 

```python
# g.fig.suptitle(f"Median per month between {da_country.time.dt.year.values.min()} and {da_country.time.dt.year.values.max()}",size=16)
```

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
df_arc_yearly=da_arc_yearly_sel.mean(dim=["x","y"]).to_dataframe().drop("spatial_ref",axis=1).rename(columns={"est_prcp":"arc"})
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


One difference between the two datasources is the resolution (CHIRPS is 0.05 and ARC2 0.1). So we check quickly if that seems to make a large difference in the included area. From the plot below we can see that the included area is large compared to the resolution and thus there is not much difference in the included area. 

```python
g=da_chirps_yearly_sel.sel(year=2018).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10))
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
```

```python
g=da_arc_yearly_sel.sel(year=2018).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10))
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
```
