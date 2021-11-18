---
jupytext:
  cell_metadata_filter: -all
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

# IRI forecast as a trigger for drought in Cjad
This notebook explores the option of using IRI's seasonal forecast as part of drought-related trigger in Burkina Faso. 
From the country team the proposed trigger is:
- Trigger #1 in March covering June-July-August or July-August-September. Threshold desired: 60%.
- Trigger #2 in May covering June-July-August or July-August-September. Threshold desired: 60%. 
- Targeted Admin1s: Barh el Gazel, Batha, Kanem, Lac (une partie), Ouaddaï (une partie), Sila (une partie), Wadi Fira
- 20% of those admin1s meeting the threshold

This notebook explores if and when these triggers would be reached.

```{code-cell} ipython3
:tags: [remove_cell]

%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
:tags: [remove_cell]

import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from rasterstats import zonal_stats
from IPython.display import Markdown as md
from myst_nb import glue
import cftime
import re
import calendar
from dateutil.relativedelta import relativedelta
import matplotlib
from rasterio.enums import Resampling

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import get_iri_data,get_iri_data_dominant
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```{code-cell} ipython3
hdx_blue="#007ce0"
```

```{code-cell} ipython3
:tags: [remove_cell]

#month number refers to the last month of the season
month_season_mapping={1:"NDJ",2:"DJF",3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND"}
```

## Inspect forecasts

```{code-cell} ipython3
:tags: [remove_cell]

#TODO: some admins only part should be included, check with team
adm_sel=['Barh-El-Gazel','Batha','Kanem','Lac','Ouaddaï','Sila','Wadi Fira']
adm_sel_str=re.sub(r"[ -]", "", "".join(adm_sel)).lower()
```

```{code-cell} ipython3
:tags: [remove_cell]

leadtime_mar=3
leadtime_may=1
```

TODO: change paths to pathlib

```{code-cell} ipython3
:tags: [remove_cell]

country="tcd"
config=Config()
parameters = config.parameters(country)
country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country)
glb_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration","glb")
iri_exploration_dir=os.path.join(country_data_exploration_dir,"iri")
stats_reg_path=os.path.join(iri_exploration_dir,f"{country}_iri_seasonal_forecast_stats_{''.join(adm_sel_str)}.csv")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

```{code-cell} ipython3
:tags: [remove_cell]

gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_reg=gdf_adm1[gdf_adm1.admin1Name.isin(adm_sel)]
```

We load the iri data indicating the dominant tercile. The negative values indicate forecasted below average rainfall, and the positive values above average.

We plot the forecast raster data for the periods and leadtimes of interest. The red areas are the admin1's we are focussing on. 
These figures are is similair to [the figure on the IRI Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html), except that the bins are defined slightly differently

```{code-cell} ipython3
#F indicates the publication month, and L the leadtime. 
#A leadtime of 1 means a forecast published in May is forecasting JJA
ds_iri_dom=get_iri_data_dominant(config,download=False)
ds_iri_dom=ds_iri_dom.rio.write_crs("EPSG:4326",inplace=True)
da_iri_dom=ds_iri_dom.dominant
da_iri_dom_clip=da_iri_dom.rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```{code-cell} ipython3
#not very neat function but does the job for now
#iri website bins
# plt_levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
def plt_raster_iri(da_iri_dom_clip,
                   pub_mon,
                   lt,
                   plt_levels,
                   plt_colors,
                  ):
    for_seas=month_season_mapping[(pub_mon+lt+1)%12+1]
    g=da_iri_dom_clip.where(da_iri_dom_clip.F.dt.month.isin([pub_mon]), drop=True).sel(L=lt).plot(
    col="F",
    col_wrap=5,
    levels=plt_levels,
    colors=plt_colors,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        'ticks': plt_levels,
    },
    figsize=(25,7)
    )
    for ax in g.axes.flat:
        gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
        gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    g.fig.suptitle(f"Forecasts published in {calendar.month_abbr[pub_mon]} predicting {for_seas} (lt={lt}) \n The subtitles indicate the publishing date",y=1.1);
```

```{code-cell} ipython3
#iri website bins
# plt_levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
plt_levels=[-100,-70,-60,-50,-45,-40,40,45,50,60,70,100]
plt_colors=['#783200','#ab461e','#d18132','#e8b832','#fafa02','#ffffff','#d1f8cc','#acf8a0','#73bb6e','#3a82b3','#0e3bf4']
```

```{code-cell} ipython3
plt_raster_iri(da_iri_dom_clip,pub_mon=3,lt=3,plt_levels=plt_levels,plt_colors=plt_colors)
```

```{code-cell} ipython3
plt_raster_iri(da_iri_dom_clip,pub_mon=3,lt=4,plt_levels=plt_levels,plt_colors=plt_colors)
```

```{code-cell} ipython3
plt_raster_iri(da_iri_dom_clip,pub_mon=5,lt=1,plt_levels=plt_levels,plt_colors=plt_colors)
```

```{code-cell} ipython3
plt_raster_iri(da_iri_dom_clip,pub_mon=5,lt=2,plt_levels=plt_levels,plt_colors=plt_colors)
```

Below we plot a few examples of "tricky" forecasts. For the left two: say the threshold would be at 40%, would the trigger be reached for the red region? For the right two plots we see a combination of below and above average across the region. What should we do with that? 

These figures are to guide the discussion on which forecasts we would have wanted to trigger and for which we wouldn't

```{code-cell} ipython3
:tags: [hide_input]

g=da_iri_dom_clip.where(da_iri_dom_clip.F.isin([cftime.Datetime360Day(2021, 2, 16, 0, 0, 0, 0),cftime.Datetime360Day(2021, 3, 16, 0, 0, 0, 0),cftime.Datetime360Day(2018, 7, 16, 0, 0, 0, 0),cftime.Datetime360Day(2019, 7, 16, 0, 0, 0, 0)]),drop=True).sel(L=3).plot(
    col="F",
    col_wrap=4,
    levels=levels,
    colors=colors,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        'ticks': levels,
    },
    figsize=(20,10)
)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

### Which cells to include for aggregation?
We inspect 3 different methods: including all cells with their centre in the region, all cells touching the region, and an approximate weighted average. 

We should discuss as team (and with meteorologists) which is the best method. 
At least with all methods we include a substantial number of cells.

```{code-cell} ipython3
:tags: []

#sel random values to enable plotting of included cells (so values are irrelevant)
da_iri_dom_blue=da_iri_dom.sel(F="2020-05-16",L=1).squeeze()
```

```{code-cell} ipython3
da_iri_dom_blue_centre=da_iri_dom_blue.rio.clip(gdf_reg["geometry"], all_touched=False)
g=da_iri_dom_blue_centre.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with cell centres: {da_iri_dom_blue_centre.count().values} cells included")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```{code-cell} ipython3
da_iri_dom_blue_touched=da_iri_dom_blue.rio.clip(gdf_reg["geometry"], all_touched=True)
g=da_iri_dom_blue_touched.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with all cells touching: {da_iri_dom_blue_touched.count().values} cells included")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```{code-cell} ipython3
#approximate of a weighted average
da_iri_dom_blue_res = da_iri_dom_blue.rio.reproject(
    da_iri_dom_blue.rio.crs,
    #resolution it will be changed to, original is 1
    resolution=0.05,
    #use nearest so cell values stay the same, only cut
    #into smaller pieces
    resampling=Resampling.nearest,
    nodata=np.nan,
).rio.clip(gdf_reg["geometry"], all_touched=False)
```

```{code-cell} ipython3
g=da_iri_dom_blue_res.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with approx weighted average")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

#### Compute stats
We can now compute the statistics of the region of interest. For now I am working with all cells touching the region, but this is something that still has to be thought about more. 

While before we looked at the dominant tercile, we now load the data containing the probability for each individual tercile. We focus on the below-average tercile.

```{code-cell} ipython3
:tags: [remove_cell]

#C indicates the tercile (below-average, normal, or above-average).  
#F indicates the publication month, and L the leadtime
ds_iri = get_iri_data(config, download=False)
ds_iri=ds_iri.rio.write_crs("EPSG:4326",inplace=True)
da_iri=ds_iri.prob
da_iri_allt=da_iri.rio.clip(gdf_reg.geometry.apply(mapping), da_iri.rio.crs, all_touched=True)
```

Here we experiment with the threshold. 40 is randomly chosen, with the reasoning that this might be the lowest threshold that could stillb e reasonable. The `perc_area` of 20 was proposed by FAO

```{code-cell} ipython3
#% probability of bavg
threshold=40 #60
#min percentage of the area that needs to reach the threshold
perc_area=20
```

```{code-cell} ipython3
da_iri_allt_bavg.sel(F=)
```

```{code-cell} ipython3
#compute stats
#dissolve the region to one polygon
gdf_reg_dissolved=gdf_reg.dissolve(by="admin0Name")
gdf_reg_dissolved=gdf_reg_dissolved[["admin0Pcod","geometry"]]

da_iri_allt_bavg=da_iri_allt.sel(C=0)
df_stats_reg=compute_raster_statistics(
        gdf=gdf_reg_dissolved,
        bound_col="admin0Pcod",
        raster_array=da_iri_allt_bavg,
        lon_coord="longitude",
        lat_coord="latitude",
        stats_list=["min","mean","max","std","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
    )
da_iri_allt_thresh=da_iri_allt.where(da_iri_allt>=threshold)
df_stats_reg_thresh=compute_raster_statistics(gdf=gdf_reg_dissolved,bound_col="admin0Pcod",raster_array=da_iri_allt_thresh,lon_coord="longitude",lat_coord="latitude",stats_list=["count"])

df_stats_reg["perc_thresh"] = df_stats_reg_thresh[f"count_admin0Pcod"]/df_stats_reg[f"count_admin0Pcod"]*100
df_stats_reg["F"]=pd.to_datetime(df_stats_reg["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_reg["month"]=df_stats_reg.F.dt.month
# df_stats_reg.time=pd.to_datetime(df_stats_reg.time.apply(lambda x: x.strftime("%Y-%m-%d")))
# df_stats_reg["end_time"]=pd.to_datetime(df_stats_reg["time"].apply(lambda x: x.strftime('%Y-%m-%d')))
# df_stats_reg["end_month"]=df_stats_reg.end_time.dt.to_period("M")
# df_stats_reg["start_time"]=df_stats_reg.end_time.apply(lambda x: x+relativedelta(months=-2))
# df_stats_reg["start_month"]=df_stats_reg.start_time.dt.to_period("M")
# df_stats_reg["season"]=df_stats_reg.end_month.apply(lambda x:month_season_mapping[x.month])
# df_stats_reg["seas_year"]=df_stats_reg.apply(lambda x: f"{x.season} {x.end_month.year}",axis=1)
# df_stats_reg["rainy_seas"]=np.where(df_stats_reg.start_month.dt.month.isin(end_months_sel),1,0)
# df_stats_reg=df_stats_reg.sort_values("start_month")
# df_stats_reg["rainy_seas_str"]=df_stats_reg["rainy_seas"].replace({0:"outside rainy season",1:"rainy season"})
# df_stats_reg["year"]=df_stats_reg.end_month.dt.year
```

NaN values indicate that the whole region is covered by a dry mask at that point. See [here](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/) for more information

```{code-cell} ipython3
df_stats_reg
```

```{code-cell} ipython3
df_stats_reg_bavg=df_stats_reg[(~df_stats_reg.perc_thresh.isnull())]
```

Something goes wrong! 

```{code-cell} ipython3
df_stats_reg_bavg[df_stats_reg_bavg["perc_thresh"]>=perc_area]
```

The total percentage of forecasts that predicted >=20% of the area >=40% of below average

```{code-cell} ipython3
len(df_stats_reg_bavg[df_stats_reg_bavg["perc_thresh"]>=perc_area])/len(df_stats_reg_bavg)*100
```

And compute the statistics over this region, see a subset below

+++

## Analyze statistics probability below average

+++

Below the distribution of probability values is shown per month. \
This only includes the values for the below-average tercile, with a leadtime of {glue:text}`leadtime`. \
It should be noted that since we only have data from Mar 2017, these distributions contain maximum 5 values. \
From the distribution, it can be seen that a probability of 50% has never been reached since Mar 2017.

```{code-cell} ipython3
#first entry refers to the publication month, second to the leadtime
trig_mom=[(3,3),(3,4),(5,1),(5,2)]
```

```{code-cell} ipython3
df_stats_reg_bavg[df_stats_reg_bavg[['month', 'L']].apply(tuple, axis=1).isin(trig_mom)]
```

```{code-cell} ipython3
(df_stats_reg_bavg.F.dt.month.isin(trig_mom[0]["F"]))&(df_stats_reg_bavg.L.isin(trig_mom[0]["L"]))
```

```{code-cell} ipython3
df_stats_trig=df_stats_reg_bavg[(df_stats_reg_bavg.F.dt.month.isin(trig_mom[0]["F"]))&(df_stats_reg_bavg.L.isin(trig_mom[0]["L"]))]
```

```{code-cell} ipython3
:tags: [remove_cell]

def comb_list_string(str_list):
    if len(str_list)>0:
        return " in "+", ".join(str_list)
    else:
        return ""

max_prob_mar=stats_mar.max_admin0Pcod.max()
num_trig_mar=len(stats_mar.loc[stats_mar['max_admin0Pcod']>=threshold_mar])
year_trig_mar=comb_list_string([str(y) for y in stats_mar.loc[stats_mar['max_admin0Pcod']>=threshold_mar].F.dt.year.unique()])

num_trig_jul=len(stats_jul.loc[stats_jul['max_admin0Pcod']>=threshold_jul])
year_trig_jul=comb_list_string([str(y) for y in stats_jul.loc[stats_jul['max_admin0Pcod']>=threshold_jul].F.dt.year.unique()])
max_prob_jul=stats_jul.max_admin0Pcod.max()
```

```{code-cell} ipython3
:tags: [remove_cell]

glue("max_prob_mar", max_prob_mar)
glue("max_prob_jul", max_prob_jul)
glue("num_trig_mar", num_trig_mar)
glue("num_trig_jul", num_trig_jul)
glue("year_trig_mar", year_trig_mar)
glue("year_trig_jul", year_trig_jul)
glue("threshold_mar", threshold_mar)
glue("threshold_jul", threshold_jul)
```

```{code-cell} ipython3
df_stats_reg_bavg
```

```{code-cell} ipython3
import altair as alt
```

```{code-cell} ipython3
histo=alt.Chart(df_stats_reg_bavg).mark_bar().encode(
    alt.X("perc_thresh:Q", bin=alt.Bin(step=5)),
    y='count()',
)
line = alt.Chart(pd.DataFrame({'x': [perc_area]})).mark_rule(color="red").encode(x='x')
histo+line
```

```{code-cell} ipython3
:tags: [hide_input]

#plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=df_stats_reg_bavg,x="month",y="max_admin0Pcod",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 3 months")
ax.set_xlabel("Publication month");
```

```{code-cell} ipython3
:tags: [hide_input]

#plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=df_stats_reg_bavg[df_stats_reg_bavg.L==3],x="month",y="max_admin0Pcod",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 3 months")
ax.set_xlabel("Publication month");
```

```{code-cell} ipython3
:tags: [hide_input]

#plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=df_stats_reg_bavg[df_stats_reg_bavg.L==1],x="month",y="max_admin0Pcod",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 1 month")
ax.set_xlabel("Publication month");
```

+++ {"tags": []}

More specifically we are interested in March and July, with a leadtime of 3 and 1 month respectively. 
The maximum values across all cells for the March forecasts has been {glue:text}`max_prob_mar:.2f`%, and for the July forecasts {glue:text}`max_prob_jul:.2f`% 
This would mean that if we would take the max cell as aggregation method, the threshold of {glue:text}`threshold_mar` for March would have been reached {glue:text}`num_trig_mar` times {glue:text}`year_trig_mar`. 
For July the threshold of {glue:text}`threshold_jul` would have been reached {glue:text}`num_trig_jul` times{glue:text}`year_trig_jul`."

```{code-cell} ipython3
:tags: [remove_cell]

stats_country=compute_zonal_stats_xarray(da_iri_allt,gdf_adm1)
stats_country["F"]=pd.to_datetime(stats_country["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
stats_country["month"]=stats_country.F.dt.month
glue("max_prob_mar_country",stats_country.loc[(stats_country.C==0)&(stats_country.L==leadtime_mar)&(stats_country.F.dt.month==3),'max_admin0Pcod'].max())
glue("max_prob_jul_country",stats_country.loc[(stats_country.C==0)&(stats_country.L==leadtime_jul)&(stats_country.F.dt.month==7),'max_admin0Pcod'].max())
```

To check if these below 50% and below 40% probabilities depend on the part of the country, we also compute the maximum values in the whole country across all years. 
<!-- While the values can be slightly higher in other regions, the 50% threshold is never reached.  -->

+++

The maximum value for the March forecast in the whole country was {glue:text}`max_prob_mar_country:.2f`%. \
<!-- For July this was {glue:text}`max_prob_jul_country:.2f`%" -->

```{code-cell} ipython3
:tags: [remove_cell]

perc_for_40th=stats_country.loc[(stats_country.C==0)&(stats_country.L==leadtime_jul),'max_admin0Pcod'].ge(40).value_counts(True)[True]*100
glue("perc_for_maxcell_40th",perc_for_40th)
```

Across all months, {glue:text}`perc_for_maxcell_40th:.2f`% of the forecasts with 1 month leadtime had a >=40% probability of below average rainfall in at least one cell across the **whole** country

```{code-cell} ipython3
:tags: [remove_cell]

#plot distribution for forecasts with C=0 (=below average), for all months
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=stats_country[(stats_country.C==0)&(stats_country.L==1)],x="month",y="max_admin0Pcod",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 3 months")
ax.set_xlabel("Publication month")
```

### Methods of aggregation
Note: all these computations only cover the region of interest

```{code-cell} ipython3
:tags: [remove_cell]

max_prob_mar=stats_mar['max_admin0Pcod'].max()
num_trig_mar_mean=len(stats_mar.loc[stats_mar['mean_cell']>=threshold_mar])
year_trig_mar_mean=comb_list_string([str(y) for y in stats_mar.loc[stats_mar['mean_cell']>=threshold_mar].F.dt.year.unique()])
num_trig_mar_perc10=len(stats_mar.loc[stats_mar['10quant_cell']>=threshold_mar])
year_trig_mar_perc10=comb_list_string([str(y) for y in stats_mar.loc[stats_mar['10quant_cell']>=threshold_mar].F.dt.year.unique()])
max_perc40_mar=stats_mar["40percth_cell"].max()
glue("num_trig_mar_mean", num_trig_mar_mean)
glue("year_trig_mar_mean", year_trig_mar_mean)
glue("num_trig_mar_perc10", num_trig_mar_perc10)
glue("year_trig_mar_perc10", year_trig_mar_perc10)
glue("max_perc40_mar", max_perc40_mar)
```

While taking the max cell is the most extreme method of aggregation, we have many other possiblities. Such as looking at the mean, or at a percentage of cells. 
<!-- For the July forecast we wouldn't trigger with any method of aggregation, since we already didn't trigger with the max methodology.  -->

For March, when using the mean method aggregation, the trigger would have been met {glue:text}`num_trig_mar_mean` times{glue:text}`year_trig_mar_mean`.

Below the distribution of the percentage of the area with >=40% probability is shown for March. From here it can be seen that the maximum percentage is {glue:text}`max_perc40_mar:.2f`%.
We look at the distribution of the percentage of the area with >=40% probability of below avg rainfall for the admins of interest, across all forecasts with a leadtime of {glue:text}`leadtime`. 
When requiring 10% of cells to be above 40% this would be met {glue:text}`num_trig_mar_perc10` times{glue:text}`year_trig_mar_perc10`.

```{code-cell} ipython3
:tags: [hide_input]

#plot distribution for forecasts with C=0 (=below average) and L=1, for March
g=sns.displot(stats_mar["40percth_cell"],color="#007CE0",binwidth=1)
```

The plot below shows the occurences across all months and all leadtimes where at least 1% of the cells had a probability of at least 40% for below average rainfall. We can see that the occurrence of this is pretty rare.

```{code-cell} ipython3
:tags: [hide_input]

#plot distribution for forecasts with C=0 (=below average) and L=1, for all months
g=sns.displot(df_stats_reg_bavg.loc[df_stats_reg_bavg["40percth_cell"]>=1,"40percth_cell"],color="#007CE0",binwidth=3)
```

<!-- While we can include the spatial severity in the trigger threshold, we should also take into account that the spatial uncertainty of seasonal forecasts is large. 

Given the size of the area of interest, it might therefore be better to only focus on whether any cell within that region reached the probability threshold. However, in this case 40% might be too sensitive of a trigger -->

+++

### Examine dominant tercile and 40% threshold
Besides setting a threshold on the below average tercile, we also want to be sure that the below average tercile is the dominant tercile. We therefore require, at the pixel level that 
probability below average >= (probability above average + 5%)
Here we check how often this occurs, for the months of interest and across all months

+++

Moreover on the above analysis, we require at least 10% of the area meeting the threshold. This results in the following activations for our periods of interest

```{code-cell} ipython3
stats_mar[stats_mar["40th_bavg_cell"]>=10]
```

```{code-cell} ipython3
stats_jul[stats_jul["40th_bavg_cell"]>=10]
```

```{code-cell} ipython3
stats_mar
```

```{code-cell} ipython3
stats_jul
```

```{code-cell} ipython3
#across all months with leadtime_mar
df_stats_reg_bavg_ltmar=df_stats_reg_bavg.loc[(df_stats_reg_bavg.L==leadtime_mar)]
df_stats_reg_bavg_ltmar[df_stats_reg_bavg_ltmar["40th_bavg_cell"]>=10]
```

```{code-cell} ipython3
#percentage of forecasts that met requirement
len(df_stats_reg_bavg_ltmar[df_stats_reg_bavg_ltmar["40th_bavg_cell"]>=10])/len(df_stats_reg_bavg_ltmar.F.unique())*100
```

```{code-cell} ipython3
#across all months with leadtime_jul
df_stats_reg_bavg_ltjul=df_stats_reg_bavg.loc[(df_stats_reg_bavg.L==leadtime_jul)]
df_stats_reg_bavg_ltjul[df_stats_reg_bavg_ltjul["40th_bavg_cell"]>=10]
```

```{code-cell} ipython3
#percentage of forecasts that met requirement
len(df_stats_reg_bavg_ltjul[df_stats_reg_bavg_ltjul["40th_bavg_cell"]>=10])/len(df_stats_reg_bavg_ltjul.F.unique())*100
```

### Examine ONLY dominant region
Understand how often it occurrs that the 10 percentile threshold is at least x% higher for below than above average

As can be seen this occurs the same number of times as when below average probability is at least 40%, but the dates don't fully overlap

```{code-cell} ipython3
diff_threshold=5
```

```{code-cell} ipython3
df_stats_reg_aavg=df_stats_reg[df_stats_reg.C==2]
df_stats_reg_aavg_ltmar=df_stats_reg_aavg[df_stats_reg_aavg.L==leadtime_mar]
```

```{code-cell} ipython3
df_stats_reg_merged=df_stats_reg_bavg.merge(df_stats_reg_aavg,on=["F","L"],suffixes=("_bavg","_aavg"))
```

```{code-cell} ipython3
df_stats_reg_merged["diff_bel_abv"]=df_stats_reg_merged["10quant_cell_bavg"]-df_stats_reg_merged["10quant_cell_aavg"]
```

```{code-cell} ipython3
df_stats_reg_merged_ltmar=df_stats_reg_merged[df_stats_reg_merged.L==leadtime_mar]
```

```{code-cell} ipython3
df_stats_reg_merged_ltmar[df_stats_reg_merged_ltmar["diff_bel_abv"]>=diff_threshold]
```

```{code-cell} ipython3
df_stats_reg_merged_ltjul=df_stats_reg_merged[df_stats_reg_merged.L==leadtime_jul]
```

```{code-cell} ipython3
df_stats_reg_merged_ltjul[df_stats_reg_merged_ltjul["diff_bel_abv"]>=diff_threshold]
```

### Examine ONLY dominant pixel
Understand how often it occurrs that at least 10% of the pixels have x% higher probability for below than above average

As can be seen this occurrs much more often

```{code-cell} ipython3
def compute_zonal_stats_xarray_dominant(raster,shapefile,lon_coord="lon",lat_coord="lat",var_name="prob"):
    raster_clip=raster.rio.set_spatial_dims(x_dim=lon_coord,y_dim=lat_coord).rio.clip(shapefile.geometry.apply(mapping),raster.rio.crs,all_touched=False)
    raster_diff_bel_abv=raster_clip.sel(C=0)-raster_clip.sel(C=2)
    grid_quant90 = raster_diff_bel_abv.quantile(0.9,dim=[lon_coord,lat_coord]).rename({var_name: "10quant_cell"})
    zonal_stats_xr = xr.merge([grid_quant90])
    zonal_stats_df=zonal_stats_xr.to_dataframe()
    zonal_stats_df=zonal_stats_df.reset_index()
    return zonal_stats_df
```

```{code-cell} ipython3
stats_dom=compute_zonal_stats_xarray_dominant(da_iri_allt,gdf_reg)
stats_dom["F"]=pd.to_datetime(stats_dom["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
stats_dom["month"]=stats_dom.F.dt.month
```

```{code-cell} ipython3
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_mar)])
```

```{code-cell} ipython3
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_mar)])/len(stats_dom.F.unique())
```

```{code-cell} ipython3
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_jul)])
```

```{code-cell} ipython3
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_jul)])/len(stats_dom.F.unique())
```

```{code-cell} ipython3
stats_dom[(stats_dom["10quant_cell"]>=10)&(stats_dom.L==leadtime_mar)]
```

```{code-cell} ipython3
stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_mar)]
```

```{code-cell} ipython3
df_stats_reg[(df_stats_reg.F.isin(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_jul)].F.unique()))&(df_stats_reg.L==leadtime_jul)]
```

## OLD: Examine dominant tercile region

+++

**NOTE: this is outdated since this compares the 10% of the area numbers, while we now compare the terciles at pixel level

+++

Besides knowing if the below average tercile reaches a certain threshold, it is also important to understand if the below average tercile is the dominant tercile. Where dominant indicates the tercile with the highes probability. Else, it wouldn't be logical to anticipate based on the likelihood of below average rainfall. 

Since we are working with aggregation we have to determine what method we use to set the probability of below average, normal, and above average precipitation. 
For this analysis we look at the cell with the maximum probability for each tercile
<!-- For this analysis we look at the 10% percentile boundary, meaning that 10% of the area has a probability of at least x% for the given tercile. -->

<!-- This threshold was set since we want a substantial part of the region to meet the threshold. It wasn't set at a higher percentage, because from the above analysis we saw that this barely occurred in the past 4 years. However, this threshold and method of aggregation is still open for discussion. 
 -->
Note: all these computations only cover the region of interest

```{code-cell} ipython3
leadtime=1
```

```{code-cell} ipython3
from dateutil.relativedelta import relativedelta
```

```{code-cell} ipython3
def get_forecastmonth(pub_month,leadtime):
    return pub_month+relativedelta(months=+int(leadtime))
df_stats_reg["for_start"]=df_stats_reg.apply(lambda x: get_forecastmonth(x.F,leadtime), axis=1)
df_stats_reg["for_start_month"]=df_stats_reg.for_start.dt.to_period("M")
df_stats_reg["for_end_month"]=df_stats_reg.apply(lambda x: get_forecastmonth(x.for_start,2), axis=1).dt.to_period("M")
```

```{code-cell} ipython3
aggr_meth="10quant_cell"
```

```{code-cell} ipython3
:tags: [remove_cell]

df_stats_reg["publication_month"]=df_stats_reg["F"].dt.to_period("M")
df_stats_reg_aggrmeth=df_stats_reg.pivot(index=['publication_month',"for_start_month","for_end_month",'L'], columns='C', values=aggr_meth).reset_index().rename(columns={0:"bel_avg",1:"normal",2:"abv_avg"})
#remove index name
df_stats_reg_aggrmeth = df_stats_reg_aggrmeth.rename_axis(None, axis=1)  
```

```{code-cell} ipython3
:tags: [remove_cell]

df_stats_reg_aggrmeth_lt=df_stats_reg_aggrmeth[df_stats_reg_aggrmeth.L==leadtime]
```

<!-- Below all publication months are shown, where the numbers indicate the 10% boundary for each tercile. Those that have a probability of at least 40 are marked in red. We can see that for only 3 months this occurred for the below average tercile. For the above average tercile this is a more common phenomenon. 

We can see that for all occurrences that there was an at least 40% probability, this only occurred in one tercile, i.e. this is also the dominant tercile.  However, the differences can be quite small, for example in March 2018 and March 2021. 

Especially around March 2021 we can see an interesting pattern, where in February and April the forecast indicates a higher probability of above average instead of below average precipitation. Note however that these are forecasting different periods. I.e. the forecast of March is projecting for AMJ while the one in April is projecting for MJJ.

When focussing on our months of interest, namely March and July, we can see that for March in 4 out of 5 years the below average was the dominant tercile. The opposite for July is true, where all years so far showed the above average as dominant tercile. -->

Below all publication months are shown, where the numbers indicate the cell with the maximum probability touching the region of interest for each tercile. Those that have a probability of at least 40% are marked in red. We can see that this occurs more often for above than below average. Moreover, it does occurr that both the below and above average tercile meet the threshold.

+++

Questions

- should there be a minimum gap in probabilities between the terciles? 
- should we somehow check that the forecast is consistent across leadtimes?
     - currently only displaying values for leadtime=1 month!

+++

Note: the NaNs in the table indicate a dry mask during those months

```{code-cell} ipython3
# df_stats_reg_aggrmeth_lt.drop("L",axis=1).set_index(["publication_month","for_start_month","for_end_month"]).to_csv(os.path.join(iri_exploration_dir,f"bfa_tercile_prob_l{leadtime}_{aggr_meth}.csv"))
```

```{code-cell} ipython3
:tags: [output_scroll]

df_stats_reg_aggrmeth_lt.drop("L",axis=1).set_index(["publication_month","for_start_month","for_end_month"]).style.apply(lambda x: ["color: red" if v >=40 else "" for v in x], axis = 1).set_precision(2)
```

The probabilities for March
<!-- and July  -->
are shown below, where the dominant tercile is highlighted

```{code-cell} ipython3
:tags: [hide_input]

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['color: red' if v else 'black' for v in is_max]

df_stats_reg_aggrmeth_lt[df_stats_reg_aggrmeth_lt.publication_month.dt.month.isin([3])].drop("L",axis=1).set_index(["publication_month"]).style.apply(highlight_max,axis=1).set_precision(2)
```

```{code-cell} ipython3
:tags: [hide_input]

# df_stats_reg_aggrmeth_lt[df_stats_reg_aggrmeth_lt.publication_month.dt.month.isin([7])].drop("L",axis=1).set_index(["publication_month"]).style.apply(highlight_max,axis=1).set_precision(2)
```
