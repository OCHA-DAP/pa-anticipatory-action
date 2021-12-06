# IRI forecast as a trigger for drought in Chad
This notebook explores the option of using IRI's seasonal forecast as part of drought-related trigger in Burkina Faso. 
From the country team the proposed trigger is:
- Trigger #1 in March covering June-July-August or July-August-September. Threshold desired: 60%.
- Trigger #2 in May covering June-July-August or July-August-September. Threshold desired: 60%. 
- Targeted Admin1s: Barh el Gazel, Batha, Kanem, Lac (une partie), Ouaddaï (une partie), Sila (une partie), Wadi Fira
- 20% of those admin1s meeting the threshold

This notebook explores if and when these triggers would be reached. As part of this exploration methods for aggregation from raster level to the percentage of the area are discussed. 


#### Skill
Before diving into any code, lets analyze the skill as produced by IRI. The GROC is shown below where grey indicates no skill, and white a dry mask. As can be seen from the images, over significant parts of Chad the forecasts don't show any skill.  

It also seems the skill becomes lower with a lower leadtime which is the opposite from the expected pattern. However the differences between leadtimes are small and thus should be interpreted with caution.


<img src="https://iri.columbia.edu/climate/verification/images/NAskillmaps/pcp/PR1_groc_jja_Ld3.gif" alt="drawing" width="700"/>
<img src="https://iri.columbia.edu/climate/verification/images/NAskillmaps/pcp/PR1_groc_jja_Ld1.gif" alt="drawing" width="700"/>
<img src="https://iri.columbia.edu/climate/verification/images/NAskillmaps/pcp/PR1_groc_jas_Ld4.gif" alt="drawing" width="700"/>
<img src="https://iri.columbia.edu/climate/verification/images/NAskillmaps/pcp/PR1_groc_jas_Ld2.gif" alt="drawing" width="700"/>


#### Load libraries and set global constants

```python
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import seaborn as sns
import cftime
import calendar
from dateutil.relativedelta import relativedelta
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
import hvplot.xarray
import altair as alt

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import get_iri_data,get_iri_data_dominant
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
hdx_blue="#007ce0"
```

```python
#month number refers to the last month of the season
month_season_mapping={1:"NDJ",2:"DJF",3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND"}
```

```python
iso3="tcd"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
adm1_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin1_shp"]
```

#### Set variables

```python
#TODO: some admins only part should be included, check with team
adm_sel=['Barh-El-Gazel','Batha','Kanem','Lac','Ouaddaï','Sila','Wadi Fira']
```

```python
#list of months and leadtimes that could be part of the trigger
#first entry refers to the publication month, second to the leadtime
trig_mom=[(3,3),(3,4),(5,1),(5,2)]
```

## Inspect forecasts


We load the iri data indicating the dominant tercile. The negative values indicate forecasted below average rainfall, and the positive values above average. We assign values between -40 and 40 as beloning to the normal tercile. On IRI's website the bins for the normal tercile are -37.5 and 37.5 so these differ slightly from what we use here. The reason for setting them to 40 for now is that in our opinion this is easier to interpret and generally preferred to use rounded numbers. 

We plot the forecast raster data for the periods and leadtimes of interest. The red areas are the admin1's we are focussing on. 
These figures are is similair to [the figure on the IRI Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html), except that the bins are defined slightly differently

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_reg=gdf_adm1[gdf_adm1.admin1Name.isin(adm_sel)]
```

```python
#F indicates the publication month, and L the leadtime. 
#A leadtime of 1 means a forecast published in May is forecasting JJA
ds_iri_dom=get_iri_data_dominant(config,download=False)
ds_iri_dom=ds_iri_dom.rio.write_crs("EPSG:4326",inplace=True)
da_iri_dom=ds_iri_dom.dominant
da_iri_dom_clip=da_iri_dom.rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
#not very neat function but does the job for now
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

```python
#iri website bins
# plt_levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
plt_levels=[-100,-70,-60,-50,-45,-40,40,45,50,60,70,100]
plt_colors=['#783200','#ab461e','#d18132','#e8b832','#fafa02','#ffffff','#d1f8cc','#acf8a0','#73bb6e','#3a82b3','#0e3bf4']
```

```python
plt_raster_iri(da_iri_dom_clip,pub_mon=3,lt=3,plt_levels=plt_levels,plt_colors=plt_colors)
```

```python
plt_raster_iri(da_iri_dom_clip,pub_mon=5,lt=1,plt_levels=plt_levels,plt_colors=plt_colors)
```

```python
plt_raster_iri(da_iri_dom_clip,pub_mon=3,lt=4,plt_levels=plt_levels,plt_colors=plt_colors)
```

```python
plt_raster_iri(da_iri_dom_clip,pub_mon=5,lt=2,plt_levels=plt_levels,plt_colors=plt_colors)
```

From the above plots we can conclude a couple of things: 
- Since 2017 no extremely high below average probabilities were forecasted in our region of interest. 
- The patterns in the region can differ, for example in 2021-03 where we see mainly above average, but with some below average areas in the eastern-south
- The forecasted patterns can change heavily with changing leadtime. For example for the JAS season with 4 and 2 months leadtime. 


Below we plot a few examples of "tricky" forecasts. For the left two: say the threshold would be at 40%, would the trigger be reached for the red region? For the right two plots we see a combination of below and above average across the region. What should we do with that? 

These figures are to guide the discussion on which forecasts we would have wanted to trigger and for which we wouldn't

```python
g=da_iri_dom_clip.where(da_iri_dom_clip.F.isin([cftime.Datetime360Day(2021, 2, 16, 0, 0, 0, 0),cftime.Datetime360Day(2021, 3, 16, 0, 0, 0, 0),cftime.Datetime360Day(2018, 7, 16, 0, 0, 0, 0),cftime.Datetime360Day(2019, 7, 16, 0, 0, 0, 0)]),drop=True).sel(L=3).plot(
    col="F",
    col_wrap=4,
    levels=plt_levels,
    colors=plt_colors,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        'ticks': plt_levels,
    },
    figsize=(20,10)
)
for ax in g.axes.flat:
    gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

### Which cells to include for aggregation?
For the trigger we have to aggregate a selection of raster cells to one number. Before we can do this, we have to decide which cells to include for the aggregation. 
We inspect 3 different methods: including all cells with their centre in the region, all cells touching the region, and an approximate weighted average. 

We should discuss as team (and with meteorologists) which is the best method. 
At least with all methods we include a substantial number of cells.

```python
#sel random values to enable plotting of included cells (so values are irrelevant)
da_iri_dom_blue=da_iri_dom.sel(F="2020-05-16",L=1).squeeze()
```

```python
da_iri_dom_blue_centre=da_iri_dom_blue.rio.clip(gdf_reg["geometry"], all_touched=False)
g=da_iri_dom_blue_centre.plot.imshow(cmap=ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with cell centres: {da_iri_dom_blue_centre.count().values} cells included")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
da_iri_dom_blue_touched=da_iri_dom_blue.rio.clip(gdf_reg["geometry"], all_touched=True)
g=da_iri_dom_blue_touched.plot.imshow(cmap=ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with all cells touching: {da_iri_dom_blue_touched.count().values} cells included")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
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

```python
g=da_iri_dom_blue_res.plot.imshow(cmap=ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with approx weighted average")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

### Threshold
While before we looked at the dominant tercile, we now load the data containing the probability for each individual tercile. We focus on the below-average tercile.

The proposed threshold by FAO was 60%. As can be seen in the image below, this is very high. 
The first plot shows all values of the region of interest across all seasons. We can see that the median is around 35 and it doesn't differ much across leadtimes. Values higher than 50 are very rare. 

We can see the same pattern when we only select the seasons and leadtimes that might be part of the trigger. 
<!-- Based on this we experiment with a threshold of 4 but it might also be set to 40 or 50, this is open for discussion. However, a threshold of 60% would be advised again as this is very unlikely from a meteorological perspective to be met when requiring 20% of the area to meet this condition. -->


For now we set the threshold to 40 for experimentation. With the reasoning that this might be the lowest threshold that could still be reasonable. However, we might want to increase the threshold to 45 or 50. How to determine this threshold is still to be disucssed we could either 
1) approach it from a meteorological perspective and discuss with scientists
2) understand probability values from a dataset that has a longer historical track record

```python
#C indicates the tercile (below-average, normal, or above-average).  
#F indicates the publication month, and L the leadtime
ds_iri = get_iri_data(config, download=False)
ds_iri=ds_iri.rio.write_crs("EPSG:4326",inplace=True)
da_iri=ds_iri.prob
#select all cells touching the region
da_iri_allt=da_iri.rio.clip(gdf_reg["geometry"], all_touched=True)
#C=0 indicates the below average tercile
da_iri_allt_bavg=da_iri_allt.sel(C=0)
```

```python
da_iri.sel(C=0).hvplot.kde('prob',alpha=0.5).opts(ylabel="Probability below average",
title="Forecasted probabilities of below average \n at raster level in the whole world across all seasons and leadtimes, 2017-2021")
```

```python
da_iri_allt_bavg.hvplot.violin('prob',by='L', color='L', cmap='Category20').opts(ylabel="Probability below average",xlabel="leadtime",
title="Observed probabilities of bavg at raster level in the region of interest")
```

```python
#transform data such that we can select by combinatiosn of publication month (F) and leadtime (L)
da_plt=da_iri_allt_bavg.assign_coords(F=da_iri_allt_bavg.F.dt.month)
da_plt=da_plt.stack(comb=["F","L"])
#only select data that is selected for trigger
da_iri_allt_trig_mom=xr.concat([da_plt.sel(comb=m) for m in trig_mom],dim="comb")
```

```python
da_iri_allt_trig_mom.hvplot.violin('prob').opts(ylabel="Probability below average",
                                                title="observed probabilities of bavg for the month and leadtime combinations \n included in the triger")
```

#### Compute stats
We can now compute the statistics of the region of interest. For now I am working with all cells touching the region, but this is something that still has to be thought about more.

We have to set two parameters: the minimum probability of below average, and the percentage of the area that should have this minimum probability assigned. 

For now we set the proability of below average threshold to 50 as this is the lowest that might still be reasonble, but this is something that has to be discussed furhter. 

For now we set the minimum percentage of the area that should reach the threshold to 20% as that was proposed by the Atelier. However, this is also open for discussion

```python
#% probability of bavg
threshold=40
#min percentage of the area that needs to reach the threshold
perc_area=20
```

```python
adm0_col="admin0Name"
pcode0_col="admin0Pcod"
```

```python
#compute stats
#dissolve the region to one polygon
gdf_reg_dissolved=gdf_reg.dissolve(by=adm0_col)
gdf_reg_dissolved=gdf_reg_dissolved[[pcode0_col,"geometry"]]

df_stats_reg_bavg=compute_raster_statistics(
        gdf=gdf_reg_dissolved,
        bound_col=pcode0_col,
        raster_array=da_iri_allt_bavg,
        lon_coord="longitude",
        lat_coord="latitude",
        stats_list=["min","mean","max","std","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
da_iri_allt_thresh=da_iri_allt_bavg.where(da_iri_allt_bavg>=threshold)
df_stats_reg_bavg_thresh=compute_raster_statistics(gdf=gdf_reg_dissolved,bound_col=pcode0_col,raster_array=da_iri_allt_thresh,
                                                   lon_coord="longitude",lat_coord="latitude",stats_list=["count"],
                                                  all_touched=True)

df_stats_reg_bavg["perc_thresh"] = df_stats_reg_bavg_thresh[f"count_admin0Pcod"]/df_stats_reg_bavg[f"count_admin0Pcod"]*100
df_stats_reg_bavg["F"]=pd.to_datetime(df_stats_reg_bavg["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_reg_bavg["month"]=df_stats_reg_bavg.F.dt.month
##leaving for now as might come in handy, but else delete in future
# df_stats_reg_bavg.time=pd.to_datetime(df_stats_reg_bavg.time.apply(lambda x: x.strftime("%Y-%m-%d")))
# df_stats_reg_bavg["end_time"]=pd.to_datetime(df_stats_reg_bavg["time"].apply(lambda x: x.strftime('%Y-%m-%d')))
# df_stats_reg_bavg["end_month"]=df_stats_reg_bavg.end_time.dt.to_period("M")
# df_stats_reg_bavg["start_time"]=df_stats_reg_bavg.end_time.apply(lambda x: x+relativedelta(months=-2))
# df_stats_reg_bavg["start_month"]=df_stats_reg_bavg.start_time.dt.to_period("M")
# df_stats_reg_bavg["season"]=df_stats_reg_bavg.end_month.apply(lambda x:month_season_mapping[x.month])
# df_stats_reg_bavg["seas_year"]=df_stats_reg_bavg.apply(lambda x: f"{x.season} {x.end_month.year}",axis=1)
# df_stats_reg_bavg["rainy_seas"]=np.where(df_stats_reg_bavg.start_month.dt.month.isin(end_months_sel),1,0)
# df_stats_reg_bavg=df_stats_reg_bavg.sort_values("start_month")
# df_stats_reg_bavg["rainy_seas_str"]=df_stats_reg_bavg["rainy_seas"].replace({0:"outside rainy season",1:"rainy season"})
# df_stats_reg_bavg["year"]=df_stats_reg_bavg.end_month.dt.year
```

NaN values indicate that the whole region is covered by a dry mask at that point. See [here](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/) for more information

```python
df_stats_reg_bavg=df_stats_reg_bavg[(~df_stats_reg_bavg.perc_thresh.isnull())]
```

```python
df_stats_reg_bavg=df_stats_reg_bavg.sort_values("perc_thresh",ascending=False)
```

## Analyze statistics probability below average


We plot the occurrences of the probability of below average being above the given threshold and given minimum percentage of the area. This so far is a preliminary analysis which can be improved once we have made some decisions. 

```python
print(f"{round(len(df_stats_reg_bavg[df_stats_reg_bavg['perc_thresh']>=perc_area])/len(df_stats_reg_bavg)*100)}%"
      f"({round(len(df_stats_reg_bavg[df_stats_reg_bavg['perc_thresh']>=perc_area]))}/{len(df_stats_reg_bavg)}) "
      "of forecasts across all seasons and leadtimes"
      f" predicted >={perc_area}% of the area >={threshold}% prob of below average")
```

```python
#select the months and leadtimes included in the trigger
df_stats_reg_bavg_trig_mom=df_stats_reg_bavg[df_stats_reg_bavg[['month', 'L']].apply(tuple, axis=1).isin(trig_mom)]
```

```python
histo=alt.Chart(df_stats_reg_bavg).mark_bar().encode(
    alt.X("perc_thresh:Q", bin=alt.Bin(step=1),title=f"% of region with >={threshold} probability of bavg"),
    y='count()',
).properties(title=[f"Occurence of the percentage of the region with >={threshold} probability of bavg","Red line indicates the threshold on the % of the area"])
line = alt.Chart(pd.DataFrame({'x': [perc_area]})).mark_rule(color="red").encode(x='x')
histo+line
```

```python
histo=alt.Chart(df_stats_reg_bavg_trig_mom).mark_bar().encode(
    alt.X("perc_thresh:Q", bin=alt.Bin(step=1),title=f"% of region with >={threshold} probability of bavg"),
    y='count()',
).properties(title=[f"Occurence of the percentage of the region with >={threshold} probability of bavg",
                    "For the publication months and leadtimes included in the trigger",
                    "Red line indicates the threshold on the % of the area"])
line = alt.Chart(pd.DataFrame({'x': [perc_area]})).mark_rule(color="red").encode(x='x')
histo+line
```

#### Dominant tercile
Just like with BFA we might also want to examine if the below average tercile is the dominant tercile. For BFA we required at the pixel level that 
probability below average >= (probability above average + 5%)


### Questions IRI / PA team
- Is it even worth using the forecasts if the GROC is <=0.5 (=grey in map)? 
- Can we set the probability threshold based on meteorological knowledge? 
- What is the preferred aggregation method? 
- Should we instead of looking at tercile forecasts look at the [flexible forecasts](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/precip_full.html)?  
- Would IRI be the best seasonal source or are there other sources that might be better?


### Extra


Would be great to also have the data as the date projected instead of date published. In that way we can compare how values change across leadtimes. We did implement this from ecmwf so could copy that. 
Below an attempt to do it in an easier fashion, but without success

```python
#https://stackoverflow.com/questions/67342119/xarray-merge-separate-day-and-hour-dimensions-into-one-time-dimension-in-python
ds_iri = get_iri_data(config, download=False)
ds_iri=ds_iri.assign_coords(L=[31,62,93,124])
ds_iri=ds_iri.assign_coords(F=ds_iri.F.values.astype("datetime64[M]"))
ds_iri=ds_iri.assign_coords(L=ds_iri.L.values.astype('timedelta64[D]'))
ds_iri=ds_iri.assign_coords(valid_time=ds_iri.F + ds_iri.L)
```
