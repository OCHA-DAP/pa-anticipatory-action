# IRI forecast as a trigger for drought in Chad
**Note: at some point we want to port this repository to its own repository according to our cookiecutter.
The analysis done in this notebook is very similair to BFA which already has its 
[own repo](https://github.com/OCHA-DAP/pa-aa-bfa-drought/blob/main/analysis/01_iri_seas_forecast.md)
so inspiration can be drawn from there once refactoring**

This notebook entails the analysis that was done for analyzing the IRI seasonal forecast as part of drought-related trigger in Chad. 
An initial proposal from in-country partners was:
- Trigger #1 in March covering June-July-August or July-August-September. Threshold desired: 60%.
- Trigger #2 in May covering June-July-August or July-August-September. Threshold desired: 60%. 
- Targeted Admin1s: Lac, Kanem, Barh-El-Gazel, Batha, and Wadi Fira
- 20% of those admin1s meeting the threshold

This notebook explores if and when these triggers would be reached. As part of this exploration methods for aggregation from raster level to the percentage of the area are discussed. 

There are four main conclusions:
1) Due to the limited data availability it is almost impossible to set an educated threshold. There is only data since 2017 and no drought events caused by large-scale lack of precipitation occurred during this period
2) The general skill (GROC) over TCD is mediocre. It is therefore advised to use the forecasts but with caution
3) A threshold of 60% is too high, as this has barely been reached at global level per raster cell, let alone over a larger area.
4) We instead recommend a threshold of4 42.5% over 20% of the area as this is expected to be reached from time to time but not too often. The 42.5% is specifically set to match with the bins of [IRI's graphics](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/). However 40% could also be a reasonable threshold


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
%load_ext jupyter_black
```

```python
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
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

from src.indicators.drought.iri_rainfallforecast import (
    get_iri_data,
    get_iri_data_dominant,
)
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
hdx_blue = "#007ce0"
```

```python
# month number refers to the last month of the season
month_season_mapping = {
    1: "NDJ",
    2: "DJF",
    3: "JFM",
    4: "FMA",
    5: "MAM",
    6: "AMJ",
    7: "MJJ",
    8: "JJA",
    9: "JAS",
    10: "ASO",
    11: "SON",
    12: "OND",
}
```

```python
iso3 = "tcd"
config = Config()
parameters = config.parameters(iso3)
country_data_raw_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
)
data_processed_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR
)
adm1_bound_path = (
    country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin1_shp"]
)
adm2_path = (
    data_processed_dir
    / iso3
    / config.SHAPEFILE_DIR
    / "tcd_adm2_area_of_interest.gpkg"
)
```

#### Set variables

```python
gdf_adm1 = gpd.read_file(adm1_bound_path)
gdf_adm2 = gpd.read_file(adm2_path)
incl_adm_col = "area_of_interest"
gdf_aoi = gdf_adm2[gdf_adm2[incl_adm_col] == True]
```

```python
# list of months and leadtimes that could be part of the trigger
# first entry refers to the publication month, second to the leadtime
trig_mom = [(3, 3), (3, 4), (5, 1), (5, 2)]
```

## Inspect forecasts


We load the iri data indicating the dominant tercile. The negative values indicate forecasted below average rainfall, and the positive values above average. We use the IRI website bins, where values between -37.5 and 37.5 are assigned to the normal tercile. We could also choose to use rounded bins instead(e.g. -40 to 40). 

We plot the forecast raster data for the periods and leadtimes of interest. The red areas are the admin1's we are focussing on. 
These figures are the same as [the figure on the IRI Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html).

```python
# F indicates the publication month, and L the leadtime.
# A leadtime of 1 means a forecast published in May is forecasting JJA
ds_iri_dom = get_iri_data_dominant(config, download=False)
ds_iri_dom = ds_iri_dom.rio.write_crs("EPSG:4326", inplace=True)
da_iri_dom = ds_iri_dom.dominant
da_iri_dom_clip = da_iri_dom.rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
# facet plot of raster forecast data
# not very neat function but does the job for now
def plt_raster_iri(
    da_iri_dom_clip,
    pub_mon,
    lt,
    plt_levels,
    plt_colors,
):
    for_seas = month_season_mapping[(pub_mon + lt + 1) % 12 + 1]
    g = (
        da_iri_dom_clip.where(
            da_iri_dom_clip.F.dt.month.isin([pub_mon]), drop=True
        )
        .sel(L=lt)
        .plot(
            col="F",
            col_wrap=5,
            levels=plt_levels,
            colors=plt_colors,
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.8,
                "aspect": 40,
                "pad": 0.1,
                "ticks": plt_levels,
            },
            figsize=(25, 7),
        )
    )
    for ax in g.axes.flat:
        gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
        gdf_aoi.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    g.fig.suptitle(
        f"Forecasts published in {calendar.month_abbr[pub_mon]} predicting {for_seas} (lt={lt}) \n The subtitles indicate the publishing date",
        y=1.1,
    );
```

```python
# iri website bins
plt_levels = [
    -100,
    -67.5,
    -57.5,
    -47.5,
    -42.5,
    -37.5,
    37.5,
    42.5,
    47.5,
    57.5,
    67.5,
    100,
]
# rounded bins for easier interpretability
# plt_levels=[-100,-70,-60,-50,-45,-40,40,45,50,60,70,100]
plt_colors = [
    "#783200",
    "#ab461e",
    "#d18132",
    "#e8b832",
    "#fafa02",
    "#ffffff",
    "#d1f8cc",
    "#acf8a0",
    "#73bb6e",
    "#3a82b3",
    "#0e3bf4",
]
```

```python
plt_raster_iri(
    da_iri_dom_clip,
    pub_mon=3,
    lt=3,
    plt_levels=plt_levels,
    plt_colors=plt_colors,
)
```

```python
plt_raster_iri(
    da_iri_dom_clip,
    pub_mon=5,
    lt=1,
    plt_levels=plt_levels,
    plt_colors=plt_colors,
)
```

```python
plt_raster_iri(
    da_iri_dom_clip,
    pub_mon=3,
    lt=4,
    plt_levels=plt_levels,
    plt_colors=plt_colors,
)
```

```python
plt_raster_iri(
    da_iri_dom_clip,
    pub_mon=5,
    lt=2,
    plt_levels=plt_levels,
    plt_colors=plt_colors,
)
```

From the above plots we can conclude a couple of things: 
- Since 2017 no extremely high below average probabilities were forecasted in our region of interest. 
- The patterns in the region can differ, for example in 2021-03 where we see mainly above average, but with some below average areas in the eastern-south
- The forecasted patterns can change heavily with changing leadtime. For example for the JAS season with 4 and 2 months leadtime.


Below we plot a few examples of "tricky" forecasts. For the left two: say the threshold would be at 40%, would the trigger be reached for the red region? For the right two plots we see a combination of below and above average across the region. What should we do with that? 

These figures are to guide the discussion on which forecasts we would have wanted to trigger and for which we wouldn't

```python
g = (
    da_iri_dom_clip.where(
        da_iri_dom_clip.F.isin(
            [
                cftime.Datetime360Day(2021, 2, 16, 0, 0, 0, 0),
                cftime.Datetime360Day(2021, 3, 16, 0, 0, 0, 0),
                cftime.Datetime360Day(2018, 7, 16, 0, 0, 0, 0),
                cftime.Datetime360Day(2019, 7, 16, 0, 0, 0, 0),
            ]
        ),
        drop=True,
    )
    .sel(L=3)
    .plot(
        col="F",
        col_wrap=4,
        levels=plt_levels,
        colors=plt_colors,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "ticks": plt_levels,
        },
        figsize=(20, 10),
    )
)
for ax in g.axes.flat:
    gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_aoi.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

### Which cells to include for aggregation?
For the trigger we have to aggregate a selection of raster cells to one number. Before we can do this, we have to decide which cells to include for the aggregation. 
We inspect 3 different methods: including all cells with their centre in the region, all cells touching the region, and an approximate mask. 

After discussion we concluded that the approximate mask is a valid method and thus use this further on.

```python
# sel random values to enable plotting of included cells (so values are irrelevant)
da_iri_dom_blue = da_iri_dom.sel(F="2020-05-16", L=1).squeeze()
```

```python
da_iri_dom_blue_centre = da_iri_dom_blue.rio.clip(
    gdf_aoi["geometry"], all_touched=False
)
g = da_iri_dom_blue_centre.plot.imshow(
    cmap=ListedColormap([hdx_blue]), figsize=(6, 10), add_colorbar=False
)
gdf_adm1.boundary.plot(ax=g.axes, color="grey")
g.axes.set_title(
    f"Included area with cell centres: {da_iri_dom_blue_centre.count().values} cells included"
)
gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
da_iri_dom_blue_touched = da_iri_dom_blue.rio.clip(
    gdf_aoi["geometry"], all_touched=True
)
g = da_iri_dom_blue_touched.plot.imshow(
    cmap=ListedColormap([hdx_blue]), figsize=(6, 10), add_colorbar=False
)
gdf_adm1.boundary.plot(ax=g.axes, color="grey")
g.axes.set_title(
    f"Included area with all cells touching: {da_iri_dom_blue_touched.count().values} cells included"
)
gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
# approximate of a mask
da_iri_dom_blue_res = da_iri_dom_blue.rio.reproject(
    da_iri_dom_blue.rio.crs,
    # resolution it will be changed to, original is 1
    resolution=0.05,
    # use nearest so cell values stay the same, only cut
    # into smaller pieces
    resampling=Resampling.nearest,
    nodata=np.nan,
).rio.clip(gdf_aoi["geometry"], all_touched=False)
```

```python
g = da_iri_dom_blue_res.plot.imshow(
    cmap=ListedColormap([hdx_blue]), figsize=(6, 10), add_colorbar=False
)
gdf_adm1.boundary.plot(ax=g.axes, color="grey")
g.axes.set_title(f"Included area with approx mask")
gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

### Threshold
While before we looked at the dominant tercile, we now load the data containing the probability for each individual tercile. We focus on the below-average tercile.

The proposed threshold by FAO was 60%. As can be seen in the image below, this is very high. 
The first plot shows all values across all raster cells in the world, across all seasons and leadtimes. We can see that the median is around 35. Values above 60 are very very rare and above 50 are already exreme. 
The second plot shows the values of only the raster cells that touch the region but across all seasons. We can see that the median is again around 35 and that the distribution doesn't differ much across leadtimes. Values higher than 50 are very rare. We should be aware though that we only have 5 years of data.  

Moreover, the pattern might be very different depending on the season. The third plot show the distribution when we only select the seasons and leadtimes that might be part of the trigger. We can again see a similair pattern, though the median is slighlty lower. However, we didn't observe below average precipitation the past 5 years so it is hard to say what the distribution might look like during a drought. 

We should also be aware that these plots show the values at raster cell level. If we thereafter require 20% of the area meeting the probability threshold, this is even less likely to occur.


Due to the limited data availability it is very hard to determine the threshold objectively. We do advise against the 60% threshold since even globally this phenomenon that seems too rare for our purpose. 

However a threshold anywhere between 40 and 50 could be reasonable. We experimented with these different thresholds. For now we propose a threshold of 42.5%. This because we estimate it to be already quite rare, in combination with the 20% of the area requirement, but at the same time we estimate it to be possible to occur. The reason we set it to 42.5 specifically is because this matches the IRI bins. Thus people can easily inspect the forecasts themselves on the maproom.

```python
# C indicates the tercile (below-average, normal, or above-average).
# F indicates the publication month, and L the leadtime
ds_iri = get_iri_data(config, download=False)
ds_iri = ds_iri.rio.write_crs("EPSG:4326", inplace=True)
da_iri = ds_iri.prob
# select all cells touching the region
da_iri_allt = da_iri.rio.clip(gdf_aoi["geometry"], all_touched=True)
# C=0 indicates the below average tercile
da_iri_allt_bavg = da_iri_allt.sel(C=0)
```

```python
# check that all touching is done correctly
g = da_iri_allt.sel(F="2021-03-16", L=2, C=0).plot()
gdf_adm1.boundary.plot(ax=g.axes)
```

```python
# upsample the resolution in order to create a mask of our aoi
resolution = 0.05
mask_list = []
for terc in da_iri_allt.C.values:
    for lt in da_iri_allt.L.values:
        da_terc_lt = da_iri_allt.sel(C=terc, L=lt)
        da_terc_lt_mask = da_terc_lt.rio.reproject(
            da_terc_lt.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        mask_list.append(da_terc_lt_mask.expand_dims({"C": [terc], "L": [lt]}))
da_iri_mask = (
    xr.combine_by_coords(mask_list)
    .rio.clip(gdf_aoi["geometry"], all_touched=False)
    .prob
)
# reproject changes longitude and latitude name to x and y
# so change back here
da_iri_mask = da_iri_mask.rename({"x": "longitude", "y": "latitude"})
da_iri_mask_bavg = da_iri_mask.sel(C=0)
```

```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2021-03-16", L=2, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
```

```python
da_iri.sel(C=0).hvplot.hist("prob", alpha=0.5).opts(
    ylabel="Probability below average",
    title="Forecasted probabilities of below average \n at raster level in the whole world across all seasons and leadtimes, 2017-2021",
)
```

```python
da_iri_mask_bavg.hvplot.violin(
    "prob", by="L", color="L", cmap="Category20"
).opts(
    ylabel="Probability below average",
    xlabel="leadtime",
    title="Observed probabilities of bavg at raster level in the region of interest",
)
```

```python
# transform data such that we can select by combination of publication month (F) and leadtime (L)
da_plt = da_iri_mask_bavg.assign_coords(F=da_iri_mask_bavg.F.dt.month)
da_plt = da_plt.stack(comb=["F", "L"])
# only select data that is selected for trigger
da_iri_mask_trig_mom = xr.concat(
    [da_plt.sel(comb=m) for m in trig_mom], dim="comb"
)
```

```python
da_iri_mask_trig_mom.hvplot.violin("prob").opts(
    ylabel="Probability below average",
    title="observed probabilities of bavg for the month and leadtime combinations \n included in the triger",
)
```

#### Compute stats
We can now compute the statistics of the region of interest. We use the approximate mask to define the cells included for the computation of the statistics. 

We have to set two parameters: the minimum probability of below average, and the percentage of the area that should have this minimum probability assigned. 

As discussed above we set the probability of below average to 42.5% (but experimentation with other thresholds has been done). 

For now we set the minimum percentage of the area that should reach the threshold to 20% as that was proposed by the Atelier. This seems reasonable to us as it is a substantial area thus possibly indicating widespread drought. At the same time requiring a larger percentage significantly lowers the chances of meeting the trigger, as we often see that extreme values are only forecasted in a smaller area.

```python
#% probability of bavg
threshold = 42.5
# min percentage of the area that needs to reach the threshold
perc_area = 20
```

```python
adm0_col = "admin0Name"
pcode0_col = "admin0Pcod"
```

```python
# compute stats
# dissolve the region to one polygon
gdf_aoi_dissolved = gdf_aoi.dissolve(by=adm0_col)
gdf_aoi_dissolved = gdf_aoi_dissolved[[pcode0_col, "geometry"]]

df_stats_reg_bavg = compute_raster_statistics(
    gdf=gdf_aoi_dissolved,
    bound_col=pcode0_col,
    raster_array=da_iri_mask_bavg,
    lon_coord="longitude",
    lat_coord="latitude",
    stats_list=["min", "mean", "max", "std", "count"],
    # computes value where 20% of the area is above that value
    percentile_list=[80],
    all_touched=True,
)
da_iri_mask_thresh = da_iri_mask_bavg.where(da_iri_mask_bavg >= threshold)
df_stats_reg_bavg_thresh = compute_raster_statistics(
    gdf=gdf_aoi_dissolved,
    bound_col=pcode0_col,
    raster_array=da_iri_mask_thresh,
    lon_coord="longitude",
    lat_coord="latitude",
    stats_list=["count"],
    all_touched=True,
)

df_stats_reg_bavg["perc_thresh"] = (
    df_stats_reg_bavg_thresh[f"count_admin0Pcod"]
    / df_stats_reg_bavg[f"count_admin0Pcod"]
    * 100
)
df_stats_reg_bavg["F"] = pd.to_datetime(
    df_stats_reg_bavg["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
)
df_stats_reg_bavg["month"] = df_stats_reg_bavg.F.dt.month
```

NaN values indicate that the whole region is covered by a dry mask at that point. See [here](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/) for more information

```python
df_stats_reg_bavg = df_stats_reg_bavg[
    (~df_stats_reg_bavg.perc_thresh.isnull())
]
```

```python
df_stats_reg_bavg = df_stats_reg_bavg.sort_values(
    "perc_thresh", ascending=False
)
```

```python
df_stats_reg_bavg[df_stats_reg_bavg.perc_thresh >= 20]
```

## Analyze statistics probability below average


We plot the occurrences of the probability of below average being above the given threshold and given minimum percentage of the area. This so far is a preliminary analysis which can be improved once we have made some decisions.

```python
print(
    f"{round(len(df_stats_reg_bavg[df_stats_reg_bavg['perc_thresh']>=perc_area])/len(df_stats_reg_bavg)*100)}%"
    f"({round(len(df_stats_reg_bavg[df_stats_reg_bavg['perc_thresh']>=perc_area]))}/{len(df_stats_reg_bavg)}) "
    "of forecasts across all seasons and leadtimes"
    f" predicted >={perc_area}% of the area >={threshold}% prob of below average"
)
```

```python
# trig_mom includes the forecasts released in march and may
# here we also add the forecasts of April and June for testing
# the second parameter of the tuple is the leadtime
trig_mom_all = trig_mom + [(4, 2), (4, 3), (6, 1)]
```

```python
# select the months and leadtimes included in the trigger
df_stats_reg_bavg_trig_mom = df_stats_reg_bavg[
    df_stats_reg_bavg[["month", "L"]].apply(tuple, axis=1).isin(trig_mom_all)
]
```

```python
histo = (
    alt.Chart(df_stats_reg_bavg)
    .mark_bar()
    .encode(
        alt.X(
            "perc_thresh:Q",
            bin=alt.Bin(step=1),
            title=f"% of region with >={threshold} probability of bavg",
        ),
        y="count()",
    )
    .properties(
        title=[
            f"Occurence of the percentage of the region with >={threshold} probability of bavg",
            "Across all seasons and leadtimes",
            "Red line indicates the threshold on the % of the area",
        ]
    )
)
line = (
    alt.Chart(pd.DataFrame({"x": [perc_area]}))
    .mark_rule(color="red")
    .encode(x="x")
)
histo + line
```

```python
histo = (
    alt.Chart(df_stats_reg_bavg_trig_mom)
    .mark_bar()
    .encode(
        alt.X(
            "perc_thresh:Q",
            bin=alt.Bin(step=1),
            title=f"% of region with >={threshold} probability of bavg",
        ),
        y="count()",
    )
    .properties(
        title=[
            f"Occurence of the percentage of the region with >={threshold} probability of bavg",
            "For the publication months and leadtimes included in the trigger",
            "Red line indicates the threshold on the % of the area",
        ]
    )
)
line = (
    alt.Chart(pd.DataFrame({"x": [perc_area]}))
    .mark_rule(color="red")
    .encode(x="x")
)
histo + line
```

```python
df_stats_reg_bavg_trig_mom["pred_month"] = df_stats_reg_bavg_trig_mom.apply(
    lambda x: x["F"] + relativedelta(months=int(x["L"])), axis=1
)
```

```python
df_stats_reg_bavg_trig_mom.sort_values(["pred_month", "L"]).head()
```

#### Dominant tercile
Just like with BFA we might also want to examine if the below average tercile is the dominant tercile. For BFA we required at the pixel level that 
probability below average >= (probability above average + 5%)

```python
da_iri.where(
    (da_iri.sel(C=0) >= 40) & (da_iri.sel(C=0) - da_iri.sel(C=2) <= 5),
    drop=True,
).sel(C=0).hvplot.hist("prob", alpha=0.5)
```
