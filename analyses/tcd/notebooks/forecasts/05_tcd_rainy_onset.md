## Chad - start of the rainy season
As we saw in previous analyses, the observation of below average seasonal precipitation doesn't show a strong correspondence with drought impact years nor Biomasse, see e.g. `04_tcd_boostrap_seas_bavg_precip_observed.md` 

Out of curiosity, I started exploring if other measures of precipitation show different patterns. This notebook explores the start of the rainy season from 2000-2021. As the onset is a topic that is often mentioned as critical, and [research has shown](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0242883) it shows a high correlation with peak NDVI . 

Two definitions are explored. One is that of 50 milimeter cumulative rainfall. As this is likely the definition that PRESASS uses from informal conversation, but official documentation is missing. 

The other is at least 25 mm of cumulative rainfall during 10 days, followed by at least 20mm cumulative rainfall during the next 20 days. This has been cited to be the definition used by AGRHYMET, see [here](https://journals.sagepub.com/doi/pdf/10.1177/1178622118790264) and [here](https://journals.ametsoc.org/view/journals/clim/18/16/jcli3423.1.xml#i1520-0442-18-16-3356-AGRHYMET1) (though it seems we have to travel to Niger to get the original document). Moreover, it [is noted](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0242883) that this definition is often used in the region, for example for the WRSI. 

Note that other definitions also exist. Based purely on precipitation, but sometimes also [including NDVI](https://d1wqtxts1xzle7.cloudfront.net/42927493/The_ITHACA_Early_Warning_System_for_drou20160222-4896-hbwwpf-with-cover-page-v2.pdf?Expires=1643101191&Signature=e3SnddtfycdxiBOmdIu0315Jb4FeK7ksQANexN9cJpEI7~N0eSkE0Fkcaqei5JZ~RdWMzVxAsws1TDeN5pczQAHm-zkVSpf-qdGM2kLMCvIyIRo7kw6-ckOfyvTBiOEXvx8kwdKrA9TXrFL3BtDQM2oDeFJ8dcl5ecQ2NBekQ1XuRH8bkj7uzwpeujjyZoZG4BOjH0zQElczqcI-YFXkkuVx4e6grtErBuySjlp45aUtiw4tXrM6vEguvfOQKRQKTJiMEy4UvAy-qUXwhXJ74ow21GpGrTzyW2APFMPZ9WZv-IwqfE3HLK9YgfQcGEULMnm46y2r5JZgbmFEyr80jQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) or [crop requirements](https://journals.sagepub.com/doi/pdf/10.1177/1178622118790264). 

For this exploration we use ARC2 data as the code for this is easy to use (thanks Seth!). It should be noted that we saw large differences between CHIRPS and ARC2 in `tcd_compare_chirps_arc2.md`. For this exploration we mainly look at the relative differences between years, but if we would use the dates as absolute dates in the future, we should carefully consider which dataset to use. 

We didn't end up using the rainy onset in the trigger but this exploration might be interesting for future pilots. 

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import os
from pathlib import Path
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import rioxarray
import xarray as xr
from scipy.stats import zscore
import numpy as np

import altair as alt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells
from src.indicators.drought.config import Config

config = Config()

data_processed_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR
)
```

```python
iso3 = "tcd"
```

```python
# adm2_path=data_processed_dir / iso3 / config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"
# gdf_adm2=gpd.read_file(adm2_path)
# gdf_aoi = gdf_adm2[gdf_adm2.area_of_interest == True]

# sometimes problems with loading the gpkg. This should give the same result
parameters = config.parameters(iso3)
adm1_path = (
    Path(os.getenv("AA_DATA_DIR"))
    / "public"
    / "raw"
    / iso3
    / "cod_ab"
    / parameters["path_admin1_shp"]
)
gdf_adm1 = gpd.read_file(adm1_path)
gdf_aoi = gdf_adm1[
    gdf_adm1.admin1Name.isin(
        ["Lac", "Kanem", "Barh-El-Gazel", "Batha", "Wadi Fira"]
    )
]
```

### Load ARC2 data

We first download the ARC2 data. ARC2 is daily data. We also group it to monthly and yearly data to compare with CHIRPS. 

```python
# #get bounds to define range to download
# gdf_adm1.total_bounds
# define class
arc2 = DrySpells(
    country_iso3=iso3,
    polygon_path=adm1_path,
    bound_col="admin1Pcod",
    monitoring_start="2000-01-01",
    monitoring_end="2021-12-31",
    range_x=("13E", "25E"),
    range_y=("7N", "24N"),
)

# #download data, only needed if not downloaded yet
# arc2.download_data(main=True)
```

```python
#want "T" as datetime coord. For some reason not working correctly
#with convert_date=True, so doing this little hack
#when revisiting, convert_date=False seems to already produce a datetimeindex
#I am confused
da_arc = arc2.load_raw_data(convert_date=False)
# da_arc['T'] = da_arc.indexes['T'].to_datetimeindex()
```

```python
# rename because .T is taking the transpose so naming it time makes sure there is no confusion
da_arc = da_arc.rename({"T": "time"})
# units attrs is very long list of "mm/day" so set to just "mm/day", mainly for plotting
da_arc.attrs["units"] = "mm/day"
da_arc = da_arc.assign_coords({"year": da_arc.time.dt.year})
```

```python
# high resolution so just take all cells with their centre in the region
da_arc_aoi = da_arc.rio.clip(gdf_aoi["geometry"])
```

### Rainy season onset: >=50 mm cumulative rainfall
We don't know from which date we should take the cumulative sum, so we assume it is from the 1st of January. 


Compute the first date per year that >=50mm for each raster cell

```python
# I would think this should be doable with a groupby instead of for
# loop but I couldn't manage to (due to the idxmin)
list_ds_rainy = []
for y in np.unique(da_arc_aoi.time.dt.year.values):
    da_sel = da_arc_aoi.sel(time=da_arc_aoi.time.dt.year == y)

    # take the cumsum
    da_sel_met_min = (
        da_sel.cumsum(dim="time")
        .
        # select dates where cumsum is at least 50
        where(lambda x: x >= 50)
        .
        # per coordinate, take the time point that first reaches 50
        idxmin(dim="time")
    )
    # dims and vars are confusing in xarray
    # when immediately making `year` a dim, the above code is veery slow
    # however, with concating we need it to be a dim
    # so remove it as coord and add it as dim
    list_ds_rainy.append(da_sel_met_min.drop("year").expand_dims(year=[y]))
da_start_rainy = xr.concat(list_ds_rainy, dim="year")
```

```python
da_start_rainy = da_start_rainy.rename("rainy_start")
```

```python
df_start_rainy = da_start_rainy.to_dataframe()
# nan for coordinates outside the region (or if 50mm is never met)
df_start_rainy = df_start_rainy.dropna()
```

We want to somehow aggregate from the raster cells to the region. We now do this by taking the median. Imo this is sensible but of course we could think about other methods as well

```python
# `median` is not working for time data, but quantile is
df_start_rainy_presass_med = (
    df_start_rainy.groupby("year")
    .rainy_start.quantile(0.5, interpolation="midpoint")
    .to_frame()
)
```

```python
df_start_rainy_presass_med[
    "month-day"
] = df_start_rainy_presass_med.rainy_start.dt.strftime("%m-%d")
```

```python
df_start_rainy_presass_med.sort_values("month-day", ascending=False)
```

The above table shows the median onset of the rainy season across the AOI. From here we can see that the 5/6 years with the latest onset are 2000, 2002, 2006, 2011, 2014, and 2015. 

We can also see that for many of the years the onsets are quite close to each other. 


### AGRHYMET rainy onset
For comparison, we also compute the rainy onset according to the AGRHYMET definition. That is 10 days with at least 25mm cumulative rainfall, followed by 20 days with at least 20mm cumulative rainfall. The onset is the first date of the 10 day period

```python
# compute 10 day rolling sum
# Rolling is right-centered, so last day of period
da_rolling_ten = (
    da_arc_aoi.rolling(time=10, min_periods=10)
    .sum()
    .dropna(dim="time", how="all")
)
```

```python
# compute 20 day rolling sum
da_rolling_twenty = (
    da_arc_aoi.rolling(time=20, min_periods=20)
    .sum()
    .dropna(dim="time", how="all")
)
```

```python
# dates where 10 day rolling sum >=25
da_rolling_ten_met = da_rolling_ten.where(lambda x: x >= 25)
```

```python
# dates where 20 day rolling sum >=20
da_rolling_twenty_met = da_rolling_twenty.where(lambda x: x >= 20)
```

```python
# rolling is right-centered
# we want the 10 day and 20 day da's to have the same dates to compare with each other
# we thus shift the 20 day rolling sum 20 days backward, such that the date indicates
# the day before the start of the rolling sum.
da_rolling_twenty_met_shift = da_rolling_twenty_met.copy()
da_rolling_twenty_met_shift["time"] = da_rolling_twenty_met_shift.indexes[
    "time"
].shift(-20, "D")
```

```python
# set all occurences to nan when either the 10 day or 20 day condition is not met
da_both_met = da_rolling_ten_met.where(
    ~(np.isnan(da_rolling_ten_met) | np.isnan(da_rolling_twenty_met_shift))
)
```

```python
# shift 9 days back to mark the beginning instead of the end
# of the 10 day rolling period
da_both_met["time"] = da_both_met.indexes["time"].shift(-9, "D")
```

```python
# select the first occurence of both conditions being met
# per year and raster cell
list_ds_rainy = []
for y in np.unique(da_both_met.time.dt.year.values):
    da_sel = da_both_met.sel(time=da_both_met.time.dt.year == y)
    woo = da_sel.idxmin(dim="time")
    list_ds_rainy.append(woo.drop("year").expand_dims(year=[y]))
da_start_rainy = xr.concat(list_ds_rainy, dim="year")
```

```python
da_start_rainy = da_start_rainy.rename("rainy_start")
```

```python
df_start_rainy = da_start_rainy.to_dataframe()
# nan for coordinates outside the region (or if 50mm is never met)
df_start_rainy = df_start_rainy.dropna()
```

```python
# `median` is not working for time data, but quantile is
df_start_rainy_agrhymet_med = (
    df_start_rainy.groupby("year")
    .rainy_start.quantile(0.5, interpolation="midpoint")
    .to_frame()
)
```

```python
df_start_rainy_agrhymet_med[
    "month-day"
] = df_start_rainy_agrhymet_med.rainy_start.dt.strftime("%m-%d")
```

```python
df_start_rainy_agrhymet_med.sort_values("month-day", ascending=False)
```

The above table shows the median onset of the rainy season across the AOI according to the AGRHYMET definition. From here we can see that the 5/6 years with the latest onset are 
2006, 2010, 2011, 2014, 2015. This shows quite a close correspondence with the result of the definition of PRESASS. The difference is that in the AGRHYMET definition 2010 is included, while in PRESASS there were 2002 and 2006. 

We can also see that the absolute dates differ a bit between the two definitions, with AGRHYMET indicating later onsets. 


We can compare the worst years of onset with the worst years the other data sources. This is shown in the heatmap below. 

```python
df_years = pd.DataFrame(
    index=range(2000, 2021),
    columns=[
        "impact_years",
        "onset_presass",
        "onset_agrhymet",
        "bavg_precip",
        "biomasse",
    ],
)
df_years.loc[:, :] = False
```

```python
df_years.loc[[2001, 2004, 2009, 2011, 2017], "impact_years"] = True
df_years.loc[[2000, 2002, 2006, 2011, 2014, 2015], "onset_presass"] = True
df_years.loc[[2006, 2010, 2011, 2014, 2015], "onset_agrhymet"] = True
df_years.loc[[2000, 2004, 2008, 2013, 2015], "bavg_precip"] = True
df_years.loc[[2002, 2004, 2006, 2009, 2011], "biomasse"] = True
df_years["year"] = df_years.index
```

```python
df_years_long = pd.melt(
    df_years,
    id_vars="year",
    value_vars=[
        "impact_years",
        "onset_presass",
        "onset_agrhymet",
        "bavg_precip",
        "biomasse",
    ],
)
```

```python
heatmap_worst = (
    alt.Chart(df_years_long)
    .mark_rect()
    .encode(
        x="year:N",
        y=alt.Y(
            "variable:N",
            sort=[
                "impact_years",
                "onset_presass",
                "onset_agrhymet",
                "bavg_precip",
                "biomasse",
            ],
        ),
        color=alt.Color(
            "value:N",
            scale=alt.Scale(range=["#D3D3D3", "red"]),
            legend=alt.Legend(title="5 worst years"),
        ),
    )
    .properties(title="5 worst years per source")
)
heatmap_worst
```

Interestingly we see that 2004 didn't have a late onset while all other sources show this was a drought year. Also 2009 isn't shown to have a late rainy onset while it was in the list of impact years and  biomasse did show low values.

The onset shows a little bit more overlap with Biomasse than below average precipitation does, though there are also still differences. For the impact years we don't see more overlap. This is shown more clearly in the confusion matrices below

```python
fig, ax = plt.subplots()
cm = confusion_matrix(
    y_target=df_years["onset_presass"], y_predicted=df_years["onset_agrhymet"]
)
plot_confusion_matrix(
    conf_mat=cm,
    show_absolute=True,
    show_normed=True,
    axis=ax,
    class_names=["No", "Yes"],
)
ax.set_ylabel("PRESASS")
ax.set_xlabel("AGRHYMET");
```

```python
fig = plt.figure(figsize=(15, 8))
i = 1
for onset in ["onset_presass", "onset_agrhymet"]:
    for comp in ["impact_years", "bavg_precip", "biomasse"]:
        ax = fig.add_subplot(2, 3, i)
        cm = confusion_matrix(
            y_target=df_years[onset], y_predicted=df_years[comp]
        )
        plot_confusion_matrix(
            conf_mat=cm,
            show_absolute=True,
            show_normed=True,
            axis=ax,
            class_names=["No", "Yes"],
        )
        ax.set_ylabel(onset)
        ax.set_xlabel(comp)
        i += 1
```
