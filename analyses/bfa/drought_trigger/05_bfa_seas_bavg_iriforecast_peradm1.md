# IRI forecast as a trigger for drought in Burkina Faso - per ADMIN1
### Note: while this was an experiment, we decided for the trigger to not go down to admin1 level, since this is meteorologically unsound. Seasonal forecasts are designed to analyze patterns at larger spatial scales and have a high geographical uncertainty. 

This notebook explores the option of using IRI's seasonal forecast as the indicator for a drought-related trigger in Burkina Faso. 
From the country team the proposed trigger is:
- Trigger #1 in March covering June-July-August. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%. 
- Targeted Admin1s: Boucle de Mounhoun, Centre Nord, Sahel, Nord.

This notebook explores these two triggers **per admin area**. The notebook `bfa_seas_bavg_iriforecast` explores these triggers when treating the targeted admin1's as one region. That notebook also provides the context, which is not duplicated here.

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
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

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import get_iri_data
from src.utils_general.raster_manipulation import (
    invert_latlon,
    change_longitude_range,
    fix_calendar,
)
```

## Inspect forecasts

```python
adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]
threshold_mar = 40
threshold_jul = 50
```

```python
leadtime_mar = 3
leadtime_jul = 1
glue("leadtime_mar", leadtime_mar)
glue("leadtime_jul", leadtime_jul)
```

```python
country = "bfa"
config = Config()
parameters = config.parameters(country)
country_data_raw_dir = os.path.join(
    config.DATA_DIR, config.PUBLIC_DIR, config.RAW_DIR, country
)
country_data_exploration_dir = os.path.join(
    config.DATA_DIR, config.PUBLIC_DIR, "exploration", country
)
glb_data_exploration_dir = os.path.join(
    config.DATA_DIR, config.PUBLIC_DIR, "exploration", "glb"
)
iri_exploration_dir = os.path.join(country_data_exploration_dir, "iri")
stats_reg_path = os.path.join(
    country_data_exploration_dir,
    f"{country}_iri_seasonal_forecast_stats_{''.join(adm_sel)}.csv",
)

adm1_bound_path = os.path.join(
    country_data_raw_dir, config.SHAPEFILE_DIR, parameters["path_admin1_shp"]
)
adm2_bound_path = os.path.join(
    country_data_raw_dir, config.SHAPEFILE_DIR, parameters["path_admin2_shp"]
)
```

```python
iri_ds = get_iri_data(config, download=False).rio.write_crs("EPSG:4326")
```

```python
gdf_adm1 = gpd.read_file(adm1_bound_path)
iri_clip = iri_ds.rio.clip(
    gdf_adm1.geometry.apply(mapping), iri_ds.rio.crs, all_touched=True
)
```

```python
gdf_reg = gdf_adm1[gdf_adm1.ADM1_FR.isin(adm_sel)]
```

Below the raw forecast data of below-average rainfall with {glue:text}`leadtime_mar` month leadtime, published in March is shown. The red areas are the 4 admin1's we are focussing on

The negative values indicate below average rainfall, and the positive values above average.


This is similair to [the figure on the IRI Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html), except that the bins are defined slightly differently

```python
dom_ds
```

```python
dom_ds = xr.open_dataset(
    os.path.join(
        glb_data_exploration_dir, "iri", "iri_seasfor_tercile_dominant.nc"
    ),
    decode_times=False,
    drop_variables="C",
)
dom_ds = dom_ds.rename({"X": "lon", "Y": "lat"})
# often IRI latitude is flipped so check for that and invert if needed
dom_ds = invert_latlon(dom_ds, lon_coord="lon", lat_coord="lat")
dom_ds = change_longitude_range(dom_ds, lon_coord="lon")
dom_ds = fix_calendar(dom_ds, timevar="F")
dom_ds = xr.decode_cf(dom_ds)
dom_clip = (
    dom_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    .rio.write_crs("EPSG:4326")
    .rio.clip(
        gdf_adm1.geometry.apply(mapping), dom_ds.rio.crs, all_touched=True
    )
)
levels = [-100, -70, -60, -50, -45, -40, 40, 45, 50, 60, 70, 100]
# iri website bins
# levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
colors = [
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
g = (
    dom_clip.where(dom_clip.F.dt.month.isin([3]), drop=True)
    .sel(L=3)
    .dominant.plot(
        col="F",
        col_wrap=5,
        levels=levels,
        colors=colors,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "ticks": levels,
        },
        figsize=(25, 7),
    )
)
df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")

g.fig.suptitle(
    "Forecasts published in March with 3 months leadtime \n The subtitles indicate the publishing month"
)
# g.fig.tight_layout()
plt.savefig(
    os.path.join(
        country_data_exploration_dir, "plots", "iri", "bfa_irifor_mar_l3.png"
    )
)
```

The same figure, but for the forecasts published in July with a {glue:text}`leadtime_jul` month leadtime are shown below

```python
dom_ds = xr.open_dataset(
    os.path.join(
        glb_data_exploration_dir, "iri", "iri_seasfor_tercile_dominant.nc"
    ),
    decode_times=False,
    drop_variables="C",
)
dom_ds = dom_ds.rename({"X": "lon", "Y": "lat"})
# often IRI latitude is flipped so check for that and invert if needed
dom_ds = invert_latlon(dom_ds, lon_coord="lon", lat_coord="lat")
dom_ds = change_longitude_range(dom_ds, lon_coord="lon")
dom_ds = fix_calendar(dom_ds, timevar="F")
dom_ds = xr.decode_cf(dom_ds)
dom_clip = (
    dom_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    .rio.write_crs("EPSG:4326")
    .rio.clip(
        gdf_adm1.geometry.apply(mapping), dom_ds.rio.crs, all_touched=True
    )
)
levels = [-100, -70, -60, -50, -45, -40, 40, 45, 50, 60, 70, 100]
# iri website bins
# levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
colors = [
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
g = (
    dom_clip.where(dom_clip.F.dt.month.isin([7]), drop=True)
    .sel(L=1)
    .dominant.plot(
        col="F",
        col_wrap=5,
        levels=levels,
        colors=colors,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "ticks": levels,
        },
        figsize=(25, 7),
    )
)
df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")

g.fig.suptitle(
    "Forecasts published in July with 1 month leadtime \n The subtitles indicate the publishing month"
)
# g.fig.tight_layout()
plt.savefig(
    os.path.join(
        country_data_exploration_dir, "plots", "iri", "bfa_irifor_jul_l1.png"
    )
)
```

```python
def interpolate_ds(
    ds, transform, upscale_factor, lon_coord="longitude", lat_coord="latitude"
):
    # Interpolated data
    new_lon = np.linspace(
        ds[lon_coord][0],
        ds[lon_coord][-1],
        ds.dims[lon_coord] * upscale_factor,
    )
    new_lat = np.linspace(
        ds[lat_coord][0],
        ds[lat_coord][-1],
        ds.dims[lat_coord] * upscale_factor,
    )

    # choose nearest as interpolation method to assure no new values are introduced but instead old values are divided into smaller raster cells
    # TODO: also change this to lat_coord and lon_coord somehow
    dsi = ds.interp(latitude=new_lat, longitude=new_lon, method="nearest")
    #     transform_interp=transform*transform.scale(len(ds.longitude)/len(dsi.longitude),len(ds.latitude)/len(dsi.latitude))

    return dsi  # , transform_interp
```

```python
iri_clip_interp = interpolate_ds(iri_clip, iri_clip.rio.transform(), 8)
```

```python
def compute_zonal_stats_xarray(
    raster,
    shapefile,
    lon_coord="longitude",
    lat_coord="latitude",
    var_name="prob",
):
    raster_clip = raster.rio.set_spatial_dims(
        x_dim=lon_coord, y_dim=lat_coord
    ).rio.clip(
        shapefile.geometry.apply(mapping), raster.rio.crs, all_touched=False
    )
    grid_mean = raster_clip.mean(dim=[lon_coord, lat_coord]).rename(
        {var_name: "mean_cell"}
    )
    grid_min = raster_clip.min(dim=[lon_coord, lat_coord]).rename(
        {var_name: "min_cell"}
    )
    grid_max = raster_clip.max(dim=[lon_coord, lat_coord]).rename(
        {var_name: "max_cell"}
    )
    grid_std = raster_clip.std(dim=[lon_coord, lat_coord]).rename(
        {var_name: "std_cell"}
    )
    grid_quant90 = raster_clip.quantile(
        0.9, dim=[lon_coord, lat_coord]
    ).rename({var_name: "10quant_cell"})
    grid_percth40 = (
        raster_clip.where(raster_clip.prob >= 40).count(
            dim=[lon_coord, lat_coord]
        )
        / raster_clip.count(dim=[lon_coord, lat_coord])
        * 100
    )
    grid_percth40 = grid_percth40.rename({var_name: "40percth_cell"})
    raster_diff_bel_abv = raster_clip.sel(C=0) - raster_clip.sel(C=2)
    grid_dom = (
        raster_clip.sel(C=0)
        .where((raster_clip.sel(C=0).prob >= 40) & (raster_diff_bel_abv >= 5))
        .count(dim=[lon_coord, lat_coord])
        / raster_clip.count(dim=[lon_coord, lat_coord])
        * 100
    )
    grid_dom = grid_dom.rename({var_name: "40th_bavg_cell"})
    zonal_stats_xr = xr.merge(
        [
            grid_mean,
            grid_min,
            grid_max,
            grid_std,
            grid_quant90,
            grid_percth40,
            grid_dom,
        ]
    ).drop("spatial_ref")
    zonal_stats_df = zonal_stats_xr.to_dataframe()
    zonal_stats_df = zonal_stats_df.reset_index()
    return zonal_stats_df
```

```python
# compute the stats per admin1, so not over the whole region at once
stats_df_list = []
for a in adm_sel:
    gdf_adm = gdf_adm1[gdf_adm1.ADM1_FR == a]
    stats_adm = compute_zonal_stats_xarray(iri_clip_interp, gdf_adm)
    stats_adm["F"] = pd.to_datetime(
        stats_adm["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
    )
    stats_adm["month"] = stats_adm.F.dt.month
    stats_adm["admin"] = a
    stats_df_list.append(stats_adm)
stats_per_adm = pd.concat(stats_df_list)
```

```python
len(stats_per_adm[stats_per_adm["40th_bavg_cell"] >= 10]) / len(
    stats_per_adm
) * 100
```

```python
stats_per_adm_bavg = stats_per_adm[(stats_per_adm.C == 0)]
```

And compute the statistics over this region, see a subset below

```python
stats_per_adm[
    (stats_per_adm.C == 0)
    & (stats_per_adm.L == leadtime_mar)
    & (stats_per_adm.F.dt.month == 3)
]
```

## Analyze statistics probability below average

```python
num_for_dates = len(stats_per_adm_bavg.F.unique())
glue("num_for_dates", num_for_dates)
```

Below the distribution of probability values is shown per admin area. \
This only includes the values for the below-average tercile, with a leadtime of {glue:text}`leadtime_mar` and {glue:text}`leadtime_jul`. \
Since we have data from Mar 2017, each distribution contains {glue:text}`num_for_dates` datapoints.

```python
# plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig, ax = plt.subplots(figsize=(10, 5))
g = sns.boxplot(
    data=stats_per_adm_bavg[stats_per_adm_bavg.L == leadtime_mar],
    x="admin",
    y="max_cell",
    ax=ax,
    color="#007CE0",
)
ax.set_ylabel("Probability")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_title("Leadtime = 3 months")
ax.set_xlabel("Admin");
```

```python
# plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig, ax = plt.subplots(figsize=(10, 5))
g = sns.boxplot(
    data=stats_per_adm_bavg[stats_per_adm_bavg.L == leadtime_jul],
    x="admin",
    y="max_cell",
    ax=ax,
    color="#007CE0",
)
ax.set_ylabel("Probability")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_title("Leadtime = 1 month")
ax.set_xlabel("Admin");
```

```python
stats_mar = stats_per_adm_bavg.loc[
    (stats_per_adm_bavg.F.dt.month == 3)
    & (stats_per_adm_bavg.L == leadtime_mar)
]
stats_jul = stats_per_adm_bavg.loc[
    (stats_per_adm_bavg.F.dt.month == 7)
    & (stats_per_adm_bavg.L == leadtime_jul)
]
```

```python
stats_mar[stats_mar.max_cell >= threshold_mar].admin.unique()
```

```python
def comb_list_string(str_list):
    if len(str_list) > 0:
        return " in " + ", ".join(str_list)
    else:
        return ""


num_trig_mar = len(
    stats_mar.loc[stats_mar["max_cell"] >= threshold_mar].F.unique()
)
year_trig_mar = comb_list_string(
    [
        str(y)
        for y in stats_mar.loc[
            stats_mar["max_cell"] >= threshold_mar
        ].F.dt.year.unique()
    ]
)
adm_trig_mar = comb_list_string(
    [
        str(y)
        for y in stats_mar.loc[
            stats_mar["max_cell"] >= threshold_mar
        ].admin.unique()
    ]
)

num_trig_jul = len(
    stats_jul.loc[stats_jul["max_cell"] >= threshold_jul].F.unique()
)
year_trig_jul = comb_list_string(
    [
        str(y)
        for y in stats_jul.loc[
            stats_jul["max_cell"] >= threshold_jul
        ].F.dt.year.unique()
    ]
)
adm_trig_jul = comb_list_string(
    [
        str(y)
        for y in stats_jul.loc[
            stats_jul["max_cell"] >= threshold_jul
        ].admin.unique()
    ]
)
```

```python
glue("num_trig_mar", num_trig_mar)
glue("num_trig_jul", num_trig_jul)
glue("year_trig_mar", year_trig_mar)
glue("year_trig_jul", year_trig_jul)
glue("adm_trig_mar", adm_trig_mar)
glue("adm_trig_jul", adm_trig_jul)
glue("threshold_mar", threshold_mar)
glue("threshold_jul", threshold_jul)
```

More specifically we are interested in March and July, with a leadtime of 3 and 1 month respectively. 

This would mean that if we would take the max cell as aggregation method, the threshold of {glue:text}`threshold_mar` for March would have been reached {glue:text}`num_trig_mar` times {glue:text}`year_trig_mar` for {glue:text}`adm_trig_mar`. 
For July the threshold of {glue:text}`threshold_jul` would have been reached {glue:text}`num_trig_jul` times{glue:text}`year_trig_jul`."


### Methods of aggregation
Note: all these computations only cover the 4 admin1's of interest

```python
num_trig_mar_mean = len(
    stats_mar.loc[stats_mar["mean_cell"] >= threshold_mar].F.unique()
)
year_trig_mar_mean = comb_list_string(
    [
        str(y)
        for y in stats_mar.loc[
            stats_mar["mean_cell"] >= threshold_mar
        ].F.dt.year.unique()
    ]
)
num_trig_mar_perc10 = len(
    stats_mar.loc[stats_mar["10quant_cell"] >= threshold_mar].F.unique()
)
year_trig_mar_perc10 = comb_list_string(
    [
        str(y)
        for y in stats_mar.loc[
            stats_mar["10quant_cell"] >= threshold_mar
        ].F.dt.year.unique()
    ]
)
adm_trig_mar_perc10 = comb_list_string(
    [
        str(y)
        for y in stats_mar.loc[
            stats_mar["10quant_cell"] >= threshold_mar
        ].admin.unique()
    ]
)
max_perc40_mar = stats_mar["40percth_cell"].max()
glue("num_trig_mar_mean", num_trig_mar_mean)
glue("year_trig_mar_mean", year_trig_mar_mean)
glue("num_trig_mar_perc10", num_trig_mar_perc10)
glue("year_trig_mar_perc10", year_trig_mar_perc10)
glue("adm_trig_mar_perc10", adm_trig_mar_perc10)
glue("max_perc40_mar", max_perc40_mar)
```

While taking the max cell is the most extreme method of aggregation, we have many other possiblities. Such as looking at the mean, or at a percentage of cells. 
<!-- For the July forecast we wouldn't trigger with any method of aggregation, since we already didn't trigger with the max methodology.  -->

For March, when using the mean method aggregation, the trigger would have been met {glue:text}`num_trig_mar_mean` times{glue:text}`year_trig_mar_mean`.

Below the distribution of the percentage of the area with >=40% probability is shown for March. 
We look at the distribution of the percentage of the area with >=40% probability of below avg rainfall for the admins of interest, across all forecasts with a leadtime of {glue:text}`leadtime`. 
When requiring 10% of cells to be above 40% this would be met {glue:text}`num_trig_mar_perc10` times{glue:text}`year_trig_mar_perc10`{glue:text}`adm_trig_mar_perc10`.

```python
# plot distribution for forecasts with C=0 (=below average) and L=1, for March
g = sns.displot(stats_mar["40percth_cell"], color="#007CE0", binwidth=1)
```

The plot below shows the occurences across all months and all leadtimes where at least 1% of the cells had a probability of at least 40% for below average rainfall. We can see that the larger percentages of the admin area with >=40% below average rainfall occur during other months and/or leadtimes

```python
# plot distribution for forecasts with C=0 (=below average) and L=1, for all months
g = sns.displot(
    stats_per_adm_bavg.loc[
        stats_per_adm_bavg["40percth_cell"] >= 1, "40percth_cell"
    ],
    color="#007CE0",
    binwidth=3,
)
```

### Differences in admin, leadtime and publication date


Below we examine the 10 percentile boundary across admins, publication months, and leadtimes. We can see small differences. However, it is really hard to quantify these differences because of the limited data. It seems some conditions give a bit more probability of the 10 percentile boundary being >=40% but this is hard to judge properly.

```python
# plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig, ax = plt.subplots(figsize=(10, 5))
g = sns.boxplot(
    data=stats_per_adm_bavg,
    x="admin",
    y="10quant_cell",
    ax=ax,
    color="#007CE0",
)
ax.set_ylabel("Probability")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Admin");
```

```python
# plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig, ax = plt.subplots(figsize=(15, 5))
g = sns.boxplot(
    data=stats_per_adm_bavg,
    x="month",
    y="10quant_cell",
    ax=ax,
    color="#007CE0",
)
ax.set_ylabel("Probability")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Publication month");
```

```python
# plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig, ax = plt.subplots(figsize=(10, 5))
g = sns.boxplot(
    data=stats_per_adm_bavg[stats_per_adm_bavg["40percth_cell"] >= 1],
    x="L",
    y="10quant_cell",
    ax=ax,
    color="#007CE0",
)
ax.set_ylabel("Probability")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Leadtime");
```

### Examine dominant tercile pixel
TODO: elaborate this section
Besides setting a threshold on the below average tercile, we also want to be sure that the below average tercile is the dominant tercile. We therefore require, at the pixel level that 
probability below average >= (probability above average + 5%)


Moreover on the above analysis, we require at least 10% of the area meeting the threshold. This results in the following activations for our periods of interest

```python
stats_mar[stats_mar["40th_bavg_cell"] >= 10]
```

```python
stats_jul[stats_jul["40th_bavg_cell"] >= 10]
```

<!-- While we can include the spatial severity in the trigger threshold, we should also take into account that the spatial uncertainty of seasonal forecasts is large. 

Given the size of the area of interest, it might therefore be better to only focus on whether any cell within that region reached the probability threshold. However, in this case 40% might be too sensitive of a trigger -->
