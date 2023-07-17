# IRI forecast for 2023 monitoring in Burkina Faso

This notebook explores the option of using IRI's seasonal forecast as the indicator for a drought-related trigger in Burkina Faso. 
From the country team the proposed trigger is:
- Trigger #1 in March covering June-July-August. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%. 
- Targeted Admin1s: Boucle de Mounhoun, Centre Nord, Sahel, Nord.

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
import re
import matplotlib
import calendar
import hvplot.xarray
from rasterio.enums import Resampling

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import (
    get_iri_data,
    get_iri_data_dominant,
)
from src.utils_general.raster_manipulation import (
    invert_latlon,
    change_longitude_range,
    fix_calendar,
)
```

```python
hdx_blue = "#007ce0"
```

```python
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

## Inspect forecasts

```python
adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]
adm_sel_str = re.sub(r"[ -]", "", "".join(adm_sel)).lower()
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
    iri_exploration_dir,
    f"{country}_iri_seasonal_forecast_stats_{''.join(adm_sel_str)}.csv",
)

adm1_bound_path = os.path.join(
    country_data_raw_dir, config.SHAPEFILE_DIR, parameters["path_admin1_shp"]
)
adm2_bound_path = os.path.join(
    country_data_raw_dir, config.SHAPEFILE_DIR, parameters["path_admin2_shp"]
)
```

```python
iri_ds = get_iri_data(config, download=True)
da_iri = iri_ds.prob
```

these are the variables of the forecast data, where C indicates the tercile (below-average, normal, or above-average).  
F indicates the publication month, and L the leadtime

```python
gdf_adm1 = gpd.read_file(adm1_bound_path)
iri_clip = iri_ds.rio.write_crs("EPSG:4326").rio.clip(
    gdf_adm1.geometry.apply(mapping), iri_ds.rio.crs, all_touched=True
)
```

```python
gdf_reg = gdf_adm1[gdf_adm1.ADM1_FR.isin(adm_sel)]
```

```python
fig, ax = plt.subplots()
gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
ax.axis("off");
```

Below the raw forecast data of below-average rainfall with {glue:text}`leadtime_mar` month leadtime, published in March is shown. The red areas are the 4 admin1's we are focussing on

The negative values indicate below average rainfall, and the positive values above average.

```python
# F indicates the publication month, and L the leadtime.
# A leadtime of 1 means a forecast published in May is forecasting JJA
ds_iri_dom = get_iri_data_dominant(config, download=True)
ds_iri_dom = ds_iri_dom.rio.write_crs("EPSG:4326", inplace=True)
da_iri_dom = ds_iri_dom.dominant
da_iri_dom_clip = da_iri_dom.rio.clip(gdf_adm1["geometry"], all_touched=True)
```

This is similair to [the figure on the IRI Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html), except that the bins are defined slightly differently

```python
# not very neat function but does the job for now
def plt_raster_iri(
    da_iri_dom_clip,
    pub_mon,
    lt,
    plt_levels,
    plt_colors,
    show_cbar=True,
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
            add_colorbar=show_cbar,
            figsize=(25, 7),
        )
    )
    for ax in g.axes.flat:
        gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
        gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    g.fig.suptitle(
        f"Forecasts published in {calendar.month_abbr[pub_mon]} predicting {for_seas} (lt={lt}) \n The subtitles indicate the publishing date",
        y=1.1,
    );
```

```python
# iri website bins
# plt_levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
plt_levels = [-100, -70, -60, -50, -45, -40, 40, 45, 50, 60, 70, 100]
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
iri_clip_diff = iri_clip.sel(C=0) - iri_clip.sel(C=2)
```

```python
plt_colors_rev = [
    "#783200",
    "#ab461e",
    "#d18132",
    "#e8b832",
    "#fafa02",
    "#fff7bc",
    "#ffffff",
    "#f7fcf0",
    "#d1f8cc",
    "#acf8a0",
    "#73bb6e",
    "#3a82b3",
    "#0e3bf4",
]
plt_colors_rev.reverse()
```

```python
iri_trig = iri_clip.sel(C=0).where(
    (iri_clip.sel(C=0).prob >= 40)
    & (iri_clip.sel(C=0) >= iri_clip.sel(C=2) + 5)
)
```

The same figure, but for the forecasts published in July with a {glue:text}`leadtime_jul` month leadtime are shown below


Some forecasts where we see a combination of below and above average are shown below. This is to guide the discussion on for which forecasts we would have wanted to trigger and for which we wouldn't

```python
iri_clip_reg = iri_clip.rio.clip(gdf_reg["geometry"])
print(
    f"centre within region: {iri_clip_reg.sel(C=0,F='2017-03-16',L=1).squeeze().prob.count().values}"
)
iri_clip_reg_allt = iri_clip.rio.clip(gdf_reg["geometry"], all_touched=True)
print(
    f"touching region: {iri_clip_reg_allt.sel(C=0,F='2017-03-16',L=1).squeeze().prob.count().values}"
)
```

```python
# we use this method instead of the rioxarray method as this allows multidimensional arrays
# however in the future (and for monitoring) we probably want a standard function for this, which replaces this one
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

    return dsi
```

```python
iri_clip_interp = interpolate_ds(iri_clip, iri_clip.rio.transform(), 20)
# recompute the crs else things go wrong
iri_clip_interp.rio.transform(recalc=True)
```

```python
df_bound = gpd.read_file(adm1_bound_path)
```

```python
g = iri_clip_reg.sel(C=0, F="2023-03-16", L=3).squeeze().prob.plot()
df_bound.boundary.plot(ax=g.axes, color="grey");
```

```python
g = iri_clip_reg.sel(C=0, F="2023-07-16", L=1).squeeze().prob.plot()
df_bound.boundary.plot(ax=g.axes, color="grey");
```

we select the region of interest, shown below

```python
iri_interp_reg = iri_clip_interp.rio.clip(gdf_reg["geometry"])
```

```python
# upsample the resolution in order to create a mask of our aoi
resolution = 0.01
mask_list = []
for terc in iri_interp_reg.C.values:
    for lt in iri_interp_reg.L.values:
        da_terc_lt = iri_interp_reg.sel(C=terc, L=lt)
        da_terc_lt_mask = da_terc_lt.rio.reproject(
            da_terc_lt.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        mask_list.append(da_terc_lt_mask.expand_dims({"C": [terc], "L": [lt]}))
da_iri_mask = (
    xr.combine_by_coords(mask_list)
    .rio.clip(gdf_reg["geometry"], all_touched=False)
    .prob
)
# reproject changes longitude and latitude name to x and y
# so change back here
da_iri_mask = da_iri_mask.rename({"x": "longitude", "y": "latitude"})
da_iri_mask_bavg = da_iri_mask.sel(C=0)
```

```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2023-03-16", L=3, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
```

```python
# check that masking is done correctly
g = da_iri_mask.sel(F="2023-07-16", L=1, C=0).plot()  # squeeze().plot()
gdf_adm1.boundary.plot(ax=g.axes)
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
stats_region = compute_zonal_stats_xarray(iri_clip_interp, gdf_reg)
stats_region["F"] = pd.to_datetime(
    stats_region["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
)
stats_region["month"] = stats_region.F.dt.month
```

```python
stats_region_bavg = stats_region[(stats_region.C == 0)]
```

And compute the statistics over this region, see a subset below

```python
stats_region[
    (stats_region.C == 0)
    & (stats_region.L == leadtime_mar)
    & (stats_region.F.dt.month == 3)
]
```

```python
stats_region[
    (stats_region.C == 0)
    & (stats_region.L == leadtime_jul)
    & (stats_region.F.dt.month == 7)
]
```

## Analyze statistics probability below average


Below the distribution of probability values is shown per month. \
This only includes the values for the below-average tercile, with a leadtime of {glue:text}`leadtime`. \
It should be noted that since we only have data from Mar 2017, these distributions contain maximum 5 values. \
From the distribution, it can be seen that a probability of 50% has never been reached since Mar 2017.

```python
stats_mar = stats_region_bavg.loc[
    (stats_region_bavg.F.dt.month == 3) & (stats_region_bavg.L == leadtime_mar)
]
stats_jul = stats_region_bavg.loc[
    (stats_region_bavg.F.dt.month == 7) & (stats_region_bavg.L == leadtime_jul)
]
```

```python
def comb_list_string(str_list):
    if len(str_list) > 0:
        return " in " + ", ".join(str_list)
    else:
        return ""


max_prob_mar = stats_mar.max_cell.max()
num_trig_mar = len(stats_mar.loc[stats_mar["max_cell"] >= threshold_mar])
year_trig_mar = comb_list_string(
    [
        str(y)
        for y in stats_mar.loc[
            stats_mar["max_cell"] >= threshold_mar
        ].F.dt.year.unique()
    ]
)

num_trig_jul = len(stats_jul.loc[stats_jul["max_cell"] >= threshold_jul])
year_trig_jul = comb_list_string(
    [
        str(y)
        for y in stats_jul.loc[
            stats_jul["max_cell"] >= threshold_jul
        ].F.dt.year.unique()
    ]
)
max_prob_jul = stats_jul.max_cell.max()
```

```python
iri_interp_reg_bavg_th = iri_interp_reg.where(
    iri_interp_reg.sel(C=0).prob >= 40
)
```

```python
stats_country = compute_zonal_stats_xarray(iri_clip_interp, gdf_adm1)
stats_country["F"] = pd.to_datetime(
    stats_country["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
)
stats_country["month"] = stats_country.F.dt.month
glue(
    "max_prob_mar_country",
    stats_country.loc[
        (stats_country.C == 0)
        & (stats_country.L == leadtime_mar)
        & (stats_country.F.dt.month == 3),
        "max_cell",
    ].max(),
)
glue(
    "max_prob_jul_country",
    stats_country.loc[
        (stats_country.C == 0)
        & (stats_country.L == leadtime_jul)
        & (stats_country.F.dt.month == 7),
        "max_cell",
    ].max(),
)
```

To check if these below 50% and below 40% probabilities depend on the part of the country, we also compute the maximum values in the whole country across all years. 
<!-- While the values can be slightly higher in other regions, the 50% threshold is never reached.  -->


The maximum value for the March forecast in the whole country was {glue:text}`max_prob_mar_country:.2f`%. \
<!-- For July this was {glue:text}`max_prob_jul_country:.2f`%" -->

```python
perc_for_40th = (
    stats_country.loc[
        (stats_country.C == 0) & (stats_country.L == leadtime_jul), "max_cell"
    ]
    .ge(40)
    .value_counts(True)[True]
    * 100
)
glue("perc_for_maxcell_40th", perc_for_40th)
```

### Examine dominant tercile and 40% threshold
Besides setting a threshold on the below average tercile, we also want to be sure that the below average tercile is the dominant tercile. We therefore require, at the pixel level that 
probability below average >= (probability above average + 5%)
Here we check how often this occurs, for the months of interest and across all months


Moreover on the above analysis, we require at least 10% of the area meeting the threshold. This results in the following activations for our periods of interest

```python
stats_mar[stats_mar["40th_bavg_cell"] >= 10]
```

```python
stats_jul[stats_jul["40th_bavg_cell"] >= 10]
```

```python
stats_mar
```

```python
stats_jul
```

```python
# across all months with leadtime_mar
stats_region_bavg_ltmar = stats_region_bavg.loc[
    (stats_region_bavg.L == leadtime_mar)
]
stats_region_bavg_ltmar[stats_region_bavg_ltmar["40th_bavg_cell"] >= 10]
```

```python
# across all months with leadtime_jul
stats_region_bavg_ltjul = stats_region_bavg.loc[
    (stats_region_bavg.L == leadtime_jul)
]
stats_region_bavg_ltjul[stats_region_bavg_ltjul["40th_bavg_cell"] >= 10]
```

```python
iri_interp_reg_trig = iri_interp_reg.sel(C=0).where(
    (iri_interp_reg.sel(C=0).prob >= 40)
    & (iri_interp_reg.sel(C=0) >= iri_interp_reg.sel(C=2) + 5)
)
```

```python
diff_threshold = 5
```

```python
stats_region_aavg = stats_region[stats_region.C == 2]
stats_region_aavg_ltmar = stats_region_aavg[
    stats_region_aavg.L == leadtime_mar
]
```

```python
stats_region_merged = stats_region_bavg.merge(
    stats_region_aavg, on=["F", "L"], suffixes=("_bavg", "_aavg")
)
```

```python
stats_region_merged["diff_bel_abv"] = (
    stats_region_merged["10quant_cell_bavg"]
    - stats_region_merged["10quant_cell_aavg"]
)
```

```python
stats_region_merged_ltmar = stats_region_merged[
    stats_region_merged.L == leadtime_mar
]
```

```python
stats_region_merged_ltmar[
    stats_region_merged_ltmar["diff_bel_abv"] >= diff_threshold
]
```

```python
stats_region_merged[stats_region_merged["F"] == "2023-03-16"]
```

```python
stats_region_merged[stats_region_merged["F"] == "2023-07-16"]
```
