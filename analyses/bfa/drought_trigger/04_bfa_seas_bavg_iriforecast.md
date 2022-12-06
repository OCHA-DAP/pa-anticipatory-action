# IRI forecast as a trigger for drought in Burkina Faso

**Note: this notebook has an improved version in the new BFA repo [here](https://github.com/OCHA-DAP/pa-aa-bfa-drought/tree/main/analysis). So leaving it for refernce as the other notebooks haven't been ported over to the new repo, but in principle that new repo should be used as much as possible**

This notebook explores the option of using IRI's seasonal forecast as the indicator for a drought-related trigger in Burkina Faso. 
From the country team the proposed trigger is:
- Trigger #1 in March covering June-July-August. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%. 
- Targeted Admin1s: Boucle de Mounhoun, Centre Nord, Sahel, Nord.

This notebook explores if and when these triggers would be reached. Moreover, an exploration is done on how the raster data can be combined to come to one value for all 4 admin1s.

<!-- - Trigger #1 in March covering Apr-May-June. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%.  -->

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
iri_ds = get_iri_data(config, download=False)
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
ds_iri_dom = get_iri_data_dominant(config, download=False)
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
plt_raster_iri(
    da_iri_dom_clip,
    pub_mon=3,
    lt=3,
    plt_levels=plt_levels,
    plt_colors=plt_colors,
)
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

```python
plt_raster_iri(
    iri_trig.prob,
    pub_mon=3,
    lt=3,
    plt_levels=[1, 0],
    plt_colors=[hdx_blue],
    show_cbar=False,
)
```

```python
plt_raster_iri(
    iri_clip_diff.prob,
    pub_mon=3,
    lt=3,
    plt_levels=[-40, -20, -15, -10, -7, -5, -2, 2, 5, 10, 20, 40],
    plt_colors=plt_colors_rev,
)
```

```python
plt_raster_iri(
    iri_clip_diff.prob,
    pub_mon=3,
    lt=3,
    plt_levels=[-40, -20, -15, -10, -7, -5, -2, 2, 5, 10, 20, 40],
    plt_colors=plt_colors_rev,
)
```

```python
plt_raster_iri(
    iri_clip.sel(C=0).prob,
    pub_mon=3,
    lt=3,
    plt_levels=[0, 30, 35, 40, 45, 50, 60, 70, 100],
    plt_colors=[
        "#ffffff",
        "#DDC0A6",
        "#DB9D94",
        "#fafa02",
        "#e8b832",
        "#d18132",
        "#ab461e",
        "#783200",
    ],
)
```

```python
plt_raster_iri(
    iri_clip.sel(C=2).prob,
    pub_mon=3,
    lt=3,
    plt_levels=[0, 30, 35, 40, 45, 50, 60, 70, 100],
    plt_colors=[
        "#ffffff",
        "#9e9ac8",
        "#54278f",
        "#d1f8cc",
        "#acf8a0",
        "#73bb6e",
        "#3a82b3",
        "#0e3bf4",
    ],
)
```

```python
plt_raster_iri(
    da_iri_dom_clip,
    pub_mon=7,
    lt=1,
    plt_levels=plt_levels,
    plt_colors=plt_colors,
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
da_iri_dom_clip = (
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
    da_iri_dom_clip.where(da_iri_dom_clip.F.dt.month.isin([7]), drop=True)
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

Some forecasts where we see a combination of below and above average are shown below. This is to guide the discussion on for which forecasts we would have wanted to trigger and for which we wouldn't

```python
g = (
    da_iri_dom_clip.where(
        da_iri_dom_clip.F.isin(
            [
                cftime.Datetime360Day(2021, 2, 16, 0, 0, 0, 0),
                cftime.Datetime360Day(2017, 5, 16, 0, 0, 0, 0),
                cftime.Datetime360Day(2017, 4, 16, 0, 0, 0, 0),
            ]
        ),
        drop=True,
    )
    .sel(L=3)
    .dominant.plot(
        col="F",
        col_wrap=3,
        levels=levels,
        colors=colors,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "ticks": levels,
        },
        figsize=(20, 10),
    )
)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

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
g = iri_clip_reg.sel(C=0, F="2017-03-16", L=1).squeeze().prob.plot()
df_bound.boundary.plot(ax=g.axes, color="grey");
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
iri_clip_interp
```

```python
# check that interpolated values look fine
g = (
    iri_clip_interp.where(iri_clip_interp.F.dt.month.isin([7]), drop=True)
    .sel(L=1, C=0)
    .prob.plot(
        col="F",
        col_wrap=3,
        cmap=mpl.cm.YlOrRd,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
        },
        figsize=(10, 10),
    )
)
df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

we select the region of interest, shown below

```python
iri_interp_reg = iri_clip_interp.rio.clip(gdf_reg["geometry"])
```

```python
g = (
    iri_interp_reg.sel(C=0, F="2020-01-16", L=1)
    .squeeze()
    .prob.plot.imshow(
        cmap=matplotlib.colors.ListedColormap([hdx_blue]),
        figsize=(6, 10),
        add_colorbar=False,
    )
)
df_bound.boundary.plot(ax=g.axes, color="grey")
g.axes.set_title(f"Approximate mask")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python

```

```python
g=iri_interp_reg_trig.sel(F="2017-03",L=3).squeeze().prob.plot.imshow(cmap=matplotlib.colors.ListedColormap(["#f2645a"]),figsize=(6,10),add_colorbar=False)
df_bound.boundary.plot(linewidth=1, ax=g.axes, color="grey")
df_bound[~df_bound.ADM1_PCODE.isin(gdf_reg.ADM1_PCODE)].plot(linewidth=1, ax=g.axes, color="#cccccc")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
iri_reg_centre=iri_clip.rio.clip(gdf_reg["geometry"])
g=iri_reg_centre.sel(C=0,F="2020-01-16",L=1).squeeze().prob.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
df_bound.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"All cells centering the region")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
iri_allt_centre=iri_clip.rio.clip(gdf_reg["geometry"],all_touched=True)
g=iri_allt_centre.sel(C=0,F="2020-01-16",L=1).squeeze().prob.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
df_bound.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"All cells touching the region")
gdf_reg.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
def compute_zonal_stats_xarray(raster,shapefile,lon_coord="longitude",lat_coord="latitude",var_name="prob"):
    raster_clip=raster.rio.set_spatial_dims(x_dim=lon_coord,y_dim=lat_coord).rio.clip(shapefile.geometry.apply(mapping),raster.rio.crs,all_touched=False)
    grid_mean = raster_clip.mean(dim=[lon_coord,lat_coord]).rename({var_name: "mean_cell"})
    grid_min = raster_clip.min(dim=[lon_coord,lat_coord]).rename({var_name: "min_cell"})
    grid_max = raster_clip.max(dim=[lon_coord,lat_coord]).rename({var_name: "max_cell"})
    grid_std = raster_clip.std(dim=[lon_coord,lat_coord]).rename({var_name: "std_cell"})
    grid_quant90 = raster_clip.quantile(0.9,dim=[lon_coord,lat_coord]).rename({var_name: "10quant_cell"})
    grid_percth40 = raster_clip.where(raster_clip.prob >=40).count(dim=[lon_coord,lat_coord])/raster_clip.count(dim=[lon_coord,lat_coord])*100
    grid_percth40=grid_percth40.rename({var_name: "40percth_cell"})
    raster_diff_bel_abv=raster_clip.sel(C=0)-raster_clip.sel(C=2)
    grid_dom = raster_clip.sel(C=0).where((raster_clip.sel(C=0).prob >=40) & (raster_diff_bel_abv>=5)).count(dim=[lon_coord,lat_coord])/raster_clip.count(dim=[lon_coord,lat_coord])*100
    grid_dom = grid_dom.rename({var_name: "40th_bavg_cell"})
    zonal_stats_xr = xr.merge([grid_mean, grid_min, grid_max, grid_std,grid_quant90,grid_percth40,grid_dom]).drop("spatial_ref")
    zonal_stats_df=zonal_stats_xr.to_dataframe()
    zonal_stats_df=zonal_stats_df.reset_index()
    return zonal_stats_df
```

```python
stats_region=compute_zonal_stats_xarray(iri_clip_interp,gdf_reg)
stats_region["F"]=pd.to_datetime(stats_region["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
stats_region["month"]=stats_region.F.dt.month
```

```python
# stats_region.to_csv(stats_reg_path,index=False)
```

```python
len(stats_region[stats_region["40th_bavg_cell"]>=10])/len(stats_region)*100
```

```python
stats_region_bavg=stats_region[(stats_region.C==0)]
```

And compute the statistics over this region, see a subset below

```python
stats_region[(stats_region.C==0)&(stats_region.L==leadtime_mar)&(stats_region.F.dt.month==3)]
```

```python
stats_region[(stats_region.C==0)&(stats_region.L==leadtime_jul)&(stats_region.F.dt.month==7)]
```

## Analyze statistics probability below average


Below the distribution of probability values is shown per month. \
This only includes the values for the below-average tercile, with a leadtime of {glue:text}`leadtime`. \
It should be noted that since we only have data from Mar 2017, these distributions contain maximum 5 values. \
From the distribution, it can be seen that a probability of 50% has never been reached since Mar 2017.

```python
stats_mar=stats_region_bavg.loc[(stats_region_bavg.F.dt.month==3)&(stats_region_bavg.L==leadtime_mar)]
stats_jul=stats_region_bavg.loc[(stats_region_bavg.F.dt.month==7)&(stats_region_bavg.L==leadtime_jul)]
```

```python
def comb_list_string(str_list):
    if len(str_list)>0:
        return " in "+", ".join(str_list)
    else:
        return ""

max_prob_mar=stats_mar.max_cell.max()
num_trig_mar=len(stats_mar.loc[stats_mar['max_cell']>=threshold_mar])
year_trig_mar=comb_list_string([str(y) for y in stats_mar.loc[stats_mar['max_cell']>=threshold_mar].F.dt.year.unique()])

num_trig_jul=len(stats_jul.loc[stats_jul['max_cell']>=threshold_jul])
year_trig_jul=comb_list_string([str(y) for y in stats_jul.loc[stats_jul['max_cell']>=threshold_jul].F.dt.year.unique()])
max_prob_jul=stats_jul.max_cell.max()
```

```python
glue("max_prob_mar", max_prob_mar)
glue("max_prob_jul", max_prob_jul)
glue("num_trig_mar", num_trig_mar)
glue("num_trig_jul", num_trig_jul)
glue("year_trig_mar", year_trig_mar)
glue("year_trig_jul", year_trig_jul)
glue("threshold_mar", threshold_mar)
glue("threshold_jul", threshold_jul)
```

```python
iri_interp_reg.sel(C=0).hvplot.violin('prob', by='L', color='L', cmap='Category20').opts(ylabel="Probability below average")
```

```python
iri_interp_reg_diff.where(iri_interp_reg.sel(C=0).prob>=40).count()
```

```python
iri_interp_reg_bavg_th=iri_interp_reg.where(iri_interp_reg.sel(C=0).prob>=40)
```

```python
iri_interp_reg_bavg_th_diff=iri_interp_reg_bavg_th.sel(C=0)-iri_interp_reg_bavg_th.sel(C=2)
iri_interp_reg_bavg_th_diff.hvplot.violin('prob', by='L', color='L', cmap='Category20').opts(ylabel="%bavg - %abv avg")
```

```python
iri_interp_reg_diff=iri_interp_reg.sel(C=0)-iri_interp_reg.sel(C=2)
iri_interp_reg_diff.hvplot.violin('prob', by='L', color='L', cmap='Category20').opts(ylabel="Probability below average")
```

```python
iri_interp_reg_diff.sel(F="2020-03",L=1).prob.plot()
```

```python
iri_interp_reg.sel(C=0).hvplot.kde('prob', by='L', alpha=0.5)
```

```python
iri_interp_reg_diff.hvplot.kde('prob', by='L', alpha=0.5)
```

NOTE: the plots below only have 5 data points so in my opinion, looking back, really back plots that shouldn't be used

```python
#plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=stats_region_bavg[stats_region_bavg.L==3],x="month",y="max_cell",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 3 months")
ax.set_xlabel("Publication month");
```

```python
#plot distribution for forecasts with C=0 (=below average) for all months with leadtime = 3
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=stats_region_bavg[stats_region_bavg.L==1],x="month",y="max_cell",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 1 month")
ax.set_xlabel("Publication month");
```

More specifically we are interested in March and July, with a leadtime of 3 and 1 month respectively. 
The maximum values across all cells for the March forecasts has been {glue:text}`max_prob_mar:.2f`%, and for the July forecasts {glue:text}`max_prob_jul:.2f`% 
This would mean that if we would take the max cell as aggregation method, the threshold of {glue:text}`threshold_mar` for March would have been reached {glue:text}`num_trig_mar` times {glue:text}`year_trig_mar`. 
For July the threshold of {glue:text}`threshold_jul` would have been reached {glue:text}`num_trig_jul` times{glue:text}`year_trig_jul`."

```python
stats_country=compute_zonal_stats_xarray(iri_clip_interp,gdf_adm1)
stats_country["F"]=pd.to_datetime(stats_country["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
stats_country["month"]=stats_country.F.dt.month
glue("max_prob_mar_country",stats_country.loc[(stats_country.C==0)&(stats_country.L==leadtime_mar)&(stats_country.F.dt.month==3),'max_cell'].max())
glue("max_prob_jul_country",stats_country.loc[(stats_country.C==0)&(stats_country.L==leadtime_jul)&(stats_country.F.dt.month==7),'max_cell'].max())
```

To check if these below 50% and below 40% probabilities depend on the part of the country, we also compute the maximum values in the whole country across all years. 
<!-- While the values can be slightly higher in other regions, the 50% threshold is never reached.  -->


The maximum value for the March forecast in the whole country was {glue:text}`max_prob_mar_country:.2f`%. \
<!-- For July this was {glue:text}`max_prob_jul_country:.2f`%" -->

```python
perc_for_40th=stats_country.loc[(stats_country.C==0)&(stats_country.L==leadtime_jul),'max_cell'].ge(40).value_counts(True)[True]*100
glue("perc_for_maxcell_40th",perc_for_40th)
```

Across all months, {glue:text}`perc_for_maxcell_40th:.2f`% of the forecasts with 1 month leadtime had a >=40% probability of below average rainfall in at least one cell across the **whole** country

```python
#plot distribution for forecasts with C=0 (=below average), for all months
fig,ax=plt.subplots(figsize=(10,5))
g=sns.boxplot(data=stats_country[(stats_country.C==0)&(stats_country.L==1)],x="month",y="max_cell",ax=ax,color="#007CE0")
ax.set_ylabel("Probability")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Leadtime = 3 months")
ax.set_xlabel("Publication month")
```

### Methods of aggregation
Note: all these computations only cover the region of interest

```python
max_prob_mar=stats_mar['max_cell'].max()
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

```python
#plot distribution for forecasts with C=0 (=below average) and L=1, for March
g=sns.displot(stats_mar["40percth_cell"],color="#007CE0",binwidth=1)
```

The plot below shows the occurences across all months and all leadtimes where at least 1% of the cells had a probability of at least 40% for below average rainfall. We can see that the occurrence of this is pretty rare.

```python
#plot distribution for forecasts with C=0 (=below average) and L=1, for all months
g=sns.displot(stats_region_bavg.loc[stats_region_bavg["40percth_cell"]>=1,"40percth_cell"],color="#007CE0",binwidth=3)
```

<!-- While we can include the spatial severity in the trigger threshold, we should also take into account that the spatial uncertainty of seasonal forecasts is large. 

Given the size of the area of interest, it might therefore be better to only focus on whether any cell within that region reached the probability threshold. However, in this case 40% might be too sensitive of a trigger -->


### Examine dominant tercile and 40% threshold
Besides setting a threshold on the below average tercile, we also want to be sure that the below average tercile is the dominant tercile. We therefore require, at the pixel level that 
probability below average >= (probability above average + 5%)
Here we check how often this occurs, for the months of interest and across all months


Moreover on the above analysis, we require at least 10% of the area meeting the threshold. This results in the following activations for our periods of interest

```python
stats_mar[stats_mar["40th_bavg_cell"]>=10]
```

```python
stats_jul[stats_jul["40th_bavg_cell"]>=10]
```

```python
stats_mar
```

```python
stats_jul
```

```python
#across all months with leadtime_mar
stats_region_bavg_ltmar=stats_region_bavg.loc[(stats_region_bavg.L==leadtime_mar)]
stats_region_bavg_ltmar[stats_region_bavg_ltmar["40th_bavg_cell"]>=10]
```

```python
#percentage of forecasts that met requirement
len(stats_region_bavg_ltmar[stats_region_bavg_ltmar["40th_bavg_cell"]>=10])/len(stats_region_bavg_ltmar.F.unique())*100
```

```python
#across all months with leadtime_jul
stats_region_bavg_ltjul=stats_region_bavg.loc[(stats_region_bavg.L==leadtime_jul)]
stats_region_bavg_ltjul[stats_region_bavg_ltjul["40th_bavg_cell"]>=10]
```

```python
#percentage of forecasts that met requirement
len(stats_region_bavg_ltjul[stats_region_bavg_ltjul["40th_bavg_cell"]>=10])/len(stats_region_bavg_ltjul.F.unique())*100
```

```python
iri_interp_reg_trig
```

```python
iri_interp_reg_trig=iri_interp_reg.sel(C=0).where((iri_interp_reg.sel(C=0).prob>=40)&(iri_interp_reg.sel(C=0)>=iri_interp_reg.sel(C=2)+5))
```

### Examine ONLY dominant region
Understand how often it occurrs that the 10 percentile threshold is at least x% higher for below than above average

As can be seen this occurs the same number of times as when below average probability is at least 40%, but the dates don't fully overlap

```python
diff_threshold=5
```

```python
stats_region_aavg=stats_region[stats_region.C==2]
stats_region_aavg_ltmar=stats_region_aavg[stats_region_aavg.L==leadtime_mar]
```

```python
stats_region_merged=stats_region_bavg.merge(stats_region_aavg,on=["F","L"],suffixes=("_bavg","_aavg"))
```

```python
stats_region_merged["diff_bel_abv"]=stats_region_merged["10quant_cell_bavg"]-stats_region_merged["10quant_cell_aavg"]
```

```python
stats_region_merged_ltmar=stats_region_merged[stats_region_merged.L==leadtime_mar]
```

```python
stats_region_merged_ltmar[stats_region_merged_ltmar["diff_bel_abv"]>=diff_threshold]
```

```python
stats_region_merged_ltjul=stats_region_merged[stats_region_merged.L==leadtime_jul]
```

```python
stats_region_merged_ltjul[stats_region_merged_ltjul["diff_bel_abv"]>=diff_threshold]
```

### Examine ONLY dominant pixel
Understand how often it occurrs that at least 10% of the pixels have x% higher probability for below than above average

As can be seen this occurrs much more often

```python
def compute_zonal_stats_xarray_dominant(raster,shapefile,lon_coord="longitude",lat_coord="latitude",var_name="prob"):
    raster_clip=raster.rio.set_spatial_dims(x_dim=lon_coord,y_dim=lat_coord).rio.clip(shapefile.geometry.apply(mapping),raster.rio.crs,all_touched=False)
    raster_diff_bel_abv=raster_clip.sel(C=0)-raster_clip.sel(C=2)
    grid_quant90 = raster_diff_bel_abv.quantile(0.9,dim=[lon_coord,lat_coord]).rename({var_name: "10quant_cell"})
    zonal_stats_xr = xr.merge([grid_quant90])
    zonal_stats_df=zonal_stats_xr.to_dataframe()
    zonal_stats_df=zonal_stats_df.reset_index()
    return zonal_stats_df
```

```python
stats_dom=compute_zonal_stats_xarray_dominant(iri_clip_interp,gdf_reg)
stats_dom["F"]=pd.to_datetime(stats_dom["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
stats_dom["month"]=stats_dom.F.dt.month
```

```python
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_mar)])
```

```python
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_mar)])/len(stats_dom.F.unique())
```

```python
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_jul)])
```

```python
len(stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_jul)])/len(stats_dom.F.unique())
```

```python
stats_dom[(stats_dom["10quant_cell"]>=10)&(stats_dom.L==leadtime_mar)]
```

```python
stats_dom[(stats_dom["10quant_cell"]>=5)&(stats_dom.L==leadtime_mar)]
```
