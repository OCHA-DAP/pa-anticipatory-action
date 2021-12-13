This notebook explores the WRSI trigger related to Chad. Given that the WRSI is not a forecast based method (although the extended version does include historical precipitation averages), I have been unable to find any evaluations of its accuracy or skill. We will thus dive straight into understanding how the historical patterns look in Chad and what a reasonable threshold and trigger might look like.

```python
 #### Load libraries and set global constants

 %load_ext autoreload
 %autoreload 2

 import geopandas as gpd
 from shapely.geometry import mapping
 import pandas as pd
 import rioxarray
 import rioxarray.merge
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

 from src.indicators.drought.wrsi import load_wrsi, filter_wrsi, wrsi_percent_below, _processed_dir
 from src.indicators.drought.biomasse import load_biomasse_mean
 from src.utils_general.raster_manipulation import compute_raster_statistics

 hdx_blue="#007ce0"

 iso3="tcd"
 config=Config()
 parameters = config.parameters(iso3)
 country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3
 adm1_bound_path=country_data_processed_dir / config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"

 #### Set variables

_save_processed_path = os.path.join(path_mod, Path(config.DATA_DIR), config.PUBLIC_DIR, config.PROCESSED_DIR, iso3, "wrsi")
```

Let's load the WRSI data and then plot just the end of season WRSI, to first inspect how it's behaving across the years. We will also subset to the relevant administrative areas. For now, we will just look at croplands since that covers the dominant portion of our area of interest.

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_reg=gdf_adm1[gdf_adm1.area_of_interest == True]

da_crop = load_wrsi("cropland", "current")
da_crop_clip = da_crop.rio.clip(gdf_reg["geometry"], all_touched=True)

 #not very neat function but does the job for now
def plt_wrsi(da, subtitle, plt_levels):
    g = da.plot(
        col="time",
        col_wrap=7,
        levels=plt_levels,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            'ticks': plt_levels,
        },
        figsize=(21,10)
    )
    
    for ax in g.axes.flat:
        gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
        gdf_reg.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    g.fig.suptitle(f"WRSI published at {subtitle} \n The subtitles indicate the start date",y=1.1);

plt_levels=[0, 60, 100, 250]

plt_wrsi(filter_wrsi(da_crop_clip, 33), "end of dekad 33", plt_levels)

```

```python
plt_wrsi(filter_wrsi(da_crop_clip, 23), "end of dekad 23", plt_levels)
```

We can see that there's definitely some spatial patterns in the WRSI, stretching frmo generally satisfied in the southeast to most often not satisfied or even close to satisfied in the northwest. Does this mean we might want to consider only certain areas?

```python
da_range = load_wrsi("rangeland", "current")
da_range_clip = da_range.rio.clip(gdf_reg["geometry"], all_touched=True)

plt_wrsi(filter_wrsi(da_range_clip, 33), "end of dekad 33", plt_levels)
```

Lots of missing data in the NW corner, where the start of season is not clearly occurring by the end of the aseason where the values are calculated. Need to be very careful here, because those areas are essentially always in severe deficit. Should we exclude these from our percentiles and calculations?

```python
plt_wrsi(da_crop_clip.where(da_crop_clip.time.dt.year == 2021, drop=True), "year 2021", plt_levels)
```

```python
plt_wrsi(da_crop_clip.where(da_crop_clip.time.dt.year == 2004, drop=True), "year 2004", plt_levels)
```

```python
plt_wrsi(da_range_clip.where(da_range_clip.time.dt.year == 2021, drop=True), "year 2021", plt_levels)
```

```python
plt_wrsi(da_range_clip.where(da_range_clip.time.dt.year == 2011, drop=True), "year 2011", plt_levels)
```

```python
da_crop_anom = load_wrsi("cropland", "anomaly")
da_crop_anom_clip = da_crop_anom.rio.clip(gdf_reg["geometry"], all_touched=True)

plt_wrsi(filter_wrsi(da_crop_anom_clip, 33), "end of dekad 33", plt_levels)
```

```python
da_range_anom = load_wrsi("rangeland", "anomaly")
da_range_anom_clip = da_range_anom.rio.clip(gdf_reg["geometry"], all_touched=True)

plt_wrsi(filter_wrsi(da_range_anom_clip, 33), "end of dekad 33", plt_levels)
```

Let's create a mask based on Biomasse. This will allow us to get rid of areas with no biomasse production and not owrry about the northwest areas that receive extremely limited rainfall.

```python
bm_mask = rioxarray.open_rasterio(os.path.join(
    os.getenv("AA_DATA_DIR"),
    "public",
    "raw",
    "general",
    "biomasse",
    "BiomassValueMean.tif"
))

bm_mask = bm_mask.rio.reproject_match(da_range_anom)
```

```python
ta = da_range_anom_clip.where(da_range_anom_clip.time.dt.year == 2021, drop = True)
ta = ta.where(ta.time.dt.month == 11, drop = True)
ta = ta.where(ta.time.dt.day == 1, drop = True)
ta.plot()
```

```python
da_range_anom_mask = da_range_anom_clip.where(bm_mask > 0)
ta = da_range_anom_mask.where(da_range_anom_mask.time.dt.year == 2021, drop = True)
ta = ta.where(ta.time.dt.month == 11, drop = True)
ta = ta.where(ta.time.dt.day == 1, drop = True)
ta.plot()
```

Great, so we can see above that we get a pretty good cutoff. However, we still have some areas in yellow that are highlighting areas that even by November, the rainy season hadn't started in this year. Let's try to slightly increase the mask.

```python
da_range_anom_mask = da_range_anom_clip.where(bm_mask > 100)
ta = da_range_anom_mask.where(da_range_anom_mask.time.dt.year == 2021, drop = True)
ta = ta.where(ta.time.dt.month == 11, drop = True)
ta = ta.where(ta.time.dt.day == 1, drop = True)
ta.plot()
```

Now with the mask we can see that we're getting very few areas in this dry region, and it might be appropriate to use for our WRSI analysis of rangelands. If we look at croplands:

```python
da_crop_anom_mask = da_crop_anom_clip.where(bm_mask > 100)
ta = da_crop_anom_mask.where(da_crop_anom_mask.time.dt.year == 2021, drop = True)
ta = ta.where(ta.time.dt.month == 11, drop = True)
ta = ta.where(ta.time.dt.day == 1, drop = True)
ta.plot()
```

Here we see almost no reduction in the coverage as there's limited arid regions covering this area. Let's now work on calculating the anomalies for various thresholds in both croplands and rangelands.

```python
d = wrsi_percent_below(da_range_anom_mask, [70, 80,90,100,110,120])
d.to_csv(os.path.join(_save_processed_path, "tcd_wrsi_anomaly_rangeland_thresholds.csv"))
```

Let's do the same for croplands anomaly.

```python
d = wrsi_percent_below(da_crop_anom_mask, [70, 80,90,100,110,120])
d.to_csv(os.path.join(_save_processed_path, "tcd_wrsi_anomaly_cropland_thresholds.csv"))
```

And for the current values.

```python
# cropland
da_crop_clip_mask = da_crop_clip.where(bm_mask > 100)
d = wrsi_percent_below(da_crop_clip_mask, [50, 60, 70, 80,90,100,110,120])
d.to_csv(os.path.join(_save_processed_path, "tcd_wrsi_current_cropland_thresholds.csv"))

# rangeland
da_range_clip_mask = da_range_clip.where(bm_mask > 100)
d = wrsi_percent_below(da_range_clip_mask, [50, 60, 70, 80,90,100,110,120])
d.to_csv(os.path.join(_save_processed_path, "tcd_wrsi_current_rangeland_thresholds.csv"))
```
