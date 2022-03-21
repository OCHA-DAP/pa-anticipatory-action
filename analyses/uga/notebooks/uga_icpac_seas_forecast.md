### ICPAC seasonal forecast
Exploration of the ICPAC seasonal forecast of above average rainfall in Uganda for the 2022 season. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
iso3="uga"
config=Config()

country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3

glb_private_raw_dir = Path(config.DATA_DIR) /config.PRIVATE_DIR / config.RAW_DIR / "glb"
icpac_raw_dir = glb_private_raw_dir / "icpac"
output_dir =  Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / "uga" / "icpac"
```

```python
parameters = config.parameters(iso3)
adm1_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin1_shp"]
adm2_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin2_shp"]
gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
icpac_filepath=icpac_raw_dir/"202202"/"PredictedProbabilityRain_Mar-May_Feb2022.nc"
```

```python
ds=xr.load_dataset(icpac_filepath)
```

```python
ds=ds.rio.set_spatial_dims("lon","lat",inplace=True).rio.write_crs("EPSG:4326",inplace=True)
```

```python
ds
```

```python
da_above=ds.above
```

```python
da_above.plot();
```

```python
g=da_above.rio.set_spatial_dims("lon","lat",inplace=True).rio.clip(gdf_adm1.geometry).plot()
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
plt.title("MAM Above Average Rainfall Forecast \n Issued by ICPAC Feb 2022")
# plt.savefig(output_dir / 'forecast_adm1_overlay_20220301.png')
```

```python
g=da_above.rio.set_spatial_dims("lon","lat",inplace=True).rio.clip(gdf_adm1.geometry).plot()
gdf_adm2.boundary.plot(ax=g.axes,color="grey");
```

```python
#aggregation method
#if False then take cells with centre within the adm
#if True take cells touching the adm
all_touched = True
```

```python
#% probability of above avg
threshold = 40
```

```python
pcode_col = "ADM1_PCODE"
```

```python
#compute stats
df_stats_aavg=compute_raster_statistics(
        gdf=gdf_adm1,
        bound_col=pcode_col,
        raster_array=da_above.rio.write_crs("EPSG:4326"),
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["min","mean","max","std","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=all_touched,
    )
da_above_thresh=da_above.where(da_above>=threshold)
df_stats_aavg_thresh=compute_raster_statistics(
        gdf=gdf_adm1,
        bound_col=pcode_col,
        raster_array=da_above_thresh,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["count"],
        all_touched=all_touched,
    )

df_stats_aavg["perc_thresh"] = (df_stats_aavg_thresh[f"count_{pcode_col}"]
                                /df_stats_aavg[f"count_{pcode_col}"]*100)
```

```python
gdf_stats_avg=gdf_adm1.merge(df_stats_aavg,on=pcode_col,how="right")
```

```python
gdf_stats_avg.plot("perc_thresh",legend=True);
```

```python
stats_summary = gdf_stats_avg[['ADM1_PCODE', 'ADM1_EN', 'min_ADM1_PCODE', 'mean_ADM1_PCODE', 'max_ADM1_PCODE', 'std_ADM1_PCODE', 'count_ADM1_PCODE', '80quant_ADM1_PCODE', 'perc_thresh']]
stats_summary
```

```python
data_output_filename = output_dir / "stats_summary_20220301.csv"
data_output_filename
# stats_summary.to_csv(data_output_filename, index=False)
```
