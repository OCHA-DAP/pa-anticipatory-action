### Compute CHIRPS stats
Compute the statistics over the region of interest. Currently only works for the .25 resolution data. 

Takes long to compute with current script (needs to be optimized at some point). So advised to only compute for historical analyses. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
import sys
from pathlib import Path

import geopandas as gpd
import xarray as xr
import pandas as pd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.chirps_rainfallobservations import (clip_chirps_daily, _get_raw_path_daily, _get_processed_path_country_daily,
                                                               load_chirps_daily_clipped)
from src.indicators.drought.config import Config
from src.utils_general.raster_manipulation import compute_raster_statistics

config = Config()
```

```python
iso3="ssd"
resolution="25"
```

```python
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
adm2_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin2_shp"]
country_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "processed" / iso3
chirps_processed_dir = country_processed_dir / "chirps" / "daily"
bentiu_bound_path = country_processed_dir / "bentiu" / "bentiu_bounding_box.gpkg"
```

```python
gdf_bentiu=gpd.read_file(bentiu_bound_path)
```

```python
#load the country data
ds_country = load_chirps_daily_clipped(iso3,config,resolution)
```

```python
#check how many cells are included in the region
ds_country.rio.clip(gdf_bentiu.geometry, all_touched = True)
```

```python
df_chirps_reg=compute_raster_statistics(
        gdf=gdf_bentiu,
        bound_col="id",
        raster_array=ds_country.precip,
        lon_coord="longitude",
        lat_coord="latitude",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_chirps_reg['year']=df_chirps_reg.time.dt.year
```

```python
# df_chirps_reg.to_csv(chirps_processed_dir/f'{iso3}_chirps_bentiu_stats_p{resolution}.csv',index=False)
```

```python

```
