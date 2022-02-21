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
chirps_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "processed" / iso3 / "chirps" / "daily"
```

```python
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
#admin2's of interest for now
#all located in Unity and Jonglei
adm2_list=['Panyijiar', 'Leer', 'Mayendit', 'Koch', 'Guit',
           'Fangak', 'Ayod', 'Duk', 'Twic East', 'Bor South',
          'Yirol East','Awerial'
          ]
```

```python
gdf_reg=gdf_adm2[gdf_adm2.ADM2_EN.isin(adm2_list)]
adm0_col="ADM0_EN"
pcode0_col="ADM0_PCODE"
```

```python
#load the country data
ds_country = load_chirps_daily_clipped(iso3,config,resolution)
```

```python
#check how many cells are included in the region
ds_country.rio.clip(gdf_reg.geometry)
```

```python
#compute stats over the region of interest
gdf_reg_dissolved=gdf_reg.dissolve(by=adm0_col)
gdf_reg_dissolved=gdf_reg_dissolved[[pcode0_col,"geometry"]]

df_chirps_reg=compute_raster_statistics(
        gdf=gdf_reg_dissolved,
        bound_col=pcode0_col,
        raster_array=ds_country.precip,
        lon_coord="longitude",
        lat_coord="latitude",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=False,
    )
df_chirps_reg['year']=df_chirps_reg.time.dt.year
```

```python
#adm0 is misleading cause it is only the region, so remove that suffix
df_chirps_reg.columns = df_chirps_reg.columns.str.replace(r'_ADM0_PCODE$', '')
```

```python
# df_chirps_reg.to_csv(chirps_processed_dir/f'{iso3}_chirps_roi_stats_p{resolution}.csv',index=False)
```
