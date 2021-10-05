```python
%load_ext autoreload
%autoreload 2
```

```python
from shapely.geometry import mapping
import os
from pathlib import Path
import sys
import rioxarray
import geopandas as gpd
import pandas as pd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import ARC2
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
arc2_test = ARC2(
    country_iso3 = "mwi",
    date_min = "2020-01-01",
    date_max = "2020-01-03",
    range_x = ("32E", "33E"),
    range_y = ("20S", "19S")
)

arc2_test.download_data(master=True)

ds = arc2_test.load_raw_data()
ds

rioxarray.open_rasterio(ds)
```

```python
ds=rioxarray.open_rasterio(os.path.join(os.getenv("AA_DATA_DIR"), "public", "exploration", "mwi", "arc2", "arc2_2021_approxmwi.nc"),masked=False).squeeze()
#clip to MWI
df_bound=gpd.read_file(os.path.join(os.getenv("AA_DATA_DIR"), "public", "raw", "mwi", "cod_ab", "mwi_adm_nso_20181016_shp", "mwi_admbnda_adm2_nso_20181016.shp"))
ds=ds.rio.write_crs("EPSG:4326")
df_bound
```

```python
compute_raster_statistics(df_bound, "ADM2_EN", ds_clip, stats_list = ["mean", "sum"], all_touched = False)
```
