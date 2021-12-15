```python
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

from src.indicators.drought.ndvi import process_ndvi

hdx_blue="#007ce0"

iso3="tcd"
config=Config()
parameters = config.parameters(iso3)
country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3
adm1_bound_path=country_data_processed_dir / config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"

 #### Set variables

_save_processed_path = os.path.join(path_mod, Path(config.DATA_DIR), config.PUBLIC_DIR, config.PROCESSED_DIR, iso3, "wrsi")
```

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_reg=gdf_adm1[gdf_adm1.area_of_interest == True]

da = process_ndvi("tcd", gdf_reg.geometry, [50, 60, 70, 80, 90, 100])

```

```python
df1 = rioxarray.open_rasterio(os.path.join(config.DATA_DIR, "public", "raw", "glb", "ndvi", "wa0333pct.tif"))
df2 = rioxarray.open_rasterio(os.path.join(config.DATA_DIR, "public", "raw", "glb", "ndvi", "wa1313pct.tif"))
```

```python
df1 = df1.rio.clip(gdf_reg.geometry, drop=True, from_disk=True)
df2 = df2.rio.clip(gdf_reg.geometry, drop=True, from_disk=True)
dfs = [df1, df2]

```

```python
thresholds = [50, 60, 70, 80, 90, 100]
an = [xr.where(df1 <= threshold, 1, 0).mean(dim=["x","y"]).to_dataframe(name="ndvi").reset_index(drop=True) for threshold in thresholds]     

pd.DataFrame({
    "thresholds":[50, 60, 70, 80, 90, 100],
    "analysis":pd.concat(an).ndvi * 100,
    "time":1
})
```

```python
xr.where(df1 <= 100, 1, 0).mean(dim=["x","y"]).values[0]
```

```python
da
```

```python

```
