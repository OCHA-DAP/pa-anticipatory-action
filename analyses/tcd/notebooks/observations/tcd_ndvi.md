```python
%load_ext autoreload
%autoreload 2

import geopandas as gpd
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.ndvi import download_ndvi, process_ndvi, load_ndvi

hdx_blue="#007ce0"

iso3="tcd"
config=Config()
parameters = config.parameters(iso3)
country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3
adm1_bound_path=country_data_processed_dir / config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"

 #### Set variables

_save_processed_path = os.path.join(path_mod, Path(config.DATA_DIR), config.PUBLIC_DIR, config.PROCESSED_DIR, iso3, "wrsi")
```

First, let's download our administrative boundaries we are interested in, in this case those for Chad. We will then process NDVI data for that area, calculating the percent of area below each anomaly threshold from 50% of the median to the median itself with steps of 10%. With that, we can then use this processed data for further analysis within R (`wrsi_exploration.R`).

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_reg=gdf_adm1[gdf_adm1.area_of_interest == True]

# download_ndvi()
process_ndvi(iso3, gdf_reg.geometry, [50, 60, 70, 80, 90, 100])
da = load_ndvi(iso3)
```

```python

```
