```python
#### Load libraries and set global constants

%load_ext autoreload
%autoreload 2

from pathlib import Path
import sys
import os

import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

iso3="tcd"
config=Config() 
parameters = config.parameters(iso3)
country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3
adm2_bound_path=country_data_processed_dir / config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"

 #### Set variables

_save_processed_path = os.path.join(path_mod, Path(config.DATA_DIR), config.PUBLIC_DIR, config.PROCESSED_DIR, iso3, "wrsi")

import src.indicators.drought.biomasse as bm
```

This is just the code for downloading and procesing the Biomasse data, then aggregating to a specific set of admin codes (our region of interest in Chad. Additional analysis and exploration is done within `biomasse_exploration.R`.

```python
# bm.download_dmp()
# dmp = bm.calculate_biomasse(admin="ADM2")
gdf_adm2 = gpd.read_file(adm2_bound_path)
gdf_reg = gdf_adm2[gdf_adm2.area_of_interest == True]
bm_df = bm.aggregate_biomasse(
    admin_pcodes = gdf_reg.admin2Pcod,
    iso3 = "tcd"
)
```

```python

```
