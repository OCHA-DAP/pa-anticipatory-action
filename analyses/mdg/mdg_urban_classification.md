### Urban classification in Madagascar

Very simple script to classify ADM4 areas in Madagascar as urban based on GHS data.

```python
from pathlib import Path
import sys
import os
import numpy as np
import geopandas as gpd
import rasterio

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.utils_general.ghs import get_ghs_data
from src.utils_general.ghs import classify_urban_areas

# Setup

iso3 = "mdg"
raw_dir = os.path.join(os.environ["AA_DATA_DIR"], 'public', 'raw', iso3)
processed_dir = os.path.join(os.environ["AA_DATA_DIR"], 'public', 'processed', iso3)

# Adjustable settings
ADM_LEVEL = "adm3"   # ADM level for aggregation
URBAN_MIN_CLASS = 21 # passed to get_ghs_data()
URBAN_PERCENT = 0.5  # passed to get_ghs_data()

adm_path = os.path.join(raw_dir, 'cod_ab', f"mdg_admbnda_{ADM_LEVEL}_BNGRC_OCHA_20181031.shp")
```

Use a couple of functions to load GHS data, and then classify urban areas off of that. Can adjust GHS grid cells considered urban areas and % of raster cells per polygon required to classify polygon as urban.

```python
# GHS tile bounding boxes for MDG
box = [(22,10),
       (22,11),
       (22,12)]

# Download data
get_ghs_data("SMOD", box, iso3, raw_dir)

# Load data
adm = gpd.read_file(adm_path, crs='4326').to_crs('ESRI:54009')

with rasterio.open(os.path.join(raw_dir, 'ghs', 'mdg_SMOD_2015_1km_mosaic.tif')) as src:
    smod = src.read(1)
    trans = src.transform

cls = classify_urban_areas(adm, smod, trans, URBAN_MIN_CLASS, URBAN_PERCENT)
adm['urban_percent'] = [x['urban_percent'] for x in cls]
adm['urban_area'] = [x['urban_area'] for x in cls]
adm['urban_area_weighted'] = [x['urban_area_weighted'] for x in cls]
```

Save out results. Generates an error if the file is already present, remove `move = 'x'` to allow overwriting.

```python
output_path = os.path.join(processed_dir, 'urban_classification', f'mdg_{ADM_LEVEL}_urban_classification.csv')
adm.drop('geometry', axis=1).to_csv(output_path)
```
