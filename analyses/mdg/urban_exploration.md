### Urban classification in Madagascar

Very simple script to classify ADM4 areas in Madagascar as urban based on GHS data.

```python
from pathlib import Path
import sys
import os
import geopandas as gpd
import rasterio
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.utils_general.ghs import get_ghs_data
from src.utils_general.ghs import classify_urban_areas

# Setup

iso3 = "mdg"
raw_dir = os.path.join(os.environ["AA_DATA_DIR"], 'public', 'raw', iso3)
processed_dir = os.path.join(os.environ["AA_DATA_DIR"], 'public', 'processed', iso3)
adm4_path = os.path.join(raw_dir, 'cod_ab', 'mdg_admbnda_adm4_BNGRC_OCHA_20181031.shp')
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

adm4 = gpd.read_file(adm4_path, crs='4326').to_crs('ESRI:54009')
with rasterio.open(os.path.join(raw_dir, 'ghs', 'mdg_SMOD_2015_1km_mosaic.tif')) as src:
    smod = src.read(1)
    trans = src.transform

cls = classify_urban_areas(adm4, smod, trans)
adm4['urban_area'] = [x['urban_area'] for x in cls]
```

Save out results.

```python
adm4.drop('geometry', axis=1).to_csv(os.path.join(processed_dir, 'urban_classification', 'mdg_adm4_urban_classification.csv'))
```

```python

```
