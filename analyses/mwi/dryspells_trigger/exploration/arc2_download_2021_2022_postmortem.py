## download arc2 data covering the 2021-2022 rainy season

# set up
import os
from pathlib import Path
import sys
from datetime import date

path_mod = f"{Path.cwd().parents[3]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells

import geopandas as gpd
import xarray as xr
import numpy as np

# global variables
POLY_PATH = Path(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'cod_ab',
    'mwi_drought_adm2.gpkg'
)

START_DATE = "2021-10-01"
END_DATE = "2022-05-01"
RANGE_X = ("32E", "36E")
RANGE_Y = ("20S", "5S")

# centroid method
arc2_centr = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = START_DATE,
    monitoring_end = END_DATE,
    range_x = RANGE_X,
    range_y = RANGE_Y
)

arc2_centr.download() # downloads raw raster