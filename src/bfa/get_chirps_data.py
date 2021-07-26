import logging

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.chirps_rainfallobservations import (
    get_chirps_data_monthly,
    compute_seasonal_lowertercile_raster,
)
from src.indicators.drought.config import Config

logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

COUNTRY_NAME = "bfa"
COUNTRY_ISO3 = "bfa"

get_chirps_data_monthly(
    config=Config(),
    country_name=COUNTRY_NAME,
    country_iso3=COUNTRY_ISO3,
)

compute_seasonal_lowertercile_raster(
    config=Config(),
    country_iso3=COUNTRY_ISO3,
)
