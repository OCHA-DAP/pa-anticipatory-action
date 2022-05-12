import logging
import os
import sys

# TODO: remove this after making top-level
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.chirps_rainfallobservations import (
    compute_seasonal_tercile_raster,
    get_chirps_data_monthly,
)
from src.indicators.drought.config import Config

logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

COUNTRY_ISO3 = "bfa"

get_chirps_data_monthly(
    config=Config(),
    country_iso3=COUNTRY_ISO3,
)

compute_seasonal_tercile_raster(config=Config(), country_iso3=COUNTRY_ISO3)
