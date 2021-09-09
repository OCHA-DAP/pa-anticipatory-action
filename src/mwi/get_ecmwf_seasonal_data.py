"""Download raster data from GLOFAS and extracts time series of water discharge
in selected locations, matching the FFWC stations data."""
import logging
import geopandas as gpd

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.indicators.flooding.glofas.area import AreaFromShape
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal.processing import (
    compute_stats_per_admin,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

COUNTRY_ISO3 = "mwi"
config = Config()
PARAMETERS = config.parameters(COUNTRY_ISO3)
COUNTRY_DIR = os.path.join(
    config.DIR_PATH, config.PUBLIC_DIR, config.ANALYSES_DIR, COUNTRY_ISO3
)
COUNTRY_DATA_RAW_DIR = os.path.join(
    config.DATA_DIR, config.PUBLIC_DIR, config.RAW_DIR, COUNTRY_ISO3
)

ADM0_BOUND_PATH = os.path.join(
    COUNTRY_DATA_RAW_DIR, config.SHAPEFILE_DIR, PARAMETERS["path_admin0_shp"]
)


def main(download=True, process=True):

    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast()
    df_country_boundaries = gpd.read_file(ADM0_BOUND_PATH)
    if download:
        area = AreaFromShape(
            df_country_boundaries.iloc[0]["geometry"].buffer(3)
        )
        ecmwf_forecast.download(country_iso3=COUNTRY_ISO3, area=area)

    if process:
        compute_stats_per_admin(country=COUNTRY_ISO3, interpolate=False)


if __name__ == "__main__":
    main()
