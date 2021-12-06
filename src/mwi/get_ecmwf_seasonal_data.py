"""Download raster data from GLOFAS and extracts time series of water discharge
in selected locations, matching the FFWC stations data."""
import logging
import os
import sys

# TODO: remove this after making top-level
from pathlib import Path

import geopandas as gpd

# import pandas as pd

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.indicators.drought.ecmwf_seasonal.processing import (
    compute_stats_per_admin,
)
from src.utils_general.area import AreaFromShape

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

USE_UNROUNDED_AREA_COORDS = False
SOURCE_CDS = False


def main(download=False, compute_stats=True, use_cache=True):

    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast(
        use_unrounded_area_coords=USE_UNROUNDED_AREA_COORDS
    )
    df_country_boundaries = gpd.read_file(ADM0_BOUND_PATH)
    if download:
        # retrieve the area, with a buffer from the boundary shape
        area = AreaFromShape(df_country_boundaries.buffer(3))
        # download the ecmwf data for the area
        ecmwf_forecast.download(country_iso3=COUNTRY_ISO3, area=area)
        # combine the downloaded ecmwf data into one file and
        # do a bit of postprocessing to get it in a nicer format
        ecmwf_forecast.process(country_iso3=COUNTRY_ISO3)

    if compute_stats:
        # aggregate the raster data to statistics on the admin1 level
        compute_stats_per_admin(
            iso3=COUNTRY_ISO3,
            add_col=["ADM1_EN"],
            # resolution=0.05,
            all_touched=False,
            use_cache=use_cache,
            source_cds=SOURCE_CDS,
            use_unrounded_area_coords=USE_UNROUNDED_AREA_COORDS,
            # date_list=[d.strftime("%Y-%m-%d") for d in
            #            pd.date_range(start='2000-01-01',
            #                          end='2021-04-01', freq="MS")],
        )


if __name__ == "__main__":
    main()
