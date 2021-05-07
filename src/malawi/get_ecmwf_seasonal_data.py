"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations,
matching the FFWC stations data
"""
import logging
import geopandas as gpd

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought import ecmwf_seasonal
from src.indicators.flooding.glofas.area import AreaFromShape
from src.indicators.drought.config import Config

# Location of stations on the Jamuna/Brahmaputra river from http://www.ffwc.gov.bd/index.php/googlemap?id=20
# Some lat lon indicated by FFWC are not on the river and have been manually moved to the closest pixel on the river
# Bahadurabad_glofas corresponds to the control point identified here:
# https://drive.google.com/file/d/1oNaavhzD2u5nZEGcEjmRn944rsQfBzfz/view
COUNTRY_NAME = "malawi"
COUNTRY_ISO3 = "mwi"

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

config=Config()
PARAMETERS = config.parameters(COUNTRY_NAME)
COUNTRY_DIR = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, COUNTRY_NAME)
COUNTRY_DATA_RAW_DIR = os.path.join(config.DATA_DIR,config.RAW_DIR,COUNTRY_NAME)

ADM0_BOUND_PATH=os.path.join(COUNTRY_DATA_RAW_DIR,config.SHAPEFILE_DIR,PARAMETERS["path_admin0_shp"])

def main(download=True, process=True):

    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast()
    df_country_boundaries = gpd.read_file(ADM0_BOUND_PATH)
    if download:
        area = AreaFromShape(df_country_boundaries.iloc[0]["geometry"].buffer(3))
        ecmwf_forecast.download(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3, area=area
        )

    if process:
        ecmwf_forecast.process(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)


if __name__ == "__main__":
    main()
