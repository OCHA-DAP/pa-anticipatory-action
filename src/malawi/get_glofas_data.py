import logging

import geopandas as gpd

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import glofas
from src.indicators.flooding.glofas.area import AreaFromShape, Station

COUNTRY_NAME = "malawi"
config = Config()
parameters = config.parameters(COUNTRY_NAME)

country_dir = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, COUNTRY_NAME)
country_data_raw_dir = os.path.join(config.DATA_DIR, config.RAW_DIR, COUNTRY_NAME)

LEADTIMES = parameters["glofas"]["leadtimes"]
COUNTRY_ISO3 = parameters["iso3_code"].lower()

SHAPEFILE_DIR = (
    Path(os.environ["AA_DATA_DIR"])
    / "public"
    / "raw"
    / COUNTRY_ISO3
    / "cod_ab"
    / parameters["path_admin0_shp"]
)

# TODO: Figure out how to get this from how it's stored in the config.yml
STATIONS = {
    "glofas_1": Station(lat=-16.55, lon=35.15),
    "glofas_2": Station(lat=-16.25, lon=34.95),
}


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=False, process=True):

    glofas_reanalysis = glofas.GlofasReanalysis()
    glofas_forecast = glofas.GlofasForecast()
    glofas_reforecast = glofas.GlofasReforecast()

    if download:
        df_admin_boundaries = gpd.read_file(SHAPEFILE)
        area = AreaFromShape(df_admin_boundaries.iloc[0]["geometry"])
        glofas_reanalysis.download(country_iso3=COUNTRY_ISO3, area=area)
        glofas_forecast.download(
            country_iso3=COUNTRY_ISO3, area=area, leadtimes=LEADTIMES,
        )
        glofas_reforecast.download(
            country_iso3=COUNTRY_ISO3, area=area, leadtimes=LEADTIMES,
        )

    if process:
        glofas_reanalysis.process(country_iso3=COUNTRY_ISO3, stations=STATIONS)
        glofas_forecast.process(
            country_iso3=COUNTRY_ISO3, stations=STATIONS, leadtimes=LEADTIMES,
        )
        glofas_reforecast.process(
            country_iso3=COUNTRY_ISO3, stations=STATIONS, leadtimes=LEADTIMES,
        )


if __name__ == "__main__":
    main()
