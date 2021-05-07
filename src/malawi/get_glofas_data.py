import logging

import geopandas as gpd

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.glofas import glofas
from src.indicators.flooding.glofas.area import AreaFromShape, Station


# Stations from here: https://drive.google.com/file/d/1oNaavhzD2u5nZEGcEjmRn944rsQfBzfz/view
COUNTRY_NAME = "malawi"
COUNTRY_ISO3 = "mwi"
LEADTIMES = [5, 10, 15, 20, 25, 30]
STATIONS = {
}
SHAPEFILE_BASE_DIR = (
    Path(os.environ["AA_DATA_DIR"]) / "raw" / COUNTRY_NAME / "Shapefiles"
)
SHAPEFILE = (
    SHAPEFILE_BASE_DIR
    / "mwi_adm_nso_20181016_shp"
    / "mwi_admbnda_adm0_nso_20181016.shp"
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=True, process=True):

    glofas_reanalysis = glofas.GlofasReanalysis()
    glofas_forecast = glofas.GlofasForecast()
    glofas_reforecast = glofas.GlofasReforecast()

    if download:
        df_admin_boundaries = gpd.read_file(SHAPEFILE)
        area = AreaFromShape(df_admin_boundaries.iloc[0]["geometry"])
        glofas_reanalysis.download(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3, area=area
        )
        glofas_forecast.download(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
        )
        glofas_reforecast.download(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
        )

    if process:
        glofas_reanalysis.process(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3, stations=STATIONS
        )
        glofas_forecast.process(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            stations=STATIONS,
            leadtimes=LEADTIMES,
        )
        glofas_reforecast.process(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            stations=STATIONS,
            leadtimes=LEADTIMES,
        )


if __name__ == "__main__":
    main()
