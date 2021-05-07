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
COUNTRY_NAME = "nepal"
COUNTRY_ISO3 = "npl"
LEADTIMES = [x + 1 for x in range(20)]
# TODO: Read in the csv file from GDrive
STATIONS = {
    "Karnali": Station(lon=28.75, lat=81.25),
    "Bimalnagar": Station(lon=28.15, lat=84.45),
    "Jomsom": Station(lon=28.65, lat=83.55),
}
SHAPEFILE_BASE_DIR = (
    Path(os.environ["AA_DATA_DIR"]) / "raw" / COUNTRY_NAME / "Shapefiles"
)
SHAPEFILE = (
    SHAPEFILE_BASE_DIR
    / "npl_admbnda_ocha_20201117"
    / "npl_admbnda_nd_20201117_shp.zip!npl_admbnda_adm0_nd_20201117.shp"
)
VERSION = 3

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=True, process=True):

    glofas_reanalysis = glofas.GlofasReanalysis()
    glofas_forecast = glofas.GlofasForecast()
    glofas_reforecast = glofas.GlofasReforecast()

    if download:
        df_admin_boundaries = gpd.read_file(f"zip://{SHAPEFILE}")
        area = AreaFromShape(df_admin_boundaries.iloc[0]["geometry"])
        glofas_reanalysis.download(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            area=area,
            version=VERSION,
        )
        glofas_reforecast.download(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
            version=VERSION,
        )
        glofas_forecast.download(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
            version=VERSION,
        )

    if process:
        glofas_reanalysis.process(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            stations=STATIONS,
            version=VERSION,
        )
        glofas_reforecast.process(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            stations=STATIONS,
            leadtimes=LEADTIMES,
            version=VERSION,
        )
        glofas_forecast.process(
            country_name=COUNTRY_NAME,
            country_iso3=COUNTRY_ISO3,
            stations=STATIONS,
            leadtimes=LEADTIMES,
            version=VERSION,
        )


if __name__ == "__main__":
    main()
