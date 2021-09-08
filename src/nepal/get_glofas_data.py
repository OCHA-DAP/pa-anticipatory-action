import logging

import geopandas as gpd

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.glofas import glofas
from src.utils_general.area import AreaFromShape, Station
from src.utils_general.utils import parse_yaml


# Stations from here:
# https://drive.google.com/file/d/1oNaavhzD2u5nZEGcEjmRn944rsQfBzfz/view
COUNTRY_ISO3 = "npl"
LEADTIMES = [
    x + 1 for x in range(10)
]  # for v3 correct coords only went to lead time 10 days
# LEADTIMES = [x + 1 for x in range(20)]

STATIONS = parse_yaml("src/nepal/config.yml")["glofas"]["stations"]

SHAPEFILE_BASE_DIR = (
    Path(os.environ["AA_DATA_DIR"])
    / "public"
    / "raw"
    / COUNTRY_ISO3
    / "cod_ab"
)
SHAPEFILE = (
    SHAPEFILE_BASE_DIR
    / "npl_admbnda_ocha_20201117"
    / "npl_admbnda_nd_20201117_shp.zip!npl_admbnda_adm0_nd_20201117.shp"
)
VERSION = 3
USE_INCORRECT_COORDS = False

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=True, process=True):

    glofas_reanalysis = glofas.GlofasReanalysis(
        use_incorrect_area_coords=USE_INCORRECT_COORDS
    )
    glofas_reforecast = glofas.GlofasReforecast(
        use_incorrect_area_coords=USE_INCORRECT_COORDS
    )

    if download:
        df_admin_boundaries = gpd.read_file(f"zip://{SHAPEFILE}")
        area = AreaFromShape(df_admin_boundaries.iloc[0]["geometry"])
        glofas_reanalysis.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
            version=VERSION,
        )
        glofas_reforecast.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
            version=VERSION,
        )

    if process:
        stations = {
            name: Station(lon=coords["lon"], lat=coords["lat"])
            for name, coords in STATIONS.items()
        }
        glofas_reanalysis.process(
            country_iso3=COUNTRY_ISO3,
            stations=stations,
            version=VERSION,
        )
        glofas_reforecast.process(
            country_iso3=COUNTRY_ISO3,
            stations=stations,
            leadtimes=LEADTIMES,
            version=VERSION,
        )


if __name__ == "__main__":
    main()
