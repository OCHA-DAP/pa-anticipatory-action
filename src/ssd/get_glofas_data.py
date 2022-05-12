"""
Download the GloFas data for SSD.

For now only downloading and processing the reanalysis data.
"""
import logging
import os
import sys

# TODO: remove this after making top-level
from pathlib import Path

import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.glofas import glofas
from src.utils_general.area import AreaFromShape, Station
from src.utils_general.utils import parse_yaml

# Stations from here:
# https://drive.google.com/file/d/1oNaavhzD2u5nZEGcEjmRn944rsQfBzfz/view
_ISO3 = "ssd"

STATIONS = parse_yaml(f"src/{_ISO3}/config.yml")["glofas"]["stations"]

SHAPEFILE_BASE_DIR = (
    Path(os.environ["AA_DATA_DIR"]) / "public" / "raw" / _ISO3 / "cod_ab"
)
SHAPEFILE = (
    SHAPEFILE_BASE_DIR
    / "ssd_admbnda_imwg_nbs_shp"
    / "ssd_admbnda_adm0_imwg_nbs_20180817.shp"
)

VERSION = 3
USE_INCORRECT_COORDS = False

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=True, process=True):

    glofas_reanalysis = glofas.GlofasReanalysis(
        use_incorrect_area_coords=USE_INCORRECT_COORDS
    )

    if download:
        df_admin_boundaries = gpd.read_file(SHAPEFILE)
        area = AreaFromShape(df_admin_boundaries["geometry"])
        glofas_reanalysis.download(
            country_iso3=_ISO3,
            area=area,
            version=VERSION,
            year_min=2000,
            year_max=2022,
        )

    if process:
        stations = {
            name: Station(lon=coords["lon"], lat=coords["lat"])
            for name, coords in STATIONS.items()
        }
        glofas_reanalysis.process(
            country_iso3=_ISO3,
            stations=stations,
            version=VERSION,
        )


if __name__ == "__main__":
    main()
