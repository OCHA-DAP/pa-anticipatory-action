import logging
import os
import sys

# TODO: remove this after making top-level
from pathlib import Path

import geopandas as gpd

import indicators.flooding.glofas.glofas_forecast
import indicators.flooding.glofas.glofas_reanalysis

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.glofas import glofas
from src.utils_general.area import AreaFromShape, Station

COUNTRY_ISO3 = "mwi"
LEADTIMES = [x + 1 for x in range(10)]

STATIONS = {
    "G5694": Station(lat=-16.05, lon=34.85),
    "G2001": Station(lat=-16.25, lon=34.95),
    "G1724": Station(lat=-16.45, lon=35.05),
}

SHAPEFILE_BASE_DIR = (
    Path(os.environ["AA_DATA_DIR"])
    / "public"
    / "raw"
    / COUNTRY_ISO3
    / "cod_ab"
)
SHAPEFILE = (
    SHAPEFILE_BASE_DIR
    / "mwi_adm_nso_20181016_shp"
    / "mwi_admbnda_adm0_nso_20181016.shp"
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=False, process=True):
    stations = STATIONS
    glofas_reanalysis = (
        indicators.flooding.glofas.glofas_reanalysis.GlofasReanalysis()
    )
    glofas_reforecast = (
        indicators.flooding.glofas.glofas_forecast.GlofasReforecast()
    )

    if download:
        df_admin_boundaries = gpd.read_file(SHAPEFILE)
        area = AreaFromShape(df_admin_boundaries.iloc[0]["geometry"])
        glofas_reanalysis.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
        )
        glofas_reforecast.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
        )

    if process:
        glofas_reanalysis.process(
            country_iso3=COUNTRY_ISO3,
            stations=stations,
        )
        glofas_reforecast.process(
            country_iso3=COUNTRY_ISO3,
            stations=stations,
            leadtimes=LEADTIMES,
        )


if __name__ == "__main__":
    main()
