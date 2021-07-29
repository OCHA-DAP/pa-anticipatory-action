import argparse
import logging
from pathlib import Path

import geopandas as gpd

from src.indicators.flooding.glofas import glofas
from src.indicators.flooding.glofas import area
from src.utils_general.utils import parse_yaml
from src.utils_general import admin


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_DIR = Path("src/country_config")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("country_iso3", help="country ISO3")
    return parser.parse_args()


def main(country_iso3, download=True, process=True):

    config = parse_yaml(CONFIG_DIR / f"{country_iso3}.yml")
    config_glofas = config["glofas"]
    leadtimes = [x + 1 for x in range(config_glofas["leadtimes"]["max"])]
    stations = {
        name: area.Station(lon=coords["lon"], lat=coords["lat"])
        for name, coords in config_glofas["stations"].items()
    }

    glofas_reanalysis = glofas.GlofasReanalysis()
    glofas_reforecast = glofas.GlofasReforecast()

    if download:
        if config_glofas["area_type"] == "shapefile":
            shapefile = admin.get_shapefile(
                country_iso3=country_iso3,
                base_dir=config["admin"]["base_dir"],
                base_zip=config["admin"]["base_zip"],
                base_shapefile=config["admin"]["base_shapefile"],
            )
            df_admin_boundaries = gpd.read_file(f"zip://{shapefile}")
            download_area = area.AreaFromShape(
                df_admin_boundaries.iloc[0]["geometry"]
            )
        elif config_glofas["area_type"] == "station":
            download_area = area.AreaFromStations(stations=stations)
        else:
            logger.error(
                "Parameter 'area_type' needs to be one of "
                "'shpaefile' or 'station'"
            )
            return -1
        glofas_reanalysis.download(
            country_iso3=country_iso3, area=download_area
        )
        glofas_reforecast.download(
            country_iso3=country_iso3,
            area=download_area,
            leadtimes=leadtimes,
        )

    if process:

        glofas_reanalysis.process(
            country_iso3=country_iso3,
            stations=stations,
        )
        glofas_reforecast.process(
            country_iso3=country_iso3,
            stations=stations,
            leadtimes=leadtimes,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.country_iso3.lower())
