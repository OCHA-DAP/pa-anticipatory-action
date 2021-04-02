"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations,
matching the FFWC stations data
"""
import logging

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding import glofas

# Location of stations on the Jamuna/Brahmaputra river from http://www.ffwc.gov.bd/index.php/googlemap?id=20
# Some lat lon indicated by FFWC are not on the river and have been manually moved to the closest pixel on the river
# Bahadurabad_glofas corresponds to the control point identified here:
# https://drive.google.com/file/d/1oNaavhzD2u5nZEGcEjmRn944rsQfBzfz/view
COUNTRY_NAME = "bangladesh"
COUNTRY_ISO3 = "bgd"
FFWC_STATIONS = {
    "Noonkhawa": [89.9509, 25.9496],
    "Chilmari": [89.7476, 25.5451],
    "Bahadurabad": [89.6607, 25.1028],
    "Bahadurabad_glofas": [89.65, 25.15],
    "Sariakandi": [89.6518, 24.8901],
    "Kazipur": [89.7498, 24.6637],
    "Serajganj": [89.7479, 24.4676],
    "Aricha": [89.6550, 23.9032],
}
LEADTIME_HOURS = [120, 240, 360, 480, 600, 720]

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=False, process=False):

    # TODO: flags / config file to toggle these things
    glofas_reanalysis = glofas.GlofasReanalysis(stations_lon_lat=FFWC_STATIONS)
    glofas_forecast = glofas.GlofasForecast(
        stations_lon_lat=FFWC_STATIONS, leadtime_hours=LEADTIME_HOURS
    )
    glofas_reforecast = glofas.GlofasReforecast(
        stations_lon_lat=FFWC_STATIONS, leadtime_hours=LEADTIME_HOURS
    )

    if download:
        glofas_reanalysis.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_forecast.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_reforecast.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)

    if process:
        glofas_reanalysis.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_forecast.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_reforecast.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)


if __name__ == "__main__":
    main(download=False, process=False)
