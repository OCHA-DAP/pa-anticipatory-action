"""Download raster data from GLOFAS and extracts time series of water
discharge in selected locations, matching the FFWC stations data."""
import logging

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.glofas import glofas
from src.utils_general.area import AreaFromStations, Station

# Location of stations on the Jamuna/Brahmaputra river from
# http://www.ffwc.gov.bd/index.php/googlemap?id=20 Some lat lon
# indicated by FFWC are not on the river and have been manually moved to
# the closest pixel on the river Bahadurabad_glofas corresponds to the
# control point identified here:
# https://drive.google.com/file/d/1oNaavhzD2u5nZEGcEjmRn944rsQfBzfz/view
COUNTRY_ISO3 = "bgd"
FFWC_STATIONS = {
    "Noonkhawa": Station(lon=89.9509, lat=25.9496),
    "Chilmari": Station(lon=89.7476, lat=25.5451),
    "Bahadurabad": Station(lon=89.6607, lat=25.1028),
    "Sariakandi": Station(lon=89.6518, lat=24.8901),
    "Kazipur": Station(lon=89.7498, lat=24.6637),
    "Serajganj": Station(lon=89.7479, lat=24.4676),
    "Aricha": Station(lon=89.6550, lat=23.9032),
    "Bahadurabad_glofas": Station(lon=89.65, lat=25.15),
}
LEADTIMES = [5, 10, 15, 20, 25, 30]
AREA_BUFFER = 0.5
VERSION = 2
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main(download=True, process=False):

    # TODO: flags / config file to toggle these things
    glofas_reanalysis = glofas.GlofasReanalysis()
    glofas_forecast = glofas.GlofasForecast()
    glofas_reforecast = glofas.GlofasReforecast()

    if download:
        # Remove the GloFAS station as it was not used originally
        ffwc_stations_for_download = FFWC_STATIONS.copy()
        del ffwc_stations_for_download["Bahadurabad_glofas"]
        area = AreaFromStations(
            stations=ffwc_stations_for_download, buffer=AREA_BUFFER
        )
        glofas_reanalysis.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
            version=VERSION,
        )
        glofas_forecast.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
            version=VERSION,
        )
        glofas_reforecast.download(
            country_iso3=COUNTRY_ISO3,
            area=area,
            leadtimes=LEADTIMES,
            version=VERSION,
            split_by_month=True,
        )

    if process:
        glofas_reanalysis.process(
            country_iso3=COUNTRY_ISO3,
            stations=FFWC_STATIONS,
            version=VERSION,
        )
        glofas_forecast.process(
            country_iso3=COUNTRY_ISO3,
            stations=FFWC_STATIONS,
            leadtimes=LEADTIMES,
            version=VERSION,
        )
        glofas_reforecast.process(
            country_iso3=COUNTRY_ISO3,
            stations=FFWC_STATIONS,
            leadtimes=LEADTIMES,
            version=VERSION,
            split_by_month=True,
        )


if __name__ == "__main__":
    main()
