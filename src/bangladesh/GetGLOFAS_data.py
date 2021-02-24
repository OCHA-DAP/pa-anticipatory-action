"""
# TODO: refactor & move to utils_general
Download raster data from GLOFAS and extracts time series of water discharge in selected locations,
matching the FFWC stations data
"""
import os
import zipfile
from datetime import date, timedelta
import logging

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import glofas

# Location of stations on the Jamuna/Brahmaputra river from http://www.ffwc.gov.bd/index.php/googlemap?id=20
# Some lat lon indicated by FFWC are not on the river and have been manually moved to the closest pixel on the river
COUNTRY_NAME = "bangladesh"
COUNTRY_ISO3 = "bgd"
FFWC_STATIONS = {
    "Noonkhawa": [89.9509, 25.9496],
    "Chilmari": [89.7476, 25.5451],
    "Bahadurabad": [89.6607, 25.1028],
    "Sariakandi": [89.6518, 24.8901],
    "Kazipur": [89.7498, 24.6637],
    "Serajganj": [89.7479, 24.4676],
    "Aricha": [89.6550, 23.9032],
}
LEADTIME_HOURS = [120, 240, 480, 600, 720]

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():

    glofas_reanalysis = glofas.GlofasReanalysis(stations_lon_lat=FFWC_STATIONS)
    glofas_reanalysis.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
    glofas_reanalysis.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)

    glofas_reforecast = glofas.GlofasReforecast(
        stations_lon_lat=FFWC_STATIONS, leadtime_hours=LEADTIME_HOURS
    )
    glofas_reforecast.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
    glofas_reforecast.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)


if __name__ == "__main__":
    main()
