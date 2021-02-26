"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations,
matching the FFWC stations data
"""
import logging

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding import glofas


# Location of stations on the Jamuna/Brahmaputra river from http://www.ffwc.gov.bd/index.php/googlemap?id=20
# Some lat lon indicated by FFWC are not on the river and have been manually moved to the closest pixel on the river
# TODO: Change to using GloFAS station locations?
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
        stations_lon_lat=FFWC_STATIONS, leadtime_hours=[720]
    )

    if download:
        glofas_reanalysis.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_forecast.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_reforecast.download(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)

    if process:
        glofas_reanalysis.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        glofas_forecast.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)
        # TODO: bgd_cems-glofas-reforecast_2004-11_lt0720.grib only contains one dataset.
        # It was not used to make the processed file. Should contact Copernicus to ask what the problem is
        glofas_reforecast.process(country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3)

    # Start of an analysis
    # Read in the data for a station, and select the year 2000
    station = "Bahadurabad"
    year = "2020"
    # TODO: add interpolation to make sure there aren't any missed dates
    da_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3
    )[station].sel(time=slice(year, year))
    da_forecast_720 = (
        glofas_forecast.read_processed_dataset(
            country_name=COUNTRY_NAME, country_iso3=COUNTRY_ISO3, leadtime_hour=720
        )[station]
        .shift(time=30)
        .sel(time=slice(year, year))
    )

    # Plot 1-3 sigma confidence regions against real data
    fig, ax = plt.subplots(figsize=(15, 5))
    for sigma in range(1,4):
        ax.fill_between(da_forecast_720.time, y1=np.percentile(da_forecast_720, norm.cdf(sigma) * 100, axis=0),
                        y2=np.percentile(da_forecast_720, (1 - norm.cdf(sigma)) * 100, axis=0),
                        alpha=0.3 / sigma, fc='b')
    ax.plot(da_forecast_720.time, np.median(da_forecast_720, axis=0), c='b', label='forecast median')
    ax.plot(da_reanalysis.time, da_reanalysis, c='k', label='reanalysis')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel('Water discharge (m^3 s^-1)')
    plt.show()


if __name__ == "__main__":
    main()
