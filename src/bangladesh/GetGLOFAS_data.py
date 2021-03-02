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
COUNTRY_ISO3 = 'bangladesh' # change to iso3 later
FFWC_STATIONS = {
    "Noonkhawa": [89.9509, 25.9496],
    "Chilmari": [89.7476, 25.5451],
    "Bahadurabad": [89.6607, 25.1028],
    "Sariakandi": [89.6518, 24.8901],
    "Kazipur": [89.7498, 24.6637],
    "Serajganj": [89.7479, 24.4676],
    "Aricha": [89.6550, 23.9032],
}
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
GLOFAS_DS_FILENAME = "CEMS_ECMWF_dis24_{}_glofas_v2.1.nc"


logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)


def main():
    glofas.download_glofas_reanalysis(country_iso3=COUNTRY_ISO3, stations_lon_lat=FFWC_STATIONS)
    glofas.download_glofas_reforecast(country_iso3=COUNTRY_ISO3, stations_lon_lat=FFWC_STATIONS)
    glofas.process_glofas_reanalysis(country_iso3=COUNTRY_ISO3, stations_lon_lat=FFWC_STATIONS)
    """"
    for year in range(1979, 2021):
        print(year)
        folder = '{}/{}/{}'.format(DIR_PATH, GLOFAS_DS_FOLDER, year)
        # get_GLOFAS_zip(c,year,folder)
        unzip('{}/download_{}.zip'.format(folder, year), folder)
        glofas_df = get_glofas_df(year, folder)
        glofas_df.to_csv('{}/{}/{}.csv'.format(DIR_PATH, GLOFAS_DS_FOLDER, year))
        shutil.rmtree(folder)
   """
######################


def extract_dis24_values(date, folder, glofas_df):
    try:
        glofas_ds_name = "{}/{}".format(
            folder, GLOFAS_DS_FILENAME.format(date.strftime("%Y%m%d"))
        )
        glofas = Dataset(glofas_ds_name, "r")

        lats = glofas.variables["lat"][:]
        lons = glofas.variables["lon"][:]

        # loop over stations
        for station_name, station_lonlat in FFWC_STATIONS.items():
            # latitude lower and upper index
            lonbound = np.argmin(np.abs(lons - station_lonlat[0]))
            latbound = np.argmin(np.abs(lats - station_lonlat[1]))
            # variables are lon,lat,time,dis24
            discharge = glofas.variables["dis24"][:, latbound, lonbound]
            if discharge is not None:
                glofas_df.at[date, f"dis24_{station_name}"] = discharge
        return glofas_df
    except Exception as e:
        print(f"discharge data not available for {date}")
        print(e)
        return glofas_df


def get_glofas_df(year, folder):
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    delta = timedelta(days=1)
    glofas_df = pd.DataFrame()
    while start_date <= end_date:
        # print(start_date.strftime("%Y-%m-%d"))
        glofas_df = extract_dis24_values(start_date, folder, glofas_df)
        start_date += delta
    glofas_df.index = pd.to_datetime(glofas_df.index)
    glofas_df.index.name = "date"
    glofas_df = glofas_df.resample("D").mean().interpolate(method="linear")
    return glofas_df


def unzip(zip_file_path, save_path):
    print(f"Unzipping {zip_file_path}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(save_path)


if __name__ == "__main__":
    main()