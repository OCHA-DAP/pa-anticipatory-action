"""
# TODO: refactor & move to utils_general
Download raster data from GLOFAS and extracts time series of water discharge in selected locations,
matching the FFWC stations data
"""
import os
import zipfile
from pathlib import Path
from datetime import date, timedelta, datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import cdsapi
from netCDF4 import Dataset
import xarray as xr

# Location of stations on the Jamuna/Brahmaputra river from http://www.ffwc.gov.bd/index.php/googlemap?id=20
# Some lat lon indicated by FFWC are not on the river and have been manually moved to the closest pixel on the river
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
GLOFAS_DS_FOLDER = Path("data/GLOFAS_Data")

GlofasContainer = namedtuple(
    "Container",
    [
        "year_min",
        "year_max",
        "leadtime_hours",
        "cds_name",
        "datasets",
        "system_version_minor",
    ],
)
# Reanalysis:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-historical
GLOFAS_REANALYSIS = GlofasContainer(
    year_min=1979,
    year_max=2020,
    leadtime_hours=None,
    cds_name="cems-glofas-historical",
    datasets=["consolidated_reanalysis"],
    system_version_minor=1,
)
# Reforecast:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-reforecast
GLOFAS_REFORECAST = GlofasContainer(
    year_min=1999,
    year_max=2018,
    leadtime_hours=[120, 240, 360, 480, 600, 720],
    cds_name="cems-glofas-reforecast",
    datasets=["control_reforecast", "ensemble_perturbed_reforecasts"],
    system_version_minor=2,
)

cdsapi_client = cdsapi.Client()


def main():
    # download_glofas_reanalysis()
    # download_glofas_reforecast()
    process_glofas_reanalysis()
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


def download_glofas_reanalysis(
    year_min: int = GLOFAS_REANALYSIS.year_min,
    year_max: int = GLOFAS_REANALYSIS.year_max,
):
    for year in range(year_min, year_max + 1):
        download_glofas_zip(
            system_version_minor=GLOFAS_REANALYSIS.system_version_minor,
            cds_name=GLOFAS_REANALYSIS.cds_name,
            dataset=GLOFAS_REANALYSIS.datasets[0],
            year=year,
        )


def download_glofas_reforecast(
    year_min: int = GLOFAS_REFORECAST.year_min,
    year_max: int = GLOFAS_REFORECAST.year_max,
    leadtime_hours: list = None,
):
    if leadtime_hours is None:
        leadtime_hours = GLOFAS_REFORECAST.leadtime_hours
    for year in range(year_min, year_max + 1):
        for month in range(1, 13):
            for leadtime_hour in leadtime_hours:
                for dataset in GLOFAS_REFORECAST.datasets:
                    download_glofas_zip(
                        system_version_minor=GLOFAS_REFORECAST.system_version_minor,
                        cds_name=GLOFAS_REFORECAST.cds_name,
                        dataset=dataset,
                        year=year,
                        month=month,
                        leadtime_hour=leadtime_hour,
                    )


def download_glofas_zip(
    cds_name: str,
    system_version_minor: int,
    dataset: str,
    year: int,
    month: int = None,
    leadtime_hour: int = None,
):
    filepath = get_glofas_filepath(
        cds_name=cds_name,
        dataset=dataset,
        year=year,
        month=month,
        leadtime_hour=leadtime_hour,
    )
    Path(filepath.parent).mkdir(parents=True, exist_ok=True)
    print(filepath)
    cdsapi_client.retrieve(
        name=cds_name,
        request=get_glofas_query(
            system_version_minor=system_version_minor,
            dataset=dataset,
            year=year,
            month=month,
            leadtime_hour=leadtime_hour,
        ),
        target=filepath,
    )


def get_glofas_filepath(
    cds_name: str, dataset: str, year: int, month: int = None, leadtime_hour: int = None
):
    directory = Path(GLOFAS_DS_FOLDER) / Path(cds_name) / Path(dataset)
    filename = f"{year}"
    if month is not None:
        filename += f"-{str(month).zfill(2)}"
    if leadtime_hour is not None:
        filename += f"_lt{str(leadtime_hour).zfill(4)}"
    filename += ".grib"
    return directory / Path(filename)


def get_glofas_query(
    system_version_minor: int,
    dataset: str,
    year: int,
    month: int = None,
    leadtime_hour: int = None,
):
    query = {
        "system_version": f"version_2_{system_version_minor}",
        "variable": "river_discharge_in_the_last_24_hours",
        "format": "grib",
        "hyear": str(year),
        "hmonth": [str(x).zfill(2) for x in range(1, 13)]
        if month is None
        else str(month).zfill(2),
        "hday": [str(x).zfill(2) for x in range(1, 32)],
        "area": get_area(),
    }
    if system_version_minor == 1:
        query["dataset"] = dataset
    elif system_version_minor == 2:
        query["product_type"] = dataset
    if leadtime_hour is not None:
        query["leadtime_hour"] = str(leadtime_hour)
    print(query)
    return query


def get_area(stations_lon_lat: dict = None, buffer=0.5):
    """
    Format is: [N, W, S, E]
    """
    # TODO: refactor this out
    if stations_lon_lat is None:
        stations_lon_lat = FFWC_STATIONS
    lon_list = [lon for (lon, lat) in stations_lon_lat.values()]
    lat_list = [lat for (lon, lat) in stations_lon_lat.values()]
    return [
        max(lat_list) + buffer,
        min(lon_list) - buffer,
        min(lat_list) - buffer,
        max(lon_list) + buffer,
    ]


def process_glofas_reanalysis():
    df_reanalysis = pd.DataFrame()
    for year in range(GLOFAS_REANALYSIS.year_min, GLOFAS_REANALYSIS.year_max + 1):
        filepath = get_glofas_filepath(
            cds_name=GLOFAS_REANALYSIS.cds_name,
            dataset=GLOFAS_REANALYSIS.datasets[0],
            year=year,
        )
        ds = xr.open_dataset(filepath, engine="cfgrib")
        df_year = pd.DataFrame()
        for station_name, lon_lat in FFWC_STATIONS.items():
            lat_index = np.abs(ds.latitude - lon_lat[0]).argmin()
            lon_index = np.abs(ds.longitude - lon_lat[1]).argmin()
            df_station = (
                ds.isel(latitude=lat_index, longitude=lon_index)
                .drop_vars(names=["step", "surface", "latitude", "longitude", "valid_time"])
                .to_dataframe()
                .rename(columns={'dis24': station_name})
            )
            df_year = df_year.merge(df_station, left_index=True,
                                                right_index=True, how='outer')
        df_reanalysis = df_reanalysis.append(df_year)


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
