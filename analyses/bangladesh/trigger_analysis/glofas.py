"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations
"""
from collections import namedtuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

DATA_DIR = Path("data")
GLOFAS_DIR = Path("GLOFAS_Data")

CDSAPI_CLIENT = cdsapi.Client()

logger = logging.getLogger(__name__)

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


def get_glofas_object(dataset_type: str) -> GlofasContainer:
    """
    Args:
        dataset_type: Must be one of 'reanalysis', 'reforecast'
    Returns: GlofasContainer instance
    """
    if dataset_type == "reanalysis":
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-historical
        glofas_object = GlofasContainer(
            year_min=1979,
            year_max=2020,
            leadtime_hours=None,
            cds_name="cems-glofas-historical",
            datasets=["consolidated_reanalysis"],
            system_version_minor=1,
        )
    elif dataset_type == "reforecast":
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-reforecast
        glofas_object = GlofasContainer(
            year_min=1999,
            year_max=2018,
            leadtime_hours=[120, 240, 360, 480, 600, 720],
            cds_name="cems-glofas-reforecast",
            datasets=["control_reforecast", "ensemble_perturbed_reforecasts"],
            system_version_minor=2,
        )
    else:
        raise UnknownGlofasDatasetType(
            msg=f"GloFAS dataset type {dataset_type} not found. "
            f'Must be one of: "reanalysis", "reforecast".'
        )
    logger.debug(f'Retrieved GloFAS object for {dataset_type}: {glofas_object}')
    return glofas_object


class UnknownGlofasDatasetType(Exception):
    def __init__(self, msg="GloFAS data type not found", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


def download_glofas_reanalysis(
    country_iso3: str,
    stations_lon_lat: dict,
    year_min: int = None,
    year_max: int = None,
):
    glofas_object = get_glofas_object(dataset_type="reanalysis")
    area = get_area(stations_lon_lat=stations_lon_lat)
    if year_min is None:
        year_min = glofas_object.year_min
    if year_max is None:
        year_max = glofas_object.year_max
    logger.info(f'Downloading GloFAS reanalysis for years {year_min} - {year_max}')
    for year in range(year_min, year_max + 1):
        logger.info(f'...{year}')
        download_glofas(
            country_iso3=country_iso3,
            system_version_minor=glofas_object.system_version_minor,
            cds_name=glofas_object.cds_name,
            dataset=glofas_object.datasets[0],
            area=area,
            year=year,
        )


def download_glofas_reforecast(
    country_iso3: str,
    stations_lon_lat: dict,
    year_min: int = None,
    year_max: int = None,
    leadtime_hours: list = None,
):
    glofas_object = get_glofas_object(dataset_type="reforecast")
    area = get_area(stations_lon_lat=stations_lon_lat)
    if year_min is None:
        year_min = glofas_object.year_min
    if year_max is None:
        year_max = glofas_object.year_max
    if leadtime_hours is None:
        leadtime_hours = glofas_object.leadtime_hours
    logger.info(f'Downloading GloFAS reanalysis for years {year_min} - {year_max} and leadtime hours {leadtime_hours}')
    for year in range(year_min, year_max + 1):
        logger.info(f'...{year}')
        for month in range(1, 13):
            for leadtime_hour in leadtime_hours:
                for dataset in glofas_object.datasets:
                    download_glofas(
                        country_iso3=country_iso3,
                        system_version_minor=glofas_object.system_version_minor,
                        cds_name=glofas_object.cds_name,
                        dataset=dataset,
                        area=area,
                        year=year,
                        month=month,
                        leadtime_hour=leadtime_hour,
                    )


def download_glofas(
    country_iso3: str,
    cds_name: str,
    system_version_minor: int,
    dataset: str,
    area: list,
    year: int,
    month: int = None,
    leadtime_hour: int = None,
    use_cache: bool = True
):
    filepath = get_glofas_filepath(
        country_iso3=country_iso3,
        cds_name=cds_name,
        dataset=dataset,
        year=year,
        month=month,
        leadtime_hour=leadtime_hour,
    )
    # If caching is on and file already exists, don't downlaod again
    if use_cache and filepath.exists():
        logger.debug(f'{filepath} already exists and cache is set to True, skipping')
        return filepath
    Path(filepath.parent).mkdir(parents=True, exist_ok=True)
    logger.debug(f'Querying for {filepath}...')
    CDSAPI_CLIENT.retrieve(
        name=cds_name,
        request=get_glofas_query(
            system_version_minor=system_version_minor,
            dataset=dataset,
            area=area,
            year=year,
            month=month,
            leadtime_hour=leadtime_hour,
        ),
        target=filepath,
    )
    logger.debug(f'...successfully downloaded {filepath}')
    return filepath


def get_glofas_filepath(
    country_iso3: str,
    cds_name: str,
    dataset: str,
    year: int,
    month: int = None,
    leadtime_hour: int = None,
):
    directory = DATA_DIR / country_iso3 / GLOFAS_DIR / cds_name / dataset
    filename = f"{country_iso3}_{cds_name}_{dataset}_{year}"
    if month is not None:
        filename += f"-{str(month).zfill(2)}"
    if leadtime_hour is not None:
        filename += f"_lt{str(leadtime_hour).zfill(4)}"
    filename += ".grib"
    return directory / Path(filename)


def get_glofas_query(
    system_version_minor: int,
    dataset: str,
    area: list,
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
        "area": area,
    }
    if system_version_minor == 1:
        query["dataset"] = dataset
    elif system_version_minor == 2:
        query["product_type"] = dataset
    if leadtime_hour is not None:
        query["leadtime_hour"] = str(leadtime_hour)
    print(query)
    return query


def get_area(stations_lon_lat: dict = None, buffer=0.5) -> list:
    """

    Args:
        stations_lon_lat: dictionary with format {station_name: [lon, lat]}
        buffer: degrees above / below maximum lat / lon from stations to include in GloFAS query

    Returns:
        list with format [N, W, S, E]
    """
    lon_list = [lon for (lon, lat) in stations_lon_lat.values()]
    lat_list = [lat for (lon, lat) in stations_lon_lat.values()]
    return [
        max(lat_list) + buffer,
        min(lon_list) - buffer,
        min(lat_list) - buffer,
        max(lon_list) + buffer,
    ]


def process_glofas_reanalysis(country_iso3: str, stations_lon_lat: dict):
    glofas_object = get_glofas_object(dataset_type="reanalysis")
    df_reanalysis = pd.DataFrame()
    for year in range(glofas_object.year_min, glofas_object.year_max + 1):
        filepath = get_glofas_filepath(
            country_iso3=country_iso3,
            cds_name=glofas_object.cds_name,
            dataset=glofas_object.datasets[0],
            year=year,
        )
        ds = xr.open_dataset(filepath, engine="cfgrib")
        df_year = pd.DataFrame()
        for station_name, lon_lat in stations_lon_lat.items():
            lat_index = np.abs(ds.latitude - lon_lat[0]).argmin()
            lon_index = np.abs(ds.longitude - lon_lat[1]).argmin()
            df_station = (
                ds.isel(latitude=lat_index, longitude=lon_index)
                .drop_vars(
                    names=["step", "surface", "latitude", "longitude", "valid_time"]
                )
                .to_dataframe()
                .rename(columns={"dis24": station_name})
            )
            df_year = df_year.merge(
                df_station, left_index=True, right_index=True, how="outer"
            )
        df_reanalysis = df_reanalysis.append(df_year)
