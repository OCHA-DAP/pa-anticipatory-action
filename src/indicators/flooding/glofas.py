"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations
"""
from pathlib import Path
import logging
import time
import os

import numpy as np
import xarray as xr
import cdsapi

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GLOFAS_DIR = Path("GLOFAS_Data")

CDSAPI_CLIENT = cdsapi.Client()

logger = logging.getLogger(__name__)


class Glofas:
    def __init__(
        self,
        stations_lon_lat: dict,
        year_min: int,
        year_max: int,
        cds_name: str,
        dataset: list,
        dataset_variable_name: str,
        system_version_minor: int = None,
        date_variable_prefix: str = "",
        leadtime_hours: list = None,
    ):
        self.stations_lon_lat = stations_lon_lat
        self.year_min = year_min
        self.year_max = year_max
        self.cds_name = cds_name
        self.dataset = dataset
        self.dataset_variable_name = dataset_variable_name
        self.system_version_minor = system_version_minor
        self.date_variable_prefix = date_variable_prefix
        self.leadtime_hours = leadtime_hours
        self.area = get_area(self.stations_lon_lat)

    def _download(
        self,
        country_name: str,
        country_iso3: str,
        year: int,
        month: int = None,
        leadtime_hour: int = None,
        use_cache: bool = True,
    ):
        filepath = self._get_raw_filepath(
            country_name=country_name,
            country_iso3=country_iso3,
            year=year,
            month=month,
            leadtime_hour=leadtime_hour,
        )
        # If caching is on and file already exists, don't downlaod again
        if use_cache and filepath.exists():
            logger.debug(
                f"{filepath} already exists and cache is set to True, skipping"
            )
            return filepath
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Querying for {filepath}...")
        CDSAPI_CLIENT.retrieve(
            name=self.cds_name,
            request=self._get_query(
                year=year,
                month=month,
                leadtime_hour=leadtime_hour,
            ),
            target=filepath,
        )
        logger.debug(f"...successfully downloaded {filepath}")
        # Wait 2 seconds between requests or else API hangs
        # TODO make sure this actually works
        time.sleep(2)
        return filepath

    def _get_raw_filepath(
        self,
        country_name: str,
        country_iso3: str,
        year: int,
        month: int = None,
        leadtime_hour: int = None,
    ):
        directory = RAW_DATA_DIR / country_name / GLOFAS_DIR / self.cds_name
        filename = f"{country_iso3}_{self.cds_name}_{year}"
        if month is not None:
            filename += f"-{str(month).zfill(2)}"
        if leadtime_hour is not None:
            filename += f"_lt{str(leadtime_hour).zfill(4)}"
        filename += ".grib"
        return directory / Path(filename)

    def _get_query(
        self,
        year: int,
        month: int = None,
        leadtime_hour: int = None,
    ) -> dict:
        query = {
            "variable": "river_discharge_in_the_last_24_hours",
            "format": "grib",
            self.dataset_variable_name: self.dataset,
            f"{self.date_variable_prefix}year": str(year),
            f"{self.date_variable_prefix}month": [str(x).zfill(2) for x in range(1, 13)]
            if month is None
            else str(month).zfill(2),
            f"{self.date_variable_prefix}day": [str(x).zfill(2) for x in range(1, 32)],
            "area": self.area,
        }
        if leadtime_hour is not None:
            query["leadtime_hour"] = str(leadtime_hour)
        if self.system_version_minor is not None:
            query["system_version"] = f"version_2_{self.system_version_minor}"
        logger.debug(f"Query: {query}")
        return query

    @staticmethod
    def _read_in_ensemble_and_perturbed_datasets(filepath_list):
        """
        Read in dataset that has both control and ensemble perturbed forecast
        and combine them
        """
        ds_list = []
        for data_type in ["cf", "pf"]:
            ds = xr.open_mfdataset(
                filepath_list,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {"dataType": data_type},
                },
            )
            # Delete history attribute in order to merge
            del ds.attrs["history"]
            # Extra processing require for control forecast
            if data_type == "cf":
                ds = expand_dims(
                    ds=ds,
                    dataset_name="dis24",
                    coord_names=["number", "time", "latitude", "longitude"],
                    expansion_dim=0,
                )
            ds_list.append(ds)
        ds = xr.combine_by_coords(ds_list)
        return ds

    def _get_station_dataset(self, ds: xr.Dataset, coord_names: list) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                station_name: (
                    coord_names,
                    ds.isel(
                        latitude=np.abs(ds.latitude - lat).argmin(),
                        longitude=np.abs(ds.longitude - lon).argmin(),
                    )["dis24"],
                )
                for station_name, (lon, lat) in self.stations_lon_lat.items()
            },
            coords={coord_name: ds[coord_name] for coord_name in coord_names},
        )

    def _write_to_processed_file(
        self, country_name: str, country_iso3: str, ds: xr.Dataset, leadtime_hour: int = None
    ) -> Path:
        filepath = self._get_processed_filepath(
            country_name=country_name, country_iso3=country_iso3, leadtime_hour=leadtime_hour
        )
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to {filepath}")
        ds.to_netcdf(filepath)
        return filepath

    def _get_processed_filepath(self, country_name: str, country_iso3: str, leadtime_hour: int = None) -> Path:
        filename = f"{country_iso3}_{self.cds_name}"
        if leadtime_hour is not None:
           filename += f"_{str(leadtime_hour).zfill(4)}"
        filename += ".nc"
        return (
            PROCESSED_DATA_DIR
            / country_name
            / GLOFAS_DIR
            / filename
        )


class GlofasReanalysis(Glofas):
    def __init__(self, stations_lon_lat: dict):
        super().__init__(
            stations_lon_lat=stations_lon_lat,
            year_min=1979,
            year_max=2021,
            cds_name="cems-glofas-historical",
            dataset=["consolidated_reanalysis"],
            dataset_variable_name="dataset",
            date_variable_prefix="h",
            system_version_minor=1,
        )

    def download(
        self,
        country_name: str,
        country_iso3: str,
    ):
        logger.info(
            f"Downloading GloFAS reanalysis for years {self.year_min} - {self.year_max}"
        )
        for year in range(self.year_min, self.year_max + 1):
            logger.info(f"...{year}")
            super()._download(
                country_name=country_name, country_iso3=country_iso3, year=year
            )

    def process(self, country_name: str, country_iso3: str):
        # Get list of files to open
        logger.info("Processing GloFAS Reanalysis")
        filepath_list = [
            self._get_raw_filepath(
                country_name=country_name,
                country_iso3=country_iso3,
                year=year,
            )
            for year in range(self.year_min, self.year_max + 1)
        ]
        # Read in the dataset
        logger.info(f"Reading in {len(filepath_list)} files")
        ds = xr.open_mfdataset(
            filepath_list, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        # Create a new dataset with just the station pixels
        logger.info("Looping through stations, this takes some time")
        ds_new = self._get_station_dataset(ds=ds, coord_names=["time"])
        # Write out the new dataset to a file
        self._write_to_processed_file(
            country_name=country_name, country_iso3=country_iso3, ds=ds_new
        )


class GlofasForecast(Glofas):
    def __init__(self, stations_lon_lat: dict, leadtime_hours: list):
        super().__init__(
            stations_lon_lat=stations_lon_lat,
            year_min=2019,
            year_max=2021,
            cds_name="cems-glofas-forecast",
            dataset=["control_forecast", "ensemble_perturbed_forecasts"],
            dataset_variable_name="product_type",
            leadtime_hours=leadtime_hours
        )

    def download(
        self,
        country_name: str,
        country_iso3: str,
    ):
        logger.info(
            f"Downloading GloFAS forecast for years {self.year_min} - {self.year_max} and leadtime hours {self.leadtime_hours}"
        )
        for year in range(self.year_min, self.year_max + 1):
            logger.info(f"...{year}")
            for leadtime_hour in self.leadtime_hours:
                super()._download(
                    country_name=country_name,
                    country_iso3=country_iso3,
                    year=year,
                    leadtime_hour=leadtime_hour,
                )

    def process(self, country_name: str, country_iso3: str):
        logger.info("Processing GloFAS Forecast")
        for leadtime_hour in self.leadtime_hours:
            logger.info(f"For lead time {leadtime_hour}")
            # Get list of files to open
            filepath_list = [
                self._get_raw_filepath(
                    country_name=country_name,
                    country_iso3=country_iso3,
                    year=year,
                    leadtime_hour=leadtime_hour,
                )
                for year in range(self.year_min, self.year_max + 1)
            ]
            # Read in both the control and ensemble perturbed forecast and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_ensemble_and_perturbed_datasets(filepath_list)
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = self._get_station_dataset(ds=ds, coord_names=["number", "time"])
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_name=country_name, country_iso3=country_iso3, ds=ds_new, leadtime_hour=leadtime_hour
            )


class GlofasReforecast(Glofas):
    def __init__(self, stations_lon_lat: dict, leadtime_hours: list):
        super().__init__(
            stations_lon_lat=stations_lon_lat,
            year_min=1999,
            year_max=2018,
            cds_name="cems-glofas-reforecast",
            dataset=["control_reforecast", "ensemble_perturbed_reforecasts"],
            dataset_variable_name="product_type",
            system_version_minor=2,
            date_variable_prefix="h",
            leadtime_hours=leadtime_hours
        )

    def download(
        self,
        country_name: str,
        country_iso3: str,
    ):
        logger.info(
            f"Downloading GloFAS reanalysis for years {self.year_min} - {self.year_max} and leadtime hours {self.leadtime_hours}"
        )
        for year in range(self.year_min, self.year_max + 1):
            logger.info(f"...{year}")
            for month in range(1, 13):
                for leadtime_hour in self.leadtime_hours:
                    super()._download(
                        country_name=country_name,
                        country_iso3=country_iso3,
                        year=year,
                        month=month,
                        leadtime_hour=leadtime_hour,
                    )

    def process(self, country_name: str, country_iso3: str):
        logger.info("Processing GloFAS Reforecast")
        for leadtime_hour in self.leadtime_hours:
            logger.info(f"For lead time {leadtime_hour}")
            # Get list of files to open
            filepath_list = [
                self._get_raw_filepath(
                    country_name=country_name,
                    country_iso3=country_iso3,
                    year=year,
                    month=month,
                    leadtime_hour=leadtime_hour,
                )
                for year in range(self.year_min, self.year_max + 1)
                for month in range(1, 13)
            ]
            # Read in both the control and ensemble perturbed forecast and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_ensemble_and_perturbed_datasets(filepath_list)
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = self._get_station_dataset(ds=ds, coord_names=["number", "time"])
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_name=country_name, country_iso3=country_iso3, ds=ds_new, leadtime_hour=leadtime_hour
            )


def get_area(stations_lon_lat: dict, buffer: float = 0.5) -> list:
    """
    Args:
        stations_lon_lat: dictionary of form {station_name: [lon (float), lat (float)]
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


def expand_dims(
    ds: xr.Dataset, dataset_name: str, coord_names: list, expansion_dim: int
):
    """
    Using expand_dims seems to cause a bug with Dask like the one described here:
    https://github.com/pydata/xarray/issues/873 (it's supposed to be fixed though)
    """
    coords = {coord_name: ds[coord_name] for coord_name in coord_names}
    coords[coord_names[expansion_dim]] = [coords[coord_names[expansion_dim]]]
    ds = xr.Dataset(
        data_vars={
            dataset_name: (
                coord_names,
                np.expand_dims(ds[dataset_name].values, expansion_dim),
            )
        },
        coords=coords,
    )
    return ds
