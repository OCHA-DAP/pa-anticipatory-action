"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations
"""
from pathlib import Path
import logging
import time
import os
from typing import Dict, List, Union

import numpy as np
import xarray as xr
import cdsapi

from src.indicators.flooding.glofas.area import Area, Station


DATA_DIR = Path(os.environ["AA_DATA_DIR"])
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GLOFAS_DIR = Path("GLOFAS_Data")
CDSAPI_CLIENT = cdsapi.Client()
DEFAULT_VERSION = 3
HYDROLOGICAL_MODELS = {2: "htessel_lisflood", 3: "lisflood"}

logger = logging.getLogger(__name__)


class Glofas:
    def __init__(
        self,
        year_min: Union[int, Dict[int, int]],
        year_max: int,
        cds_name: str,
        dataset: List[str],
        dataset_variable_name: str,
        system_version_minor: Dict[int, int],
        date_variable_prefix: str = "",
    ):
        """
        Create an instance of a GloFAS object, from which you can download and process raw data, and
        read in the processed data.
        :param year_min: The earliest year that the dataset is available. Can be a single integer,
        or a dictionary with structure {major_version: year_min} if the minimum year depends on the GloFAS
        model version.
        :param year_max: The most recent that the dataset is available
        :param cds_name: The name of the dataset in CDS
        :param dataset: The sub-datasets that you would like to download (as a list of strings)
        :param dataset_variable_name: The variable name with which to pass the above datasets in the CDS query
        :param system_version_minor: The minor version of the GloFAS model. Depends on the major version,
        so is given as a dictionary with the format {major_version: minor_version}
        :param date_variable_prefix: Some GloFAS datasets have the prefix "h" in front of some query keys
        """
        self.year_min = year_min
        self.year_max = year_max
        self.cds_name = cds_name
        self.dataset = dataset
        self.dataset_variable_name = dataset_variable_name
        self.system_version_minor = system_version_minor
        self.date_variable_prefix = date_variable_prefix

    def _download(
        self,
        country_name: str,
        country_iso3: str,
        area: Area,
        version: int,
        year: int,
        month: int = None,
        leadtime: int = None,
        use_cache: bool = True,
    ):
        filepath = self._get_raw_filepath(
            country_name=country_name,
            country_iso3=country_iso3,
            version=version,
            year=year,
            month=month,
            leadtime=leadtime,
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
                area=area,
                version=version,
                year=year,
                month=month,
                leadtime=leadtime,
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
        version: int,
        year: int,
        month: int = None,
        leadtime: int = None,
    ):
        directory = (
            RAW_DATA_DIR
            / country_name
            / GLOFAS_DIR
            / f"version_{version}"
            / self.cds_name
        )
        filename = f"{country_iso3}_{self.cds_name}_v{version}_{year}"
        if month is not None:
            filename += f"-{str(month).zfill(2)}"
        if leadtime is not None:
            filename += f"_lt{str(leadtime).zfill(2)}d"
        filename += ".grib"
        return directory / Path(filename)

    def _get_query(
        self,
        area: Area,
        version: int,
        year: int,
        month: int = None,
        leadtime: int = None,
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
            "area": area.list_for_api(),
            "system_version": f"version_{version}_{self.system_version_minor[version]}",
            "hydrological_model": HYDROLOGICAL_MODELS[version],
        }
        if leadtime is not None:
            query["leadtime_hour"] = str(leadtime * 24)
        logger.debug(f"Query: {query}")
        return query

    @staticmethod
    def _read_in_ensemble_and_perturbed_datasets(filepath_list: List[Path]):
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

    def _write_to_processed_file(
        self,
        country_name: str,
        country_iso3: str,
        version: int,
        ds: xr.Dataset,
        leadtime: int = None,
    ) -> Path:
        filepath = self._get_processed_filepath(
            country_name=country_name,
            country_iso3=country_iso3,
            version=version,
            leadtime=leadtime,
        )
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to {filepath}")
        ds.to_netcdf(filepath)
        return filepath

    def _get_processed_filepath(
        self, country_name: str, country_iso3: str, version: int, leadtime: int = None
    ) -> Path:
        filename = f"{country_iso3}_{self.cds_name}_v{version}"
        if leadtime is not None:
            filename += f"_lt{str(leadtime).zfill(2)}d"
        filename += ".nc"
        return PROCESSED_DATA_DIR / country_name / GLOFAS_DIR / filename

    def read_processed_dataset(
        self, country_name: str, country_iso3: str, version: int, leadtime: int = None
    ):
        filepath = self._get_processed_filepath(
            country_name=country_name,
            country_iso3=country_iso3,
            version=version,
            leadtime=leadtime,
        )
        return xr.open_dataset(filepath)


class GlofasReanalysis(Glofas):
    def __init__(self):
        super().__init__(
            year_min=1979,
            year_max=2020,
            cds_name="cems-glofas-historical",
            dataset=["consolidated_reanalysis"],
            dataset_variable_name="dataset",
            system_version_minor={2: 1, 3: 1},
            date_variable_prefix="h",
        )

    def download(
        self,
        country_name: str,
        country_iso3: str,
        area: Area,
        version: int = DEFAULT_VERSION,
    ):
        logger.info(
            f"Downloading GloFAS reanalysis v{version} for years {self.year_min} - {self.year_max}"
        )
        for year in range(self.year_min, self.year_max + 1):
            logger.info(f"...{year}")
            super()._download(
                country_name=country_name,
                country_iso3=country_iso3,
                area=area,
                year=year,
                version=version,
            )

    def process(
        self,
        country_name: str,
        country_iso3: str,
        stations: Dict[str, Station],
        version: int = DEFAULT_VERSION,
    ):
        # Get list of files to open
        logger.info(f"Processing GloFAS Reanalysis v{version}")
        filepath_list = [
            self._get_raw_filepath(
                country_name=country_name,
                country_iso3=country_iso3,
                version=version,
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
        ds_new = _get_station_dataset(stations=stations, ds=ds, coord_names=["time"])
        # Write out the new dataset to a file
        self._write_to_processed_file(
            country_name=country_name,
            country_iso3=country_iso3,
            version=version,
            ds=ds_new,
        )


class GlofasForecast(Glofas):
    def __init__(self):
        super().__init__(
            year_min={2: 2019, 3: 2020},
            year_max=2021,
            cds_name="cems-glofas-forecast",
            dataset=["control_forecast", "ensemble_perturbed_forecasts"],
            system_version_minor={2: 1, 3: 1},
            dataset_variable_name="product_type",
        )

    def download(
        self,
        country_name: str,
        country_iso3: str,
        area: Area,
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
    ):
        logger.info(
            f"Downloading GloFAS forecast v{version} for years {self.year_min[version]} - {self.year_max} and leadtime hours {leadtimes}"
        )
        for year in range(self.year_min[version], self.year_max + 1):
            logger.info(f"...{year}")
            for leadtime in leadtimes:
                super()._download(
                    country_name=country_name,
                    country_iso3=country_iso3,
                    area=area,
                    year=year,
                    leadtime=leadtime,
                    version=version,
                )

    def process(
        self,
        country_name: str,
        country_iso3: str,
        stations: Dict[str, Station],
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
    ):
        logger.info(f"Processing GloFAS Forecast v{version}")
        for leadtime in leadtimes:
            logger.info(f"For lead time {leadtime}")
            # Get list of files to open
            filepath_list = [
                self._get_raw_filepath(
                    country_name=country_name,
                    country_iso3=country_iso3,
                    version=version,
                    year=year,
                    leadtime=leadtime,
                )
                for year in range(self.year_min[version], self.year_max + 1)
            ]
            # Read in both the control and ensemble perturbed forecast and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_ensemble_and_perturbed_datasets(filepath_list)
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = _get_station_dataset(
                stations=stations, ds=ds, coord_names=["number", "time"]
            )
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_name=country_name,
                country_iso3=country_iso3,
                ds=ds_new,
                leadtime=leadtime,
                version=version,
            )


class GlofasReforecast(Glofas):
    def __init__(self):
        super().__init__(
            year_min=1999,
            year_max=2018,
            cds_name="cems-glofas-reforecast",
            dataset=["control_reforecast", "ensemble_perturbed_reforecasts"],
            dataset_variable_name="product_type",
            system_version_minor={2: 2, 3: 1},
            date_variable_prefix="h",
        )

    def download(
        self,
        country_name: str,
        country_iso3: str,
        area: Area,
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
        split_by_month: bool = False,
    ):
        logger.info(
            f"Downloading GloFAS reforecast v{version} for years {self.year_min} - {self.year_max} and leadtime hours {leadtimes}"
        )
        for year in range(self.year_min, self.year_max + 1):
            logger.info(f"...{year}")
            month_range = range(1, 13) if split_by_month else [None]
            for month in month_range:
                for leadtime in leadtimes:
                    super()._download(
                        country_name=country_name,
                        country_iso3=country_iso3,
                        area=area,
                        version=version,
                        year=year,
                        month=month,
                        leadtime=leadtime,
                    )

    def process(
        self,
        country_name: str,
        country_iso3: str,
        stations: Dict[str, Station],
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
        split_by_month: bool = False,
    ):
        logger.info(f"Processing GloFAS Reforecast v{version}")
        for leadtime in leadtimes:
            logger.info(f"For lead time {leadtime}")
            # Get list of files to open
            month_range = range(1, 13) if split_by_month else [None]
            filepath_list = [
                self._get_raw_filepath(
                    country_name=country_name,
                    country_iso3=country_iso3,
                    version=version,
                    year=year,
                    month=month,
                    leadtime=leadtime,
                )
                for year in range(self.year_min, self.year_max + 1)
                for month in month_range
            ]
            # Read in both the control and ensemble perturbed forecast and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_ensemble_and_perturbed_datasets(
                filepath_list=filepath_list
            )
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = _get_station_dataset(
                stations=stations, ds=ds, coord_names=["number", "time"]
            )
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_name=country_name,
                country_iso3=country_iso3,
                version=version,
                ds=ds_new,
                leadtime=leadtime,
            )


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


def _get_station_dataset(
    stations: Dict[str, Station], ds: xr.Dataset, coord_names: List[str]
) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            station_name: (
                coord_names,
                ds.isel(
                    longitude=np.abs(ds.longitude - station.lon).argmin(),
                    latitude=np.abs(ds.latitude - station.lat).argmin(),
                )["dis24"],
            )
            for station_name, station in stations.items()
        },
        coords={coord_name: ds[coord_name] for coord_name in coord_names},
    )
